from __future__ import annotations

import json
import os
import re
import sys
import threading
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import emoji
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

GIGACHECK_MODEL = "iitolstykh/GigaCheck-Classifier-Multi"
GIGACHECK_REVISION = "1b885318d0317a7c7138a52b3b53e0c747604da4"
MODERNBERT_MODEL = "GeorgeDrayson/modernbert-ai-detection-raid-mage"
MODERNBERT_REVISION = "e047d9c56f51a7e5c5ef5b0dd556f519bb9dd624"
ADAL_MODEL = "Shushant/ADAL_AI_Detector"
ADAL_REVISION = "b598375d7fb2e8926e5514b15a68b2819bc31a0a"
EDITLENS_ROBERTA_MODEL = "pangram/editlens_roberta-large"
EDITLENS_ROBERTA_REVISION = "f93e1ace74528cfb48f337ab2fe946fb71a728cb"
EDITLENS_LLAMA_MODEL = "pangram/editlens_Llama-3.2-3B"
EDITLENS_LLAMA_REVISION = "b4a297be1d4c36a2a02c7709ae12f3c6626fae58"
LLAMA_BASE_MODEL = "meta-llama/Llama-3.2-3B"
LLAMA_BASE_REVISION = "13afe5124825b4f3751f836b40dafda64c1ed062"
EDITLENS_BACKENDS = ("editlens-roberta", "editlens-llama")
LOCAL_BACKENDS = ("gigacheck", "modernbert", "adal", *EDITLENS_BACKENDS)
DEFAULT_BACKENDS = (*LOCAL_BACKENDS, "greyscope")

EDITLENS_BUCKET_LABELS = (
    "Human",
    "Lightly AI-Assisted",
    "Moderately AI-Assisted",
    "AI-Generated",
)
BOILERPLATE_STARTS = (
    "Sure",
    "Here",
    "Abstract",
    "Title",
    "I'm happy to help",
    "Certainly",
)


class NormedLinear(torch.nn.Module):
    """Linear head preceded by LayerNorm, as used by EditLens QLoRA checkpoints."""

    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        super().__init__()
        self.norm = torch.nn.LayerNorm(hidden_size, device=device, dtype=dtype)
        self.linear = torch.nn.Linear(
            hidden_size, num_labels, bias=False, device=device, dtype=dtype
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Keep the small classification head in float32; base-model hidden
        # states may be fp16, bf16, or fp32 depending on quantization settings.
        return self.linear(self.norm(hidden_states.float()))


def clean_editlens_text(text: str) -> str:
    """Mirror the preprocessing used by the official EditLens inference script."""
    text = emoji.demojize(text)
    if "</think>" in text:
        text = text.split("</think>")[1].strip()

    paragraphs = [paragraph for paragraph in text.split("\n") if paragraph.strip()]
    if paragraphs:
        first = re.sub(r"^[^a-zA-Z0-9]*", "", paragraphs[0])
        first = emoji.replace_emoji(first, "")
        if len(paragraphs) > 1 and first.startswith(BOILERPLATE_STARTS):
            text = "\n".join(paragraphs[1:])

    return re.sub(r"\s+", " ", text.lower()).strip()


def editlens_decode(logits: torch.Tensor) -> tuple[float, dict[str, Any]]:
    """Convert EditLens bucket logits into a 0-1 AI-involvement score."""
    if logits.ndim == 2:
        logits = logits[0]
    probabilities = logits.float().softmax(dim=-1)
    n_buckets = probabilities.numel()
    if n_buckets < 2:
        raise ValueError("EditLens checkpoint must have at least two buckets")

    weights = torch.arange(
        n_buckets, dtype=probabilities.dtype, device=probabilities.device
    )
    score = float((probabilities * weights).sum() / (n_buckets - 1))
    labels = (
        EDITLENS_BUCKET_LABELS
        if n_buckets == len(EDITLENS_BUCKET_LABELS)
        else tuple(f"bucket_{index}" for index in range(n_buckets))
    )
    bucket_probs = {
        label: round(float(probability), 6)
        for label, probability in zip(labels, probabilities.tolist())
    }
    graded_label = max(bucket_probs, key=lambda label: bucket_probs[label])
    return score, {
        "score_kind": "ai_involvement",
        "graded_label": graded_label,
        "bucket_probs": bucket_probs,
    }


@dataclass(frozen=True)
class TextChunk:
    start: int
    end: int
    token_count: int


def chunk_by_tokens(
    text: str,
    tokenizer: Any,
    max_tokens: int,
    overlap: int = 96,
) -> list[TextChunk]:
    if not text:
        return []
    encoding = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )
    offsets = encoding["offset_mapping"]
    if not offsets:
        return []

    chunks: list[TextChunk] = []
    token_start = 0
    while token_start < len(offsets):
        token_end = min(token_start + max_tokens, len(offsets))
        start = offsets[token_start][0]
        end = offsets[token_end - 1][1]
        chunks.append(TextChunk(start=start, end=end, token_count=token_end - token_start))
        if token_end == len(offsets):
            break
        token_start = token_end - min(overlap, max_tokens // 4)
    return chunks


class LocalDetector:
    def __init__(self, backend: str) -> None:
        self.backend = backend.lower()
        if self.backend not in LOCAL_BACKENDS:
            raise ValueError(f"Unknown detector backend: {self.backend}")
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA GPU is required for this setup")

        self.device = torch.device("cuda:0")
        self.model: Any = None
        self.tokenizer: Any = None
        self.model_id = ""
        self.languages: list[str] = []
        self.max_content_tokens = 0
        self.max_length = 0
        self.ai_label_id = 0
        self.n_buckets = 0
        self.score_kind = "probability"
        self.weight_size_gb = 0.0
        self._inference_lock = threading.Lock()
        self._load()
        self.weight_size_gb = sum(
            parameter.numel() * parameter.element_size() for parameter in self.model.parameters()
        ) / 1024**3

    def _load(self) -> None:
        if self.backend == "gigacheck":
            self._load_gigacheck()
        elif self.backend == "modernbert":
            self._load_classifier(
                MODERNBERT_MODEL, MODERNBERT_REVISION, ai_label_id=1, max_tokens=8190
            )
        elif self.backend == "adal":
            self._load_classifier(
                ADAL_MODEL, ADAL_REVISION, ai_label_id=0, max_tokens=510
            )
        elif self.backend == "editlens-roberta":
            self._load_editlens_roberta()
        else:
            self._load_editlens_llama()

    def _load_gigacheck(self) -> None:
        vendor = Path(__file__).resolve().parents[2] / ".vendor" / "gigacheck"
        if not vendor.exists():
            raise RuntimeError("GigaCheck source is missing from .vendor/gigacheck")
        sys.path.insert(0, str(vendor))

        self.model_id = GIGACHECK_MODEL
        self.languages = ["en", "ru"]
        self.model = AutoModel.from_pretrained(
            GIGACHECK_MODEL,
            revision=GIGACHECK_REVISION,
            trust_remote_code=True,
            device_map="cuda:0",
            torch_dtype=torch.bfloat16,
        )
        # GigaCheck's train() omits PyTorch's `return self`, so eval() returns None.
        self.model.eval()
        self.tokenizer = self.model.tokenizer
        self.max_content_tokens = int(self.model.config.max_length) - 2
        self.max_length = self.max_content_tokens + 2

    def _load_classifier(
        self,
        model_id: str,
        revision: str,
        ai_label_id: int,
        max_tokens: int,
    ) -> None:
        self.model_id = model_id
        self.languages = ["en"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=torch.bfloat16,
        ).to(self.device).eval()
        self.ai_label_id = ai_label_id
        self.max_content_tokens = max_tokens
        self.max_length = max_tokens + 2

    def _load_editlens_roberta(self) -> None:
        self.model_id = EDITLENS_ROBERTA_MODEL
        self.languages = ["en"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            EDITLENS_ROBERTA_MODEL,
            revision=EDITLENS_ROBERTA_REVISION,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            EDITLENS_ROBERTA_MODEL,
            revision=EDITLENS_ROBERTA_REVISION,
            torch_dtype=torch.bfloat16,
        ).to(self.device).eval()
        self.n_buckets = int(self.model.config.num_labels)
        self.max_content_tokens = 510
        self.max_length = 512
        self.score_kind = "ai_involvement"

    def _load_editlens_llama(self) -> None:
        from peft import PeftModel
        from transformers import BitsAndBytesConfig

        self.model_id = EDITLENS_LLAMA_MODEL
        self.languages = ["en"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            LLAMA_BASE_MODEL,
            revision=LLAMA_BASE_REVISION,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base_model = AutoModelForSequenceClassification.from_pretrained(
            LLAMA_BASE_MODEL,
            revision=LLAMA_BASE_REVISION,
            num_labels=len(EDITLENS_BUCKET_LABELS),
            quantization_config=quantization_config,
            device_map={"": 0},
        )
        base_model.config.pad_token_id = self.tokenizer.pad_token_id
        if hasattr(base_model, "score") and isinstance(
            base_model.score, torch.nn.Linear
        ):
            base_model.score = NormedLinear(
                base_model.config.hidden_size,
                len(EDITLENS_BUCKET_LABELS),
                device=next(base_model.parameters()).device,
            )
        self.model = PeftModel.from_pretrained(
            base_model,
            EDITLENS_LLAMA_MODEL,
            revision=EDITLENS_LLAMA_REVISION,
        ).eval()
        self.n_buckets = len(EDITLENS_BUCKET_LABELS)
        self.max_content_tokens = 1022
        self.max_length = 1024
        self.score_kind = "ai_involvement"

    def detect(self, text: str, threshold: float = 0.5) -> dict[str, Any]:
        text = text.strip()
        if not text:
            raise ValueError("Text must not be empty")
        if not 0.0 < threshold < 1.0:
            raise ValueError("Threshold must be between 0 and 1")

        chunks = chunk_by_tokens(text, self.tokenizer, self.max_content_tokens)
        scored_chunks = []
        aggregated_buckets: dict[str, float] = {}
        with self._inference_lock, torch.inference_mode():
            for chunk in chunks:
                chunk_text = text[chunk.start : chunk.end]
                score, details = self._score(chunk_text)
                scored_chunks.append(
                    {
                        **asdict(chunk),
                        "ai_probability": round(score, 6),
                        "label": "ai" if score >= threshold else "human",
                        **details,
                    }
                )
                for label, probability in details.get("bucket_probs", {}).items():
                    aggregated_buckets[label] = (
                        aggregated_buckets.get(label, 0.0)
                        + probability * chunk.token_count
                    )

        if not scored_chunks:
            raise ValueError("Text produced no tokens")
        total_tokens = sum(chunk["token_count"] for chunk in scored_chunks)
        probability = sum(
            chunk["ai_probability"] * chunk["token_count"] for chunk in scored_chunks
        ) / total_tokens
        word_count = len(text.split())
        warnings = []
        if word_count < 100:
            warnings.append("Short texts are unreliable; use at least 100-150 words.")
        if self.languages == ["en"] and any("\u0400" <= char <= "\u04ff" for char in text):
            warnings.append(f"{self.backend} is English-only; ignore its score for Russian text.")
        invisible_count = sum(text.count(char) for char in ("\u200b", "\u200c", "\u200d", "\ufeff"))
        if invisible_count:
            warnings.append(f"Found {invisible_count} invisible Unicode characters.")

        result = {
            "label": "ai" if probability >= threshold else "human",
            "ai_probability": round(probability, 6),
            "threshold": threshold,
            "model": self.model_id,
            "backend": self.backend,
            "languages": self.languages,
            "word_count": word_count,
            "chunks": scored_chunks,
            "warnings": warnings,
            "disclaimer": "This score is probabilistic and must not be used as sole proof of authorship.",
        }
        if aggregated_buckets:
            bucket_probs = {
                label: round(value / total_tokens, 6)
                for label, value in aggregated_buckets.items()
            }
            result["score_kind"] = "ai_involvement"
            result["bucket_probs"] = bucket_probs
            result["graded_label"] = max(
                bucket_probs, key=lambda label: bucket_probs[label]
            )
        return result

    def _score(self, text: str) -> tuple[float, dict[str, Any]]:
        if self.backend == "gigacheck":
            output = self.model([text])
            ai_label_id = next(
                label_id
                for label_id, label in self.model.config.id2label.items()
                if label.lower() == "ai"
            )
            return float(output.classification_head_probs[0, ai_label_id].item()), {}

        if self.backend in EDITLENS_BACKENDS:
            cleaned = clean_editlens_text(text) or text.strip()
            inputs = self.tokenizer(
                cleaned,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)
            logits = self.model(**inputs).logits
            return editlens_decode(logits)

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        probabilities = self.model(**inputs).logits.float().softmax(dim=-1)[0]
        return float(probabilities[self.ai_label_id].item()), {}

    def status(self) -> dict[str, Any]:
        return {
            "ready": True,
            "backend": self.backend,
            "model": self.model_id,
            "languages": self.languages,
            "score_kind": self.score_kind,
            "device": torch.cuda.get_device_name(0),
            "weight_size_gb": round(self.weight_size_gb, 2),
        }


class RemoteDetector:
    def __init__(self, backend: str, url: str) -> None:
        self.backend = backend
        self.url = url.rstrip("/")
        self.timeout = float(os.getenv("GREYSCOPE_TIMEOUT", "900"))

    def _request(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(
            f"{self.url}{path}",
            data=data,
            headers={"content-type": "application/json"} if data else {},
            method="POST" if data else "GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            try:
                detail = json.load(error).get("detail", error.reason)
            except (json.JSONDecodeError, AttributeError):
                detail = error.reason
            raise RuntimeError(f"{self.backend} failed: {detail}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(
                f"{self.backend} is unavailable at {self.url}: {error.reason}"
            ) from error

    def detect(self, text: str, threshold: float = 0.5) -> dict[str, Any]:
        return self._request("/api/detect", {"text": text, "threshold": threshold})

    def status(self) -> dict[str, Any]:
        status = self._request("/api/status")
        status.setdefault("backend", self.backend)
        return status


def consensus_for(results: list[dict[str, Any]]) -> dict[str, Any]:
    ai_votes = sum(result["label"] == "ai" for result in results)
    human_votes = len(results) - ai_votes
    if ai_votes == human_votes:
        label = "uncertain"
    else:
        label = "ai" if ai_votes > human_votes else "human"
    return {
        "label": label,
        "ai_votes": ai_votes,
        "human_votes": human_votes,
        "total": len(results),
        "agreement": "unanimous" if ai_votes in {0, len(results)} else "majority",
    }


class DetectorEnsemble:
    def __init__(self, backends: list[str] | None = None) -> None:
        configured = backends or [
            backend.strip()
            for backend in os.getenv("DETECTOR_MODELS", ",".join(DEFAULT_BACKENDS)).split(",")
            if backend.strip()
        ]
        if not configured:
            raise ValueError("DETECTOR_MODELS must contain at least one backend")
        if len(configured) != len(set(configured)):
            raise ValueError("DETECTOR_MODELS must not contain duplicates")
        unknown = set(configured) - set(DEFAULT_BACKENDS)
        if unknown:
            raise ValueError(f"Unknown detector backend: {', '.join(sorted(unknown))}")
        self.detectors = [create_detector(backend) for backend in configured]
        self._inference_lock = threading.Lock()

    def detect(self, text: str, threshold: float = 0.5) -> dict[str, Any]:
        with self._inference_lock:
            results = [detector.detect(text, threshold) for detector in self.detectors]
        return {
            "consensus": consensus_for(results),
            "detectors": results,
            "word_count": results[0]["word_count"],
            "threshold": threshold,
            "disclaimer": "Detector scores are probabilistic and must not be used as sole proof of authorship.",
        }

    def status(self) -> dict[str, Any]:
        models = [detector.status() for detector in self.detectors]
        return {
            "ready": all(model["ready"] for model in models),
            "models": models,
            "device": torch.cuda.get_device_name(0),
            "vram_allocated_gb": round(torch.cuda.memory_allocated(0) / 1024**3, 2),
        }


def create_detector(backend: str) -> LocalDetector | RemoteDetector:
    if backend == "greyscope":
        return RemoteDetector(backend, os.getenv("GREYSCOPE_URL", "http://greyscope:8001"))
    return LocalDetector(backend)
