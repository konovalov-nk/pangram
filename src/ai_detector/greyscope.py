from __future__ import annotations

import json
import re
import threading
import unicodedata
from pathlib import Path
from typing import Any

import torch

GREYSCOPE_MODEL = "yaoandy107/greyscope-v2-qwen3.5-4b-int4"
GREYSCOPE_REVISION = "41bcdf26fd06929ba56a5e4083eb5d55ef27db0b"

_BOILERPLATE_STARTS = ("Sure", "Here", "Abstract", "Title", "I'm happy to help", "Certainly")
_INVISIBLE = "\u00ad\u200b\u200c\u200d\u200e\u200f\u2060\u180e\ufeff"
_CONFUSABLES = {
    0x0410: "A", 0x0412: "B", 0x0415: "E", 0x041A: "K", 0x041C: "M",
    0x041D: "H", 0x041E: "O", 0x0420: "P", 0x0421: "C", 0x0422: "T",
    0x0423: "Y", 0x0425: "X", 0x0405: "S", 0x0406: "I", 0x0408: "J",
    0x0430: "a", 0x0435: "e", 0x043E: "o", 0x0440: "p", 0x0441: "c",
    0x0443: "y", 0x0445: "x", 0x0455: "s", 0x0456: "i", 0x0458: "j",
    0x0391: "A", 0x0392: "B", 0x0395: "E", 0x0396: "Z", 0x0397: "H",
    0x0399: "I", 0x039A: "K", 0x039C: "M", 0x039D: "N", 0x039F: "O",
    0x03A1: "P", 0x03A4: "T", 0x03A5: "Y", 0x03A7: "X", 0x03BF: "o",
    0x03C1: "p",
}
_TRANSLATE = {**_CONFUSABLES, **{ord(char): None for char in _INVISIBLE}}


def clean_text(text: str) -> str:
    import emoji

    text = unicodedata.normalize("NFKC", text.translate(_TRANSLATE))
    text = emoji.demojize(text)
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    paragraphs = [paragraph for paragraph in text.split("\n") if paragraph.strip()]
    if paragraphs:
        first = re.sub(r"^[^a-zA-Z0-9]*", "", paragraphs[0])
        first = emoji.replace_emoji(first, "")
        if any(first.startswith(phrase) for phrase in _BOILERPLATE_STARTS) and len(paragraphs) > 1:
            text = "\n".join(paragraphs[1:])
    return re.sub(r"\s+", " ", text.lower()).strip()


def decode_logits(logits: torch.Tensor, calibration: dict[str, Any]) -> dict[str, Any]:
    conditional = logits.float().sigmoid()
    cumulative = conditional.cumprod(dim=-1)
    scalar = float(cumulative.mean().item())
    low, high = calibration["score_min"], calibration["score_max"]
    score = min(max((scalar - low) / (high - low), 0.0), 1.0)

    upper = torch.cat((torch.ones(1), cumulative.cpu()))
    lower = torch.cat((cumulative.cpu(), torch.zeros(1)))
    probabilities = upper - lower
    if score < calibration["h_thresh"]:
        graded_label = "human"
    elif score > calibration["ai_thresh"]:
        graded_label = "AI-generated"
    else:
        graded_label = "AI-edited"
    return {
        "score": score,
        "graded_label": graded_label,
        "bucket_probs": {
            name: float(probability)
            for name, probability in zip(calibration["bucket_descriptions"], probabilities)
        },
    }


class GreyscopeDetector:
    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA GPU is required for Greyscope")
        self.device = torch.device("cuda:0")
        self.model: Any = None
        self.tokenizer: Any = None
        self.calibration: dict[str, Any] | None = None
        self.max_content_tokens = 0
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    def load(self) -> None:
        if self.model is not None:
            return
        with self._load_lock:
            if self.model is not None:
                return
            from huggingface_hub import hf_hub_download
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                GREYSCOPE_MODEL,
                revision=GREYSCOPE_REVISION,
            )
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"
            self.model = AutoModelForSequenceClassification.from_pretrained(
                GREYSCOPE_MODEL,
                revision=GREYSCOPE_REVISION,
                dtype=torch.bfloat16,
                device_map="cuda:0",
            ).eval()
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            calibration_path = hf_hub_download(
                GREYSCOPE_MODEL,
                "calibration.json",
                revision=GREYSCOPE_REVISION,
            )
            self.calibration = json.loads(Path(calibration_path).read_text())
            empty_prompt = self.calibration["prompt_template"].format(text="")
            prompt_tokens = len(self.tokenizer(empty_prompt, add_special_tokens=False)["input_ids"])
            self.max_content_tokens = self.calibration["max_length"] - prompt_tokens - 8

    def _chunks(self, text: str) -> list[tuple[int, int, int]]:
        offsets = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
        )["offset_mapping"]
        chunks = []
        token_start = 0
        while token_start < len(offsets):
            token_end = min(token_start + self.max_content_tokens, len(offsets))
            chunks.append(
                (offsets[token_start][0], offsets[token_end - 1][1], token_end - token_start)
            )
            if token_end == len(offsets):
                break
            token_start = token_end - min(96, self.max_content_tokens // 4)
        return chunks

    @torch.inference_mode()
    def detect(self, text: str, threshold: float = 0.5) -> dict[str, Any]:
        if not text.strip():
            raise ValueError("Text must not be empty")
        if not 0.0 < threshold < 1.0:
            raise ValueError("Threshold must be between 0 and 1")
        self.load()
        assert self.calibration is not None

        body = clean_text(text)
        if not body:
            raise ValueError("Text is empty after preprocessing")
        scored_chunks = []
        with self._inference_lock:
            for start, end, token_count in self._chunks(body):
                prompt = self.calibration["prompt_template"].format(text=body[start:end])
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.calibration["max_length"],
                    add_special_tokens=False,
                ).to(self.device)
                decoded = decode_logits(self.model(**inputs).logits[0], self.calibration)
                scored_chunks.append(
                    {
                        "start": start,
                        "end": end,
                        "token_count": token_count,
                        "ai_probability": round(decoded["score"], 6),
                        "label": "ai" if decoded["score"] >= threshold else "human",
                        "graded_label": decoded["graded_label"],
                        "bucket_probs": {
                            name: round(value, 6) for name, value in decoded["bucket_probs"].items()
                        },
                    }
                )

        total_tokens = sum(chunk["token_count"] for chunk in scored_chunks)
        score = (
            sum(chunk["ai_probability"] * chunk["token_count"] for chunk in scored_chunks)
            / total_tokens
        )
        bucket_probs = {
            name: round(
                sum(chunk["bucket_probs"][name] * chunk["token_count"] for chunk in scored_chunks)
                / total_tokens,
                6,
            )
            for name in self.calibration["bucket_descriptions"]
        }
        word_count = len(text.split())
        warnings = []
        if word_count < 100:
            warnings.append("Short texts are unreliable; use at least 100-150 words.")
        if any("\u0400" <= char <= "\u04ff" for char in text):
            warnings.append("greyscope does not support Russian; ignore its score for Russian text.")
        if threshold != self.calibration["binary_threshold"]:
            warnings.append(
                f"Consensus uses the selected threshold; Greyscope's calibrated binary threshold is "
                f"{self.calibration['binary_threshold']:.3f}."
            )

        if score < self.calibration["h_thresh"]:
            graded_label = "human"
        elif score > self.calibration["ai_thresh"]:
            graded_label = "AI-generated"
        else:
            graded_label = "AI-edited"
        return {
            "label": "ai" if score >= threshold else "human",
            "ai_probability": round(score, 6),
            "score_kind": "ai_involvement",
            "graded_label": graded_label,
            "bucket_probs": bucket_probs,
            "calibrated_binary_threshold": self.calibration["binary_threshold"],
            "threshold": threshold,
            "model": GREYSCOPE_MODEL,
            "backend": "greyscope",
            "languages": ["en", "ja", "zh-tw"],
            "word_count": word_count,
            "chunks": scored_chunks,
            "warnings": warnings,
            "disclaimer": "AI involvement is an estimate and must not be used as sole proof of authorship.",
        }

    def status(self) -> dict[str, Any]:
        return {
            "ready": self.model is not None,
            "backend": "greyscope",
            "model": GREYSCOPE_MODEL,
            "languages": ["en", "ja", "zh-tw"],
            "device": torch.cuda.get_device_name(0),
            "weight_size_gb": 3.22,
            "vram_allocated_gb": round(torch.cuda.memory_allocated(0) / 1024**3, 2),
        }
