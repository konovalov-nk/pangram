# Local AI Text Detector

A local ensemble of AI-text detectors for an RTX 5090, with a web UI, JSON API, and CLI.

By default, one request runs through six independent models:

- `iitolstykh/GigaCheck-Classifier-Multi`: EN/RU, 7.1B.
- `GeorgeDrayson/modernbert-ai-detection-raid-mage`: EN, 150M, context up to 8192 tokens.
- `Shushant/ADAL_AI_Detector`: EN, 355M, adversarial training.
- `pangram/editlens_roberta-large`: official Open Pangram, EN, 355M.
- `pangram/editlens_Llama-3.2-3B`: official Open Pangram, EN, 3B QLoRA INT4.
- `yaoandy107/greyscope-v2-qwen3.5-4b-int4`: EN/JA/ZH-TW, graded AI involvement, INT4.

The API returns each model's score separately. The final `consensus` is a majority vote at the selected threshold; probabilities are not averaged because the models are calibrated on different data.

This is not the production Pangram 4 model: that one is closed and available only through the hosted API. The ensemble includes official Open Pangram based on EditLens. Both Open Pangram models are English-only and licensed under CC BY-NC-SA 4.0, so non-commercial use only. The two Open Pangram variants correlate with each other and therefore cast two votes in the consensus.

## Docker Compose

Requirements: Docker Compose, NVIDIA Container Toolkit, an external Docker network `proxy` from the sibling `../caddy-proxy-network` project, and about 45 GB of free space for the first build and for downloading all weights.

Start the whole stack with one command: `make up` creates `.env`, builds the images, starts both containers, waits for them to become healthy, and warms up all models with the first request. Afterwards use `make logs`, `make warmup`, `make restart`, and `make down`.

Manual equivalent:

```bash
cp .env.example .env
docker compose up --build -d
```

The Compose project name is pinned as `pangram`. The container joins the external `proxy` network, does not publish a host port, and registers with `caddy-docker-proxy` under the `pangram` namespace with `caddy.wan=true`. Open <https://pangram.example.com> (replace the domain with your own).

Public HTTP/TLS forwarding for that name is registered in `../caddy-proxy-network/routes.yaml` and applied on the server with `make expose-route` from the proxy project.

Models are downloaded on the first request. GigaCheck, ModernBERT, ADAL, and both Open Pangram models run in `detector`, while Greyscope runs in a separate GPU sidecar because of incompatible Transformers requirements. `HF_CACHE_DIR` defaults to `~/.cache/huggingface`, so both containers share one cache.

Open Pangram checkpoints are gated on Hugging Face. Before the first run, accept the license on the `pangram/editlens_roberta-large` and `pangram/editlens_Llama-3.2-3B` pages, then run `hf auth login` on the host: the container uses the already downloaded cache and does not need a token at startup. The EditLens sources are not part of this repository; to inspect them, run `git clone https://github.com/pangramlabs/EditLens .vendor/editlens && git -C .vendor/editlens checkout 05a588f15d792330ccaf91be8ee4fdb54ce26835`.

## CLI

Stop the web service before running GigaCheck separately via the CLI, otherwise two copies of the 7B model will compete for VRAM.

```bash
docker compose stop detector
docker compose run --rm detector ai-detector --model all --file /data/document.txt
printf 'Long text...' | docker compose run --rm -T detector ai-detector --model gigacheck
docker compose up -d
```

For file input, add a read-only bind mount with your documents to `compose.yml`, for example `./documents:/data:ro`.

## API

```bash
curl https://pangram.example.com/api/detect \
  -H 'content-type: application/json' \
  -d '{"text":"Long text to analyze...","threshold":0.5}'
```

`POST /api/detect` returns `consensus` and a `detectors` array with independent scores, labels, and chunks for each model. Greyscope additionally returns `graded_label`, `bucket_probs`, `score_kind=ai_involvement`, and its calibrated binary threshold. Both Open Pangram models return `score_kind=ai_involvement`, `graded_label` (Human / Lightly / Moderately / AI-Generated), and `bucket_probs`. Documents longer than a model's context are not silently truncated: they are split into token-aware chunks.

## Verification

```bash
uv run --extra dev pytest
docker compose config --quiet
docker compose exec detector python -c \
  "import torch; print(torch.cuda.get_device_name(0))"
docker compose exec greyscope python -c \
  "import torch, transformers; print(torch.cuda.get_device_name(0), transformers.__version__)"
```

## Limitations

- Results are unstable on texts shorter than 100-150 words.
- Translation, paraphrasing, manual editing, and newer generators reduce accuracy.
- Even replacing paragraph breaks with spaces can significantly change some models' scores; pass the original formatting and treat the results as heuristics.
- The `0.5` threshold is not calibrated for any specific domain. For real use, build your own human/AI dataset and choose a threshold at an acceptable false-positive rate.
- Greyscope returns a degree of AI involvement, not a probability of misconduct or deception. Its bundled binary threshold `0.8369` is calibrated separately; the shared UI threshold is used only to keep the consensus consistent.
- Open Pangram/EditLens is English-only, trained on texts up to 1024 tokens, and licensed for non-commercial use only. Its score is the expected value of the bucket distribution, not a calibrated probability of misconduct.
- A score does not prove authorship and must not be the sole basis for sanctions.

## Why These Models

- [GigaCheck Classifier Multi](https://huggingface.co/iitolstykh/GigaCheck-Classifier-Multi): EN/RU, Apache-2.0; the authors report 98.46% mean accuracy and 97.93% TPR at 1% FPR on their test split. This is not an independent evaluation.
- [ModernBERT RAID/MAGE](https://huggingface.co/GeorgeDrayson/modernbert-ai-detection-raid-mage): English, Apache-2.0, a lightweight long-context model; training on RAID means the RAID leaderboard is not an independent evaluation.
- [ADAL](https://huggingface.co/Shushant/ADAL_AI_Detector): English, Apache-2.0, RoBERTa-large trained against paraphrase, synonym, homoglyph, and other evasion attacks.
- [Greyscope v2 INT4](https://huggingface.co/yaoandy107/greyscope-v2-qwen3.5-4b-int4): EN/JA/ZH-TW, Apache-2.0, an ordinal classifier for human/AI-edited/AI-generated text; its public benchmark numbers are not our independent evaluation.
- [Open Pangram: EditLens RoBERTa](https://huggingface.co/pangram/editlens_roberta-large) and [EditLens Llama](https://huggingface.co/pangram/editlens_Llama-3.2-3B): Pangram's official open weights, English, CC BY-NC-SA 4.0, a regression model for the degree of AI involvement; [source code](https://github.com/pangramlabs/EditLens).
- [Pangram](https://docs.pangram.com/): production Pangram 4 remains closed and is available only through the hosted API.

Model weight versions, Hugging Face remote code, and the GigaCheck and EditLens sources are pinned to specific commit SHAs for reproducibility and security. Model weights are not part of this repository.
