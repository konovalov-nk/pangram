# Local AI Text Detector

Локальный ансамбль AI-text detectors для RTX 5090 с веб-интерфейсом, JSON API и CLI.

Один запрос по умолчанию проходит через шесть независимых моделей:

- `iitolstykh/GigaCheck-Classifier-Multi`: EN/RU, 7.1B.
- `GeorgeDrayson/modernbert-ai-detection-raid-mage`: EN, 150M, контекст до 8192 токенов.
- `Shushant/ADAL_AI_Detector`: EN, 355M, adversarial training.
- `pangram/editlens_roberta-large`: официальный Open Pangram, EN, 355M.
- `pangram/editlens_Llama-3.2-3B`: официальный Open Pangram, EN, 3B QLoRA INT4.
- `yaoandy107/greyscope-v2-qwen3.5-4b-int4`: EN/JA/ZH-TW, graded AI involvement, INT4.

API возвращает оценки каждой модели отдельно. Итоговый `consensus` считается большинством голосов при выбранном пороге; вероятности не усредняются, потому что модели откалиброваны на разных данных.

Это не production-модель Pangram 4: она закрыта и доступна только через hosted API. В ансамбль добавлен официальный Open Pangram на технологии EditLens. Обе модели English-only и распространяются по CC BY-NC-SA 4.0, то есть только для некоммерческого использования. Две версии Open Pangram коррелируют между собой, поэтому дают два голоса в consensus.

## Docker Compose

Требования: Docker Compose, NVIDIA Container Toolkit, external Docker network `proxy` из соседнего проекта `../caddy-proxy-network` и примерно 45 GB свободного места для первой сборки и загрузки всех весов.

Запуск всего стека одной командой: `make up` создаёт `.env`, собирает образы, поднимает оба контейнера, ждёт их готовности и прогревает все модели первым запросом. Дальше: `make logs`, `make warmup`, `make restart`, `make down`.

Эквивалент вручную:

```bash
cp .env.example .env
docker compose up --build -d
```

Compose project name (namespace) закреплён как `pangram`. Контейнер подключается к external network `proxy`, не публикует host-порт и регистрирует в `caddy-docker-proxy` namespace `pangram` с `caddy.wan=true`. Открыть <https://pangram.example.com> (замените домен на свой).

Public HTTP/TLS forwarding для этого имени регистрируется в `../caddy-proxy-network/routes.yaml` и применяется на сервере через `make expose-route` из proxy-проекта.

Модели загрузятся при первом запросе. GigaCheck, ModernBERT, ADAL и обе модели Open Pangram работают в `detector`, а Greyscope запускается в отдельном GPU-sidecar из-за несовместимых требований к Transformers. `HF_CACHE_DIR` по умолчанию указывает на `~/.cache/huggingface`, поэтому оба контейнера используют общий cache.

Чекпоинты Open Pangram gated на Hugging Face. Перед первым запуском примите лицензию на страницах `pangram/editlens_roberta-large` и `pangram/editlens_Llama-3.2-3B`, затем выполните `hf auth login` на хосте: контейнер использует уже скачанный cache и не требует токена при старте. Исходники EditLens в репозиторий не входят; для сверки: `git clone https://github.com/pangramlabs/EditLens .vendor/editlens && git -C .vendor/editlens checkout 05a588f15d792330ccaf91be8ee4fdb54ce26835`.

## CLI

Остановите веб-сервис перед отдельным CLI-запуском GigaCheck, иначе две копии 7B-модели будут конкурировать за VRAM.

```bash
docker compose stop detector
docker compose run --rm detector ai-detector --model all --file /data/document.txt
printf 'Long text...' | docker compose run --rm -T detector ai-detector --model gigacheck
docker compose up -d
```

Для файлов добавьте read-only bind mount с каталогом документов в `compose.yml`, например `./documents:/data:ro`.

## API

```bash
curl https://pangram.example.com/api/detect \
  -H 'content-type: application/json' \
  -d '{"text":"Long text to analyze...","threshold":0.5}'
```

`POST /api/detect` возвращает `consensus` и массив `detectors` с независимыми оценками, labels и chunks каждой модели. Для Greyscope дополнительно возвращаются `graded_label`, `bucket_probs`, `score_kind=ai_involvement` и его откалиброванный binary threshold. Для обеих моделей Open Pangram возвращаются `score_kind=ai_involvement`, `graded_label` (Human / Lightly / Moderately / AI-Generated) и `bucket_probs`. Документы длиннее контекста конкретной модели не обрезаются молча: они разбиваются на token-aware chunks.

## Проверка

```bash
uv run pytest
docker compose config --quiet
docker compose exec detector python -c \
  "import torch; print(torch.cuda.get_device_name(0))"
docker compose exec greyscope python -c \
  "import torch, transformers; print(torch.cuda.get_device_name(0), transformers.__version__)"
```

## Ограничения

- На текстах короче 100-150 слов результат нестабилен.
- Перевод, парафраз, ручная редактура и новые генераторы снижают качество.
- Даже замена абзацных переносов пробелами может сильно изменить score некоторых моделей; передавайте исходное форматирование и оценивайте результаты как эвристики.
- Порог `0.5` не откалиброван под конкретный домен. Для реального применения соберите собственный набор human/AI-текстов и выберите порог при допустимом false-positive rate.
- Greyscope возвращает степень AI involvement, а не вероятность нарушения или обмана. Его bundled binary threshold `0.8369` откалиброван отдельно; общий UI-порог используется только для единообразного consensus.
- Open Pangram/EditLens English-only, обучен на текстах до 1024 токенов и лицензирован только для некоммерческого использования. Его score — ожидаемое значение bucket-распределения, а не откалиброванная вероятность нарушения.
- Оценка не доказывает авторство и не должна быть единственным основанием для санкций.

## Почему эти модели

- [GigaCheck Classifier Multi](https://huggingface.co/iitolstykh/GigaCheck-Classifier-Multi): EN/RU, Apache-2.0, авторы сообщают mean accuracy 98.46% и TPR 97.93% при FPR 1% на своём test split. Это не независимая проверка.
- [ModernBERT RAID/MAGE](https://huggingface.co/GeorgeDrayson/modernbert-ai-detection-raid-mage): English, Apache-2.0, лёгкая long-context модель; обучение на RAID означает, что RAID leaderboard не является независимой проверкой.
- [ADAL](https://huggingface.co/Shushant/ADAL_AI_Detector): English, Apache-2.0, RoBERTa-large с обучением против paraphrase, synonym, homoglyph и других evasion attacks.
- [Greyscope v2 INT4](https://huggingface.co/yaoandy107/greyscope-v2-qwen3.5-4b-int4): EN/JA/ZH-TW, Apache-2.0, ordinal classifier для human/AI-edited/AI-generated; публичные benchmark numbers не являются нашей независимой проверкой.
- [Open Pangram: EditLens RoBERTa](https://huggingface.co/pangram/editlens_roberta-large) и [EditLens Llama](https://huggingface.co/pangram/editlens_Llama-3.2-3B): официальные открытые веса Pangram, English, CC BY-NC-SA 4.0, регрессия степени AI involvement; [исходники](https://github.com/pangramlabs/EditLens).
- [Pangram](https://docs.pangram.com/): production Pangram 4 остаётся закрытым и доступен только через hosted API.

Версии model weights, Hugging Face remote code и исходников GigaCheck и EditLens закреплены на конкретных commit SHA для воспроизводимости и безопасности.
