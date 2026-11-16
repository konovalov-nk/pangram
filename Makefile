.PHONY: up down logs ps warmup restart

up: .env
	docker compose up --build -d --wait
	@$(MAKE) --no-print-directory warmup

.env:
	cp .env.example .env

warmup:
	@echo "Loading all detector models into GPU memory (first request)..."
	@docker compose exec -T detector curl --silent --show-error --fail --max-time 1800 \
		http://127.0.0.1:8000/api/detect \
		-H 'content-type: application/json' \
		--data-binary @- < scripts/warmup-payload.json > /dev/null
	@echo "All models loaded and ready."

restart:
	docker compose restart
	@$(MAKE) --no-print-directory warmup

down:
	docker compose down

logs:
	docker compose logs -f

ps:
	docker compose ps
