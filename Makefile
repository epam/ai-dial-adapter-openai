ENV ?= DEV
PORT ?= 5001
IMAGE_NAME ?= vertex-ai-adapter

ifeq ($(ENV),DEV)
	REQ = requirements-dev.txt
else ifeq ($(ENV),PROD)
	REQ = requirements.txt
endif

VENV=.venv
BUILD=$(VENV)/bin/activate
SHELL := /bin/sh

all: $(BUILD)

$(BUILD): $(REQ)
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install -r $(REQ)
	@echo "\033[31mActivate venv by running:\n> source $(BUILD)\033[0m"

.PHONY: all server-run client-run clean lint format test docker-build docker-run

server-run: $(BUILD)
	( \
    	source $(VENV)/bin/activate; \
		source ./load_env.sh; load_env; \
		python -m debug_app --port=$(PORT); \
	)

client-run: $(BUILD)
	( \
    	source $(VENV)/bin/activate; \
		source ./load_env.sh; load_env; \
		python -m client.client_adapter; \
	)

clean:
	rm -rf $(VENV)
	find . -type f -name '*.pyc' -not -path './.venv/*' -delete

lint: $(BUILD)
	( \
    	source $(VENV)/bin/activate; \
		pyright; \
		flake8; \
	)
	$(MAKE) format ARGS="--check"

format: $(BUILD)
	( \
		source $(VENV)/bin/activate; \
		autoflake . $(ARGS); \
		isort . $(ARGS); \
		black . $(ARGS); \
	)

# Add options "-s --log-cli-level=NOTSET" to pytest to see all logs
test: $(BUILD)
	( \
		source $(VENV)/bin/activate; \
		source ./load_env.sh; load_env; \
		python -m pytest . -v --durations=0 -rA ; \
	)

docker-build: Dockerfile
	docker build --platform linux/amd64 -t $(IMAGE_NAME) .

docker-run: docker-build
	docker run --platform linux/amd64 --env-file ./.env -p $(PORT):5000 $(IMAGE_NAME)