PORT ?= 5001
IMAGE_NAME ?= ai-dial-adapter-openai
PLATFORM ?= linux/amd64
VENV_DIR ?= .venv
POETRY ?= $(VENV_DIR)/bin/poetry
POETRY_VERSION ?= 1.8.5
ARGS=

.PHONY: all init_env install build serve clean lint format test integration_tests docker_build docker_run

all: build

init_env:
	python -m venv $(VENV_DIR)
	$(VENV_DIR)/bin/pip install poetry==$(POETRY_VERSION) --quiet

install: init_env
	$(POETRY) install

build: install
	$(POETRY) build

serve: install
	$(POETRY) run uvicorn "aidial_adapter_openai.app:app" --reload --host "0.0.0.0" --port $(PORT) --workers=1 --env-file ./.env

clean:
	$(POETRY) run clean
	$(POETRY) env remove --all

lint: install
	$(POETRY) run nox -s lint

format: install
	$(POETRY) run nox -s format

test: install
	$(POETRY) run nox -s test -- $(ARGS)

integration_test: install
	$(POETRY) run nox -s integration_test -- $(ARGS)

docker_serve:
	docker build --platform $(PLATFORM) -t $(IMAGE_NAME):dev .
	docker run --platform $(PLATFORM) --env-file ./.env --rm -p $(PORT):5000 $(IMAGE_NAME):dev

help:
	@echo '===================='
	@echo 'build                        - build the source and wheels archives'
	@echo 'clean                        - clean virtual env and build artifacts'
	@echo '-- LINTING --'
	@echo 'format                       - run code formatters'
	@echo 'lint                         - run linters'
	@echo '-- RUN --'
	@echo 'serve                        - run the dev server locally'
	@echo 'docker_serve                 - run the dev server from the docker'
	@echo '-- TESTS --'
	@echo 'test                         - run tests'
