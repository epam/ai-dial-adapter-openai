PORT ?= 5001
IMAGE_NAME ?= ai-dial-adapter-openai
PLATFORM ?= linux/amd64
VENV_DIR ?= .venv
POETRY ?= poetry
POETRY_PYTHON ?= python
PYDANTIC_V2 ?= 1
ARGS ?=

# Any non-empty CI value (even 'false' or '0') means that CI is enabled
CI ?=

.PHONY: all init_env install build serve lint format test integration_tests docker_build docker_run

-include .env.dev
export

all: build

init_env:
	$(if $(CI),,$(POETRY) env use $(POETRY_PYTHON))

install: init_env
	$(POETRY) install

build: install
	$(POETRY) build

serve: install
	$(POETRY) run uvicorn "aidial_adapter_openai.app:app" --reload --host "0.0.0.0" --port $(PORT) --workers=1 --env-file ./.env

lint: install
	$(POETRY) run nox -s lint

format: install
	$(POETRY) run nox -s format

test: install
	$(POETRY) run -- nox -s test -- $(ARGS)

integration_test: install
	$(POETRY) run -- nox -s integration_test -- $(ARGS)

install_git_hooks: install
	$(VENV_DIR)/bin/pre-commit install

docker_serve:
	docker build --platform $(PLATFORM) -t $(IMAGE_NAME):dev .
	docker run --platform $(PLATFORM) --env-file ./.env --rm -p $(PORT):5000 $(IMAGE_NAME):dev

help:
	@echo '===================='
	@echo 'build                        - build the source and wheels archives'
	@echo 'clean                        - clean virtual env and build artifacts'
	@echo 'install_git_hooks            - install the git hooks'
	@echo '-- LINTING --'
	@echo 'format                       - run code formatters'
	@echo 'lint                         - run linters'
	@echo '-- RUN --'
	@echo 'serve                        - run the dev server locally'
	@echo 'docker_serve                 - run the dev server from the docker'
	@echo '-- TESTS --'
	@echo 'test                         - run tests'
