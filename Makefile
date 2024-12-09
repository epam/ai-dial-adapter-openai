PORT ?= 5001
IMAGE_NAME ?= ai-dial-adapter-openai
PLATFORM ?= linux/amd64
ARGS=

.PHONY: all install build serve clean lint format test integration_tests docker_build docker_run

all: build

install:
	poetry install

build: install
	poetry build

serve: install
	poetry run uvicorn "aidial_adapter_openai.app:app" --reload --host "0.0.0.0" --port $(PORT) --workers=1 --env-file ./.env

clean:
	poetry run clean
	poetry env remove --all

lint: install
	poetry run nox -s lint

format: install
	poetry run nox -s format

test: install
	poetry run nox -s test -- $(ARGS)

integration_test: install
	poetry run nox -s integration_test -- $(ARGS)

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
