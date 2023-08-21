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

all: $(BUILD)

$(BUILD): $(REQ)
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install -r $(REQ)
	echo "\033[31mActivate venv by running:\n> source $(BUILD)\033[0m"

.PHONY: all server-run client-run clean lint format test docker-build docker-run

server-run: $(BUILD)
	@source ./load_env.sh; load_env; \
	python ./debug_app.py --port=$(PORT)

client-run: $(BUILD)
	@source ./load_env.sh; load_env; \
	(PYTHONPATH=. python ./client/client_adapter.py)

clean:
	rm -rf $(VENV)
	find . -type f -name '*.pyc' -not -path './.venv/*' -delete

lint: $(BUILD)
	PYRIGHT_PYTHON_FORCE_VERSION=latest pyright
	flake8

format: $(BUILD)
	black . --exclude '(/\.venv/)'

# Add options "-s --log-cli-level=NOTSET" to pytest to see all logs
test: $(BUILD)
	@source ./load_env.sh; load_env; \
	pytest . -v --durations=0 -rA

docker-build: Dockerfile
	docker build --platform linux/amd64 -t $(IMAGE_NAME) .

docker-run: docker-build
	docker run --platform linux/amd64 --env-file ./.env -p $(PORT):5000 $(IMAGE_NAME)