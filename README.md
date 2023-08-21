# OpenAI adapter for Vertex AI models

The server provides `{project_id}/chat/completions`, `{project_id}/completions` and `/models` endpoint compatible with those of OpenAI API.

## Installation

```sh
make all
```

## Configuration

Copy `.env.example` to `.env` and fill the gaps with Google credentials:

```
GOOGLE_APPLICATION_CREDENTIALS=<json with private key>
```

## Running server locally

Run the server:

```sh
make server-run
```

Open `localhost:5001/docs` to make sure the server is up and running.

Run the client:

```sh
make client-run
```

Select the Vertex AI model and chat with the model.

## Docker

Build the image:

```sh
make docker-build
```

Run the image:

```sh
make docker-build
```

Open `localhost:5001/docs` to make sure the server is up and running.

Run the client:

```sh
make client-run
```

## Dev

Run linters and formatters before committing:

```sh
make lint
make format
```

## Running tests

```sh
make test
```
