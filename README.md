# OpenAI adapter for Vertex AI models

The server provides `{project_id}/chat/completions`, `{project_id}/completions` and `/models` endpoint compatible with those of OpenAI API.

## Installation

```sh
./install.sh
```

## Configuration

Create `.env` file and enter Google credentials:

```
GOOGLE_APPLICATION_CREDENTIALS=<json with private key>
```

## Running server locally

Run the server:

```sh
python ./app.py
```

Open `localhost:8080/docs` to make sure the server is up and running.

Run the client:

```sh
python ./client.py
```

Select the Vertex AI model and chat with the model.

## Docker

Build the image:

```sh
./build.sh
```

Run the image:

```sh
./run.sh
```

Open `localhost:8080/docs` to make sure the server is up and running.

Run the client:

```sh
python ./client.py
```

## Dev

Run linters before committing:

```sh
(pyright; flake8)
```
