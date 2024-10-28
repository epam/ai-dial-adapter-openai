# OpenAI Adapter

## Overview

The project implements [AI DIAL API](https://epam-rail.com/dial_api) for language models from [Azure OpenAI](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models).

## Developer environment

This project uses [Python>=3.11](https://www.python.org/downloads/) and [Poetry>=1.6.1](https://python-poetry.org/) as a dependency manager.

Check out Poetry's [documentation on how to install it](https://python-poetry.org/docs/#installation) on your system before proceeding.

To install requirements:

```sh
poetry install
```

This will install all requirements for running the package, linting, formatting and tests.

### IDE configuration

The recommended IDE is [VSCode](https://code.visualstudio.com/).
Open the project in VSCode and install the recommended extensions.

The VSCode is configured to use PEP-8 compatible formatter [Black](https://black.readthedocs.io/en/stable/index.html).

Alternatively you can use [PyCharm](https://www.jetbrains.com/pycharm/).

Set-up the Black formatter for PyCharm [manually](https://black.readthedocs.io/en/stable/integrations/editors.html#pycharm-intellij-idea) or
install PyCharm>=2023.2 with [built-in Black support](https://blog.jetbrains.com/pycharm/2023/07/2023-2/#black).

## Run

Run the development server locally:

```sh
make serve
```

Run the server from Docker container:

```sh
make docker_serve
```

### Make on Windows

As of now, Windows distributions do not include the make tool. To run make commands, the tool can be installed using
the following command (since [Windows 10](https://learn.microsoft.com/en-us/windows/package-manager/winget/)):

```sh
winget install GnuWin32.Make
```

For convenience, the tool folder can be added to the PATH environment variable as `C:\Program Files (x86)\GnuWin32\bin`.
The command definitions inside Makefile should be cross-platform to keep the development environment setup simple.

## Environment Variables

Copy `.env.example` to `.env` and customize it for your environment.

### Categories of deployments

The following variables cluster all deployments into the groups of deployments which share the same API.

|Variable|Default|Description|
|---|---|---|
|DALLE3_DEPLOYMENTS|``|Comma-separated list of deployments that support DALL-E 3 API. Example: `dall-e-3,dalle3,dall-e`|
|DALLE3_AZURE_API_VERSION|2024-02-01|The version API for requests to Azure DALL-E-3 API|
|GPT4_VISION_DEPLOYMENTS|``|Comma-separated list of deployments that support GPT-4V API. Example: `gpt-4-vision-preview,gpt-4-vision`|
|GPT4_VISION_MAX_TOKENS|1024|Default value of `max_tokens` parameter for GPT-4V when it wasn't provided in the request|
|MISTRAL_DEPLOYMENTS|``|Comma-separated list of deployments that support Mistral Large Azure API. Example: `mistral-large-azure,mistral-large`|
|DATABRICKS_DEPLOYMENTS|``|Comma-separated list of Databricks chat completion deployments. Example: `databricks-dbrx-instruct,databricks-mixtral-8x7b-instruct,databricks-llama-2-70b-chat`|
|GPT4O_DEPLOYMENTS|``|Comma-separated list of GPT-4o chat completion deployments. Example: `gpt-4o-2024-05-13`|
|AZURE_AI_VISION_DEPLOYMENTS|``|Comma-separated list of Azure AI Vision embedding deployments. The endpoint of the deployment is expected point to the Azure service: `https://<service-name>.cognitiveservices.azure.com/`|

Deployments that do not fall into any of the categories are considered to support text-to-text chat completion OpenAI API or text embeddings OpenAI API.

### Other variables

|Variable|Default|Description|
|---|---|---|
|LOG_LEVEL|INFO|Log level. Use DEBUG for dev purposes and INFO in prod|
|WEB_CONCURRENCY|1|Number of workers for the server|
|MODEL_ALIASES|`{}`|Mapping from the request deployment id to [model name of tiktoken](https://github.com/openai/tiktoken/blob/main/tiktoken/model.py). Required for the token calculation on the adapter side. Example: `{"my-gpt-deployment":"gpt-3.5-turbo-0301"}`|
|DIAL_USE_FILE_STORAGE|False|Save image model artifacts to DIAL File storage (DALL-E images are uploaded to the DIAL file storage and its base64 encodings are replaced with links to the storage)|
|DIAL_URL||URL of the core DIAL server (required when DIAL_USE_FILE_STORAGE=True)|
|NON_STREAMING_DEPLOYMENTS|``|Comma-separated list of deployments which do not support streaming. The adapter is going to emulate the streaming by calling the model and converting its response into a single-chunk stream. Example: `o1-mini`, `o1-preview`|
|ACCESS_TOKEN_EXPIRATION_WINDOW|10|The Azure access token is renewed this many seconds before its actual expiration time. The buffer ensures that the token does not expire in the middle of an operation due to processing time and potential network delays.|
|AZURE_OPEN_AI_SCOPE|https://cognitiveservices.azure.com/.default|Provided scope of access token to Azure OpenAI services|
|API_VERSIONS_MAPPING|`{}`|The mapping of versions API for requests to Azure OpenAI API. Example: `{"2023-03-15-preview": "2023-05-15", "": "2024-02-15-preview"}`. An empty key sets the default api version for the case when the user didn't pass it in the request|
|ELIMINATE_EMPTY_CHOICES|False|When enabled, the response stream is guaranteed to exclude chunks with an empty list of choices. This is useful when a DIAL client doesn't support such chunks. An empty list of choices can be generated by Azure OpenAI in at least two cases: (1) when the **Content filter** is not disabled, Azure includes [prompt filter results](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/content-filter?tabs=warning%2Cuser-prompt%2Cpython-new#prompt-annotation-message) in the first chunk with an empty list of choices; (2) when `stream_options.include_usage` is enabled, the last chunk contains usage data and an empty list of choices. This variable replaces the deprecated `FIX_STREAMING_ISSUES_IN_NEW_API_VERSIONS` which served the same function.|
|CORE_API_VERSION||Supported value `0.6` to work with the old version of the DIAL File API|

## Lint

Run the linting before committing:

```sh
make lint
```

To auto-fix formatting issues run:

```sh
make format
```

## Test

Run unit tests locally:

```sh
make test
```

## Clean

To remove the virtual environment and build artifacts:

```sh
make clean
```
