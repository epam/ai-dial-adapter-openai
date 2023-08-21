# PaLM 2 models

## Chat

[chat-bison](https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/text-chat)

```
Max input token: 4,096
Max output tokens: 1,024
Training data: Up to Feb 2023
Max turns : 2,500
```

The response has `metadata` field containing the token information.

The meaning of the field is **undocumented**. Moreover this field wasn't event present in the response a few week ago.

```json
{
  "predictions": ...,
  "metadata": {
    "tokenMetadata": {
      "inputTokenCount": {
        "totalTokens": 54,
        "totalBillableCharacters": 182
      },
      "outputTokenCount": {
        "totalBillableCharacters": 44,
        "totalTokens": 12
      }
    }
  }
}
```

Reverse engineered:

```
inputTokenCount.totalBillableCharacters = (request.context + request.messages.*.content).Filter(isVisibleSymbol).Length

outputTokenCount.totalBillableCharacters = (response.content).Filter(isVisibleSymbol).Length
```

The tokenizer is unknown.

## Text completion

[PaLM 2 for Chat](https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/chat-bison)

```

Max input token: 8,192
Max output tokens: 1,024
Training data: Up to Feb 2023

```

# Python SDKs

**PaLM API** and **Vertex AI** are two different things and have to different Python SDKs (https://developers.generativeai.google/)

## Vertex AI

Authorization is via GCP credentials (e.g. [ADC](https://cloud.google.com/docs/authentication/application-default-credentials)). That's what we use in the adapter.

Cons: the API doesn't provide information about token count _(or this information is undocumented)_.

### vertex-ai package

Docs: https://cloud.google.com/vertex-ai/docs/start/introduction-unified-platform

## PaLM API

Authorization is via [PaLM API key](https://developers.generativeai.google/tutorials/setup) ([see also](https://cloud.google.com/docs/authentication/api-keys)).

### google-generativeai package (high-level)

https://developers.generativeai.google/api/python/google/generativeai

It's possible to get model info (like input/output token limit) using `ModelService` API.

### google-ai-generativelanguage package (low-level)

https://developers.generativeai.google/api/python/google/ai/generativelanguage

The package a low-level auto-generated client library for the PaLM API.

### REST API

https://developers.generativeai.google/api/rest/generativelanguage

### Token counter endpoint

SDK: https://developers.generativeai.google/api/python/google/generativeai/count_message_tokens

REST: https://developers.generativeai.google/api/rest/generativelanguage/models/countMessageTokens
