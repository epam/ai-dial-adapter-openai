from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, Response
import logging
import json
import uvicorn
import os
import tiktoken
from openai_override import OpenAIException, OpenAIChatCompletion
import openai
from uuid import uuid4
from time import time

app = FastAPI()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler()]
)
model_aliases = json.loads(os.environ.get('MODEL_ALIASES', '{}'))

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def generate_chunk(data):
    return 'data: ' + json.dumps(data, separators=(',', ':')) + '\n\n'

async def generate_stream(messages, response, model, deployment):
    encoding = tiktoken.encoding_for_model(model)

    prompt_tokens = 3
    tokens_per_message = 4 if model == 'gpt-3.5-turbo' else 3
    for message in messages:
        prompt_tokens += tokens_per_message

        for key, value in message.items():
            prompt_tokens += len(encoding.encode(value))
            if key == "name":
                prompt_tokens += 1

    last_chunk = None
    is_stream_finished = False
    try:
        total_content = ''
        async for chunk in response:
            chunk_dict = chunk.to_dict_recursive()

            if chunk_dict['choices'][0]['finish_reason'] != None:
                is_stream_finished = True
                completion_tokens = len(encoding.encode(total_content))
                chunk_dict['usage'] = {
                    'completion_tokens': completion_tokens,
                    'prompt_tokens': prompt_tokens,
                    'total_tokens': prompt_tokens + completion_tokens
                }
            else:
                total_content += chunk_dict['choices'][0]['delta'].get('content', '')

            last_chunk = chunk_dict
            yield generate_chunk(chunk_dict)
    except OpenAIException as e:
        last_chunk = None
        yield e.body

    if not is_stream_finished:
        completion_tokens = len(encoding.encode(total_content))

        if last_chunk != None:
            logger.info('Don\'t received chunk with the finish reason')

            last_chunk['usage'] = {
                'completion_tokens': completion_tokens,
                'prompt_tokens': prompt_tokens,
                'total_tokens': prompt_tokens + completion_tokens
            }
            last_chunk['choices'][0]['delta']['content'] = ''
            last_chunk['choices'][0]['delta']['finish_reason'] = 'length'

            yield generate_chunk(chunk_dict)
        else:
            logger.info('Received 0 chunks')

            yield generate_chunk(
                {
                    "id": "chatcmpl-" + str(uuid4()),
                    "object": "chat.completion.chunk",
                    "created": str(int(time())),
                    "model": deployment,
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "length",
                            "delta": {}
                        }
                    ],
                    "usage": {
                        "completion_tokens": completion_tokens,
                        "prompt_tokens": prompt_tokens,
                        "total_tokens": prompt_tokens + completion_tokens
                    }
                }
            )

    yield 'data: [DONE]\n'

@app.post('/openai/deployments/{deployment_id}/chat/completions')
async def chat(deployment_id: str, request: Request):
    data = await request.json()

    is_stream = data.get('stream', False)
    messages = data['messages']
    dial_api_key = request.headers.get('X-UPSTREAM-KEY')
    api_base = request.headers.get('X-UPSTREAM-ENDPOINT')

    try:
        response = await OpenAIChatCompletion().acreate(
            engine=deployment_id,
            api_key=dial_api_key,
            api_base=api_base,
            **data
        )
    except OpenAIException as e:
        return Response(
            status_code=e.code,
            headers=e.headers,
            content=e.body
        )

    if is_stream:        
        return StreamingResponse(generate_stream(messages, response, model_aliases.get(deployment_id, deployment_id), deployment_id), media_type='text/event-stream')
    else:
        return response

@app.post('/openai/deployments/{deployment_id}/embeddings')
async def embedding(deployment_id: str, request: Request):
    data = await request.json()
    dial_api_key = request.headers.get('X-UPSTREAM-KEY')
    api_base = request.headers.get('X-UPSTREAM-ENDPOINT')

    return await openai.Embedding.acreate(
        deployment_id=deployment_id,
        api_key=dial_api_key,
        api_base=api_base,
        **data
    )


@app.get('/health')
def health():
    return {
        'status': 'ok'
    }

if __name__ == '__main__':
    uvicorn.run(app, port=5000)
