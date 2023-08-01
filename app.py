from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, Response, JSONResponse
import logging
import json
import uvicorn
import os
import tiktoken
from openai_override import OpenAIException, OpenAIChatCompletion, OpenAIEmbedding
from uuid import uuid4
from time import time
from json import JSONDecodeError
from openai import error

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

def calculate_prompt_tokens(messages, model, encoding):
    prompt_tokens = 3
    tokens_per_message = 4 if model == 'gpt-3.5-turbo' else 3
    
    for message in messages:
        prompt_tokens += tokens_per_message

        for key, value in message.items():
            prompt_tokens += len(encoding.encode(value))
            if key == "name":
                prompt_tokens += 1
    
    return prompt_tokens

async def generate_stream(request_id, messages, response, model, deployment):
    encoding = tiktoken.encoding_for_model(model)

    prompt_tokens = calculate_prompt_tokens(messages, model, encoding)

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
            # logger.info('Chunk ' + request_id + ': ' + str(chunk_dict))
            yield generate_chunk(chunk_dict)
    except OpenAIException as e:
        yield e.body
        yield 'data: [DONE]\n'
        return

    if not is_stream_finished:
        completion_tokens = len(encoding.encode(total_content))

        if last_chunk != None:
            # logger.info('Don\'t received chunk with the finish reason')

            last_chunk['usage'] = {
                'completion_tokens': completion_tokens,
                'prompt_tokens': prompt_tokens,
                'total_tokens': prompt_tokens + completion_tokens
            }
            last_chunk['choices'][0]['delta']['content'] = ''
            last_chunk['choices'][0]['delta']['finish_reason'] = 'length'

            yield generate_chunk(chunk_dict)
        else:
            # logger.info('Received 0 chunks')

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

    # logger.info('Response ' + request_id + ': ' + total_content)

    yield 'data: [DONE]\n'

async def get_data_or_generate_error(request):
    try:
        data = await request.json()
    except JSONDecodeError as e:
        return JSONResponse(
            status_code=400,
            content={
                'error': {
                    'message': 'Your request contained invalid JSON: ' + str(e),
                    'type': 'invalid_request_error',
                    'param': None,
                    'code': None
                }
            }
        )

    if type(data) != dict:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": str(data) + " is not of type 'object'",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": None
                }
            }
        )
    
    return data


@app.post('/openai/deployments/{deployment_id}/chat/completions')
async def chat(deployment_id: str, request: Request):
    request_id = str(uuid4())
    data = await get_data_or_generate_error(request)

    if type(data) == JSONResponse:
        return data

    is_stream = data.get('stream', False)
    dial_api_key = request.headers.get('X-UPSTREAM-KEY')
    api_base = request.headers.get('X-UPSTREAM-ENDPOINT')

    # logger.info('Request ' + request_id + ': ' + str(data) + ' (base: ' + api_base + ', key: ' + dial_api_key[0:3] + '...' + dial_api_key[len(dial_api_key)-3:len(dial_api_key)] + ')')

    try:
        response = await OpenAIChatCompletion().acreate(
            engine=deployment_id,
            api_key=dial_api_key,
            api_base=api_base,
            request_timeout=(10, 600), # connect timeout and total timeout
            **data
        )
    except OpenAIException as e:
        return Response(
            status_code=e.code,
            headers=e.headers,
            content=e.body
        )
    except error.Timeout:
        return JSONResponse(
            status_code=504,
            content={
                'error': {
                    'message': 'Request timed out',
                    'type': 'timeout',
                    'param': None,
                    'code': None
                }
            }
        )

    if is_stream:
        messages = data['messages']
        return StreamingResponse(generate_stream(request_id, messages, response, model_aliases.get(deployment_id, deployment_id), deployment_id), media_type='text/event-stream')
    else:
        return response

@app.post('/openai/deployments/{deployment_id}/embeddings')
async def embedding(deployment_id: str, request: Request):
    data = await get_data_or_generate_error(request)

    if type(data) == JSONResponse:
        return data

    dial_api_key = request.headers.get('X-UPSTREAM-KEY')
    api_base = request.headers.get('X-UPSTREAM-ENDPOINT')

    try:
        return await OpenAIEmbedding().acreate(
            deployment_id=deployment_id,
            api_key=dial_api_key,
            api_base=api_base,
            request_timeout=(10, 600), # connect timeout and total timeout
            **data
        )
    except OpenAIException as e:
        return Response(
            status_code=e.code,
            headers=e.headers,
            content=e.body
        )
    except error.Timeout:
        return JSONResponse(
            status_code=504,
            content={
                'error': {
                    'message': 'Request timed out',
                    'type': 'timeout',
                    'param': None,
                    'code': None
                }
            }
        )

@app.get('/health')
def health():
    return {
        'status': 'ok'
    }

if __name__ == '__main__':
    uvicorn.run(app, port=5000)
