from openai._base_client import AsyncHttpxClientWrapper

http_client = AsyncHttpxClientWrapper(follow_redirects=True)
