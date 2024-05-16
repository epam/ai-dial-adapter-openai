import json

import requests
from openai.types.beta.function_tool_param import FunctionToolParam

from aidial_adapter_openai.utils.env import get_env

BING_SEARCH_URL = "https://api.bing.microsoft.com/v7.0/search"

search_bing_tool: FunctionToolParam = {
    "type": "function",
    "function": {
        "name": "search_bing",
        "description": "Searches bing to get up-to-date information from the web.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                }
            },
            "required": ["query"],
        },
    },
}


def search_bing(query: str) -> str:
    """
    Perform a bing search against the given query

    @param query: Search query
    @return: List of search results

    """
    headers = {"Ocp-Apim-Subscription-Key": get_env("BING_SUBSCRIPTION_KEY")}
    params = {"q": query, "textDecorations": False}
    response = requests.get(BING_SEARCH_URL, headers=headers, params=params)
    response.raise_for_status()
    search_results = response.json()

    output = []

    for result in search_results["webPages"]["value"]:
        output.append(
            {
                "title": result["name"],
                "link": result["url"],
                "snippet": result["snippet"],
            }
        )

    return json.dumps(output)
