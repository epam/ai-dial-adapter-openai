from aidial_adapter_openai.configuration.deployment_type import (
    ChatCompletionDeploymentType,
)
from tests.integration_tests.chat_completion.test_case import (
    TestSuite,
    exclude_deployments,
)
from tests.utils.openai import (
    GET_WEATHER_FUNCTION,
    ai,
    ai_tools,
    function_to_tool,
    is_valid_tool_call,
    sys,
    tool_request,
    tool_response,
    user,
)


def create_fun_args(city: str):
    return {
        "location": city,
        "format": "celsius",
    }


def check_fun_args(city: str):
    return {
        "location": lambda s: city.lower() in s.lower(),
        "format": "celsius",
    }


def supports_parallel_tool_calls(deployment_type: ChatCompletionDeploymentType):
    return deployment_type not in [
        ChatCompletionDeploymentType.MISTRAL,
        ChatCompletionDeploymentType.DATABRICKS,
        ChatCompletionDeploymentType.GPT_TEXT_ONLY,
    ]


def supports_functions(deployment_type: ChatCompletionDeploymentType):
    return deployment_type not in [ChatCompletionDeploymentType.DATABRICKS]


@exclude_deployments(
    [
        ChatCompletionDeploymentType.DALLE3,
        ChatCompletionDeploymentType.MISTRAL,
    ]
)
def build_tools_common(s: TestSuite) -> None:
    if not s.supports_function_calling:
        return

    if s.supports_parallel_function_calling:
        city_config = [[("Glasgow", 15)], [("Glasgow", 15), ("London", 20)]]
    else:
        city_config = [[("Glasgow", 15)]]

    function = GET_WEATHER_FUNCTION
    tool = function_to_tool(function)
    fun_name = function["name"]
    for cities in city_config:
        city_names = [name for name, _ in cities]
        city_temps = [temp for _, temp in cities]

        test_name_suffix = "_".join(city_names)
        city_test_query = " and in ".join(city_names)
        query = f"What's the temperature in {city_test_query} in celsius?"

        init_messages = [user("2+3=?"), ai("5"), user(query)]
        if s.supports_system_prompt:
            init_messages.insert(0, sys("act as a helpful assistant"))

        def create_tool_call_id(idx: int):
            return f"{fun_name}_{idx+1}"

        s.test_case(
            name=f"weather tool {test_name_suffix}",
            messages=init_messages,
            tools=[tool],
            expected=lambda s, n=city_names: all(
                is_valid_tool_call(
                    s.tool_calls,
                    idx,
                    lambda idx: True,
                    fun_name,
                    check_fun_args(n[idx]),
                )
                for idx in range(len(n))
            )
            and s.response.choices[0].finish_reason == "tool_calls",
        )

        tool_reqs = ai_tools(
            [
                tool_request(
                    create_tool_call_id(idx),
                    fun_name,
                    create_fun_args(name),
                )
                for idx, (name, _) in enumerate(cities)
            ]
        )

        tool_resps = [
            tool_response(create_tool_call_id(idx), f"{temp} celsius")
            for idx, (_, temp) in enumerate(cities)
        ]

        # Databricks doesn't allow to continue chat after first tool call
        if s.deployment_type != ChatCompletionDeploymentType.DATABRICKS:
            s.test_case(
                name=f"weather tool followup {test_name_suffix}",
                messages=[*init_messages, tool_reqs, *tool_resps],
                tools=[tool],
                expected=lambda s, t=city_temps: s.content_contains_all(t),
            )
