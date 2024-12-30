from aidial_adapter_openai.constant import ChatCompletionDeploymentType
from tests.integration_tests.base import TestSuite, exclude_deployments
from tests.utils.openai import (
    GET_WEATHER_FUNCTION,
    ai,
    ai_function,
    ai_tools,
    function_request,
    function_response,
    function_to_tool,
    is_valid_function_call,
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
    return deployment_type not in [
        ChatCompletionDeploymentType.DATABRICKS,
    ]


@exclude_deployments(
    [
        ChatCompletionDeploymentType.GPT4_VISION,
        ChatCompletionDeploymentType.DALLE3,
        ChatCompletionDeploymentType.MISTRAL,
    ]
)
def build_tools_common(s: TestSuite) -> None:
    if supports_parallel_tool_calls(s.deployment_type):
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
        init_messages = [
            sys("act as a helpful assistant"),
            user("2+3=?"),
            ai("5"),
            user(query),
        ]
        if supports_functions(s.deployment_type):
            # Functions
            s.test_case(
                name=f"weather function {test_name_suffix}",
                messages=init_messages,
                functions=[function],
                expected=lambda s, n=city_names[0]: is_valid_function_call(
                    s.function_call, fun_name, check_fun_args(n)
                ),
            )

            function_req = ai_function(
                function_request(fun_name, create_fun_args(city_names[0]))
            )
            function_resp = function_response(
                fun_name, f"{city_temps[0]} celsius"
            )

            if len(cities) == 1:
                s.test_case(
                    name=f"weather_function_followup_{test_name_suffix}",
                    messages=[
                        *init_messages,
                        function_req,
                        function_resp,
                    ],
                    functions=[function],
                    expected=lambda s, t=city_temps[0]: s.content_contains_all(
                        [t]
                    ),
                )
            else:
                s.test_case(
                    name=f"weather function followup {test_name_suffix}",
                    messages=[
                        *init_messages,
                        function_req,
                        function_resp,
                    ],
                    functions=[function],
                    expected=lambda s, n=city_names[1]: is_valid_function_call(
                        s.function_call, fun_name, check_fun_args(n)
                    ),
                )

        # Tools
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
            ),
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
