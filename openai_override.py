from openai import util, openai_response, api_requestor, error
import openai

class OpenAIException(Exception):
    def __init__(self, body, code, resp, headers):
        self.body = body
        self.code = code
        self.resp = resp
        self.headers = headers

        super().__init__(resp)


class OpenAIAPIRequestor(api_requestor.APIRequestor):
    def handle_error_response(self, rbody, rcode, resp, rheaders, stream_error=False):
        raise OpenAIException(
            rbody,
            rcode,
            resp,
            rheaders
        )

class OpenAIChatCompletion(openai.ChatCompletion):
    async def acreate(
        self,
        api_key=None,
        api_base=None,
        api_type=None,
        request_id=None,
        api_version=None,
        organization=None,
        **params,
    ):
        (
            deployment_id,
            engine,
            timeout,
            stream,
            headers,
            request_timeout,
            typed_api_type,
            requestor,
            url,
            params,
        ) = self.__prepare_create_request(
            api_key, api_base, api_type, api_version, organization, **params
        )
        response, _, api_key = await requestor.arequest(
            "post",
            url,
            params=params,
            headers=headers,
            stream=stream,
            request_id=request_id,
            request_timeout=request_timeout,
        )

        if stream:
            # must be an iterator
            assert not isinstance(response, openai_response.OpenAIResponse)
            return (
                util.convert_to_openai_object(
                    line,
                    api_key,
                    api_version,
                    organization,
                    engine=engine,
                    plain_old_data=self.plain_old_data,
                )
                async for line in response
            )
        else:
            obj = util.convert_to_openai_object(
                response,
                api_key,
                api_version,
                organization,
                engine=engine,
                plain_old_data=self.plain_old_data,
            )

            if timeout is not None:
                await obj.await_(timeout=timeout or None)

        return obj

    @classmethod
    def __prepare_create_request(
        self,
        api_key=None,
        api_base=None,
        api_type=None,
        api_version=None,
        organization=None,
        **params,
    ):
        deployment_id = params.pop("deployment_id", None)
        engine = params.pop("engine", deployment_id)
        model = params.get("model", None)
        timeout = params.pop("timeout", None)
        stream = params.get("stream", False)
        headers = params.pop("headers", None)
        request_timeout = params.pop("request_timeout", None)
        typed_api_type = self._get_api_type_and_version(api_type=api_type)[0]
        if typed_api_type in (util.ApiType.AZURE, util.ApiType.AZURE_AD):
            if deployment_id is None and engine is None:
                raise error.InvalidRequestError(
                    "Must provide an 'engine' or 'deployment_id' parameter to create a %s"
                    % self,
                    "engine",
                )
        else:
            if model is None and engine is None:
                raise error.InvalidRequestError(
                    "Must provide an 'engine' or 'model' parameter to create a %s"
                    % self,
                    "engine",
                )

        if timeout is None:
            # No special timeout handling
            pass
        elif timeout > 0:
            # API only supports timeouts up to MAX_TIMEOUT
            params["timeout"] = min(timeout, self.MAX_TIMEOUT)
            timeout = (timeout - params["timeout"]) or None
        elif timeout == 0:
            params["timeout"] = self.MAX_TIMEOUT

        requestor = OpenAIAPIRequestor(
            api_key,
            api_base=api_base,
            api_type=api_type,
            api_version=api_version,
            organization=organization,
        )
        url = self.class_url(engine, api_type, api_version)
        return (
            deployment_id,
            engine,
            timeout,
            stream,
            headers,
            request_timeout,
            typed_api_type,
            requestor,
            url,
            params,
        )
