import json
from typing import Any

from aidial_client import UserInfo
from typing_extensions import TypedDict

from aidial_adapter_openai.dial_api.storage import DIAL_URL
from aidial_adapter_openai.utils.env import get_env_list
from aidial_adapter_openai.utils.log_config import logger as log

_SESSION_TAGS_FIELDS_VAR = "AWS_SESSION_TAGS_FIELDS"

# AWS STS session tag constraints:
# https://docs.aws.amazon.com/IAM/latest/UserGuide/id_session-tags.html#id_session-tags_operations
_MAX_ENTRIES = 50
_MAX_KEY_LEN = 128
_MAX_VALUE_LEN = 256


class SessionTag(TypedDict):
    Key: str
    Value: str


def _get_element_at_path(node: Any, path: str) -> Any:
    for segment in path.split("."):
        if isinstance(node, dict):
            node = node[segment]
        elif isinstance(node, list):
            node = node[int(segment)]
        else:
            raise TypeError(f"cannot index into {type(node).__name__}")
    return node


def resolve_paths(data: dict[str, Any], paths: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}

    for path in paths:
        if not path:
            continue

        try:
            element = _get_element_at_path(data, path)
        except (KeyError, IndexError, TypeError, ValueError) as e:
            log.warning(
                f"Skipping unresolved AWS STS session tags path "
                f"{path!r}: {type(e).__name__}: {e}"
            )
            continue

        result[path] = (
            element if isinstance(element, str) else json.dumps(element)
        )

    return result


def _format_paths(paths: list[str]) -> str:
    return ", ".join(paths)


def to_session_tags(flat: dict[str, str]) -> list[SessionTag]:
    """
    Truncates the keys and the values to the lengths allowed by AWS and
    drops the entries that do not survive the truncation.
    """

    safe: dict[str, str] = {}
    changed_keys: list[str] = []
    changed_values: list[str] = []
    empty_keys: list[str] = []
    collisions: list[str] = []

    items = list(flat.items())
    for index, (key, value) in enumerate(items):
        if len(safe) >= _MAX_ENTRIES:
            omitted = [path for path, _ in items[index:]]
            log.warning(
                f"AWS STS session tags entry cap reached; "
                f"omitted {len(omitted)} configured path(s): "
                f"{_format_paths(omitted)}"
            )
            break

        safe_key = key[:_MAX_KEY_LEN]
        safe_value = value[:_MAX_VALUE_LEN]

        if safe_key != key:
            changed_keys.append(key)
        if safe_value != value:
            changed_values.append(key)

        if not safe_key:
            empty_keys.append(key)
            continue
        if safe_key in safe:
            collisions.append(key)
            continue

        safe[safe_key] = safe_value

    if changed_keys:
        log.warning(
            f"Sanitized AWS STS session tags key(s): "
            f"{_format_paths(changed_keys)}"
        )
    if changed_values:
        log.warning(
            f"Sanitized AWS STS session tags value(s) for path(s): "
            f"{_format_paths(changed_values)}"
        )
    if empty_keys:
        log.warning(
            f"Dropped AWS STS session tags path(s) with empty sanitized "
            f"key(s): {_format_paths(empty_keys)}"
        )
    if collisions:
        log.warning(
            f"Dropped AWS STS session tags path(s) whose sanitized key "
            f"collides with an earlier entry: {_format_paths(collisions)}"
        )

    return [{"Key": key, "Value": value} for key, value in safe.items()]


def from_user_info(user_info: UserInfo, paths: list[str]) -> list[SessionTag]:
    tags = to_session_tags(
        resolve_paths(user_info.model_dump(mode="json"), paths)
    )
    log.debug(f"Built AWS STS session tags: {tags}")
    return tags


async def resolve_session_tags(api_key: str | None) -> list[SessionTag] | None:
    """
    Builds the AWS STS session tags out of the DIAL user info of the caller.
    Returns None when the tags are not configured or cannot be resolved.
    """

    paths = get_env_list(_SESSION_TAGS_FIELDS_VAR)
    if not paths:
        return None

    if DIAL_URL is None or api_key is None:
        log.warning(
            f"Skipping AWS STS session tags; {_SESSION_TAGS_FIELDS_VAR} is "
            "configured, but the DIAL_URL env variable or the api-key header "
            "is missing."
        )
        return None

    from aidial_adapter_openai.app import get_dial_client_pool

    client = get_dial_client_pool().create_client(
        base_url=DIAL_URL, api_key=api_key
    )

    try:
        user_info = await client.user.info()
    except Exception as e:
        log.warning(
            f"Skipping AWS STS session tags; failed to fetch DIAL user info: "
            f"{type(e).__name__}: {e}"
        )
        return None

    return from_user_info(user_info, paths) or None
