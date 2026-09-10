import json
import re
import unicodedata
from collections.abc import Container
from dataclasses import dataclass
from typing import Any, ClassVar, Self

from aidial_client import UserInfo
from typing_extensions import TypedDict

from aidial_adapter_openai.dial_api.storage import DIAL_URL
from aidial_adapter_openai.utils.env import get_env_dict
from aidial_adapter_openai.utils.log_config import logger as log

# The tags to pass, as a JSON object mapping the AWS tag key to the value
# source to take it from, e.g.
# {"application": "Bedrock.modelId", "project": "UserInfo.project"}.
# Setting it enables the feature.
AWS_SESSION_TAGS = get_env_dict("AWS_SESSION_TAGS")

# AWS STS session tag constraints:
# https://docs.aws.amazon.com/IAM/latest/UserGuide/id_session-tags.html#id_session-tags_operations
_MAX_ENTRIES = 50
_MAX_KEY_LEN = 128
_MAX_VALUE_LEN = 256

# AWS also rejects a key or a value that doesn't match the character pattern
# [\p{L}\p{Z}\p{N}_.:/=+\-@], which the docs above don't mention. Note that
# `,` isn't allowed, so a JSON-serialized value never passes as-is.
# `re` doesn't support \p{...}, hence the Unicode categories:
# L = letters, Z = separators (spaces), N = numbers.
_ALLOWED_TAG_CATEGORIES = frozenset("LZN")
_ALLOWED_TAG_CHARS = frozenset("_.:/=+-@")
_TAG_CHAR_PLACEHOLDER = "_"

_DEFAULT_ROLE_SESSION_NAME = "BedrockAccessSession"

# The value source of the session tag naming the user project. When such a
# tag is passed, the project names the role session instead of the default
# above.
_PROJECT_VALUE_SOURCE = "UserInfo.project"
# The marker goes in front, so that it survives the truncation of a long
# project and groups the adapter sessions together in the AWS logs.
_PROJECT_ROLE_SESSION_NAME_PREFIX = "Project_"

# AWS STS RoleSessionName constraints:
# https://docs.aws.amazon.com/STS/latest/APIReference/API_AssumeRole.html
_MAX_ROLE_SESSION_NAME_LEN = 64
_INVALID_ROLE_SESSION_NAME_CHARS = re.compile(r"[^\w+=,.@-]")


class SessionTag(TypedDict):
    Key: str
    ValueSource: str
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


def _format_tags(tags: list[str]) -> str:
    return ", ".join(tags)


def _sanitize_chars(value: str) -> str:
    """
    Replaces the characters AWS rejects, one for one so that the length of
    the value is preserved.
    """

    return "".join(
        char
        if char in _ALLOWED_TAG_CHARS
        or unicodedata.category(char)[0] in _ALLOWED_TAG_CATEGORIES
        else _TAG_CHAR_PLACEHOLDER
        for char in value
    )


def _dedupe_key(key: str, taken: Container[str]) -> str:
    """
    Postfixes a key that sanitization or truncation made collide with an
    earlier one, keeping it within the length limit.
    """

    for index in range(1, _MAX_ENTRIES + 1):
        postfix = f"_{index}"
        candidate = f"{key[: _MAX_KEY_LEN - len(postfix)]}{postfix}"
        if candidate not in taken:
            return candidate

    return key


def _sanitize_session_tags(tags: list[SessionTag]) -> list[SessionTag]:
    # Only the key and the value reach AWS, so they're the ones fitted to the
    # AWS constraints; the value source is carried through untouched.
    ret: list[SessionTag] = []
    taken: set[str] = set()
    changed_keys: list[str] = []
    changed_values: list[str] = []
    empty_keys: list[str] = []
    collisions: list[str] = []

    for index, tag in enumerate(tags):
        if len(ret) >= _MAX_ENTRIES:
            omitted = [omitted_tag["Key"] for omitted_tag in tags[index:]]
            log.warning(
                f"AWS STS session tags entry cap reached; "
                f"omitted {len(omitted)} configured tag(s): "
                f"{_format_tags(omitted)}"
            )
            break

        key, value = tag["Key"], tag["Value"]
        safe_key = _sanitize_chars(key)[:_MAX_KEY_LEN]
        safe_value = _sanitize_chars(value)[:_MAX_VALUE_LEN]

        if safe_key != key:
            changed_keys.append(key)
        if safe_value != value:
            changed_values.append(key)
        if not safe_key:
            empty_keys.append(tag["ValueSource"])
            continue
        if safe_key in taken:
            collisions.append(key)
            safe_key = _dedupe_key(safe_key, taken)

        taken.add(safe_key)
        ret.append(
            {
                "Key": safe_key,
                "ValueSource": tag["ValueSource"],
                "Value": safe_value,
            }
        )

    if changed_keys:
        log.warning(
            f"Sanitized AWS STS session tags key(s): "
            f"{_format_tags(changed_keys)}"
        )
    if changed_values:
        log.warning(
            f"Sanitized AWS STS session tags value(s): "
            f"{_format_tags(changed_values)}"
        )
    if empty_keys:
        log.warning(
            f"Dropped AWS STS session tags with an empty key, configured for "
            f"value source(s): {_format_tags(empty_keys)}"
        )
    if collisions:
        log.warning(
            f"Postfixed AWS STS session tags whose sanitized key "
            f"collides with an earlier entry: {_format_tags(collisions)}"
        )

    return ret


def get_role_session_name(session_tags: list[SessionTag] | None) -> str:
    project = next(
        (
            tag["Value"]
            for tag in session_tags or []
            if tag["ValueSource"] == _PROJECT_VALUE_SOURCE
        ),
        None,
    )

    # A user without a project is resolved to the JSON "null" by the session
    # tags, which makes for a meaningless session name.
    if project is None or project in {"", "null"}:
        return _DEFAULT_ROLE_SESSION_NAME

    project = f"{_PROJECT_ROLE_SESSION_NAME_PREFIX}{project}"
    project = _INVALID_ROLE_SESSION_NAME_CHARS.sub("_", project)
    return project[:_MAX_ROLE_SESSION_NAME_LEN]


@dataclass
class Tags:
    bedrock_model_id: list[str]
    user_info_paths: list[tuple[str, str]]

    _BEDROCK_MODEL_ID: ClassVar[str] = "Bedrock.modelId"
    _USER_INFO_PREFIX: ClassVar[str] = "UserInfo."

    @classmethod
    def parse(cls, tags: dict[str, str]) -> Self:
        bedrock_model_id: list[str] = []
        user_info_paths: list[tuple[str, str]] = []

        for tag_key, value_source in tags.items():
            if value_source == cls._BEDROCK_MODEL_ID:
                bedrock_model_id.append(tag_key)
            elif isinstance(value_source, str) and value_source.startswith(
                cls._USER_INFO_PREFIX
            ):
                user_info_paths.append(
                    (tag_key, value_source.removeprefix(cls._USER_INFO_PREFIX))
                )
            else:
                log.warning(
                    f"Skipping AWS STS session tag {tag_key!r}: unknown value "
                    f"source {value_source!r}; expected {cls._BEDROCK_MODEL_ID} "
                    f"or {cls._USER_INFO_PREFIX}<path>"
                )

        return cls(
            bedrock_model_id=bedrock_model_id,
            user_info_paths=user_info_paths,
        )

    @property
    def wants_user_info(self) -> bool:
        return bool(self.user_info_paths)

    def to_session_tags(
        self, bedrock_model_id: str | None, user_info: UserInfo | None
    ) -> list[SessionTag]:
        session_tags: list[SessionTag] = []

        if bedrock_model_id is not None:
            session_tags.extend(
                {
                    "Key": key,
                    "ValueSource": self._BEDROCK_MODEL_ID,
                    "Value": bedrock_model_id,
                }
                for key in self.bedrock_model_id
            )
        elif self.bedrock_model_id:
            log.warning(
                f"Skipping the {self._BEDROCK_MODEL_ID} AWS STS session "
                f"tag(s); the request carries no model id: "
                f"{_format_tags(self.bedrock_model_id)}"
            )

        if user_info is not None:
            resolved = resolve_paths(
                user_info.model_dump(mode="json"),
                [path for _, path in self.user_info_paths],
            )
            session_tags.extend(
                {
                    "Key": key,
                    "ValueSource": f"{self._USER_INFO_PREFIX}{path}",
                    "Value": resolved[path],
                }
                for key, path in self.user_info_paths
                if path in resolved
            )

        ret = _sanitize_session_tags(session_tags)
        log.debug(f"Built AWS STS session tags: {ret}")
        return ret


async def _fetch_user_info(api_key: str | None) -> UserInfo | None:
    if api_key is None:
        log.warning(
            "Skipping UserInfo AWS STS session tags; "
            "the request carries no DIAL API key"
        )
        return None

    if DIAL_URL is None:
        log.warning(
            "Skipping UserInfo AWS STS session tags; "
            "DIAL_URL env variable is not set"
        )
        return None

    # Fixes circular import.
    from aidial_adapter_openai.app import get_dial_client_pool

    client = get_dial_client_pool().create_client(
        base_url=DIAL_URL, api_key=api_key
    )

    try:
        return await client.user.info()
    except Exception as e:
        log.warning(
            f"Skipping UserInfo AWS STS session tags; failed to fetch DIAL "
            f"user info: {type(e).__name__}: {e}"
        )
        return None


async def resolve_session_tags(
    api_key: str | None, deployment_id: str | None
) -> list[SessionTag] | None:
    if not AWS_SESSION_TAGS:
        return None

    tags = Tags.parse(AWS_SESSION_TAGS)
    user_info = (
        await _fetch_user_info(api_key) if tags.wants_user_info else None
    )

    return tags.to_session_tags(deployment_id, user_info) or None
