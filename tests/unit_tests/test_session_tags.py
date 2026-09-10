import logging
from typing import Any

import pytest
import respx
from aidial_client import UserInfo

from aidial_adapter_openai.utils import session_tags
from aidial_adapter_openai.utils.session_tags import (
    SessionTag,
    Tags,
    _sanitize_session_tags,
    get_role_session_name,
    resolve_paths,
    resolve_session_tags,
)

_DIAL_URL = "http://test-dial-url"
_API_KEY = "test-api-key"
_MODEL = "openai.gpt-5.4"
_LOGGER = "aidial_adapter_openai"

_USER_INFO = {
    "roles": ["admin", "user"],
    "project": "test-project",
    "userClaims": {"email": "user@example.com", "groups": ["a", "b"]},
}


def _tag(key: str, value_source: str, value: str) -> SessionTag:
    return {"Key": key, "ValueSource": value_source, "Value": value}


def _model_tag(key: str, value: str) -> SessionTag:
    return _tag(key, "Bedrock.modelId", value)


def _user_info_tag(key: str, path: str, value: str) -> SessionTag:
    return _tag(key, f"UserInfo.{path}", value)


@pytest.fixture(autouse=True)
def dial_url(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(session_tags, "DIAL_URL", _DIAL_URL)


@pytest.fixture(autouse=True)
def no_session_tags(monkeypatch: pytest.MonkeyPatch):
    """
    AWS_SESSION_TAGS is read into the module at import time, so the
    environment of whoever runs the tests must not leak into the tests.
    """
    monkeypatch.setattr(session_tags, "AWS_SESSION_TAGS", None)


def _configure(monkeypatch: pytest.MonkeyPatch, tags: dict[str, str]) -> None:
    monkeypatch.setattr(session_tags, "AWS_SESSION_TAGS", tags)


# ---------------------------------------------------------------- resolve_paths


@pytest.mark.parametrize(
    "paths, expected",
    [
        pytest.param([], {}, id="no_paths"),
        pytest.param([""], {}, id="empty_path_is_ignored"),
        pytest.param(
            ["project"], {"project": "test-project"}, id="string_as_is"
        ),
        pytest.param(["roles.0"], {"roles.0": "admin"}, id="list_index"),
        pytest.param(
            ["userClaims.email"],
            {"userClaims.email": "user@example.com"},
            id="nested_object",
        ),
        pytest.param(
            ["roles"], {"roles": '["admin", "user"]'}, id="list_is_serialized"
        ),
        pytest.param(
            ["userClaims"],
            {
                "userClaims": '{"email": "user@example.com", "groups": ["a", "b"]}'
            },
            id="object_is_serialized",
        ),
        pytest.param(["missing"], {}, id="missing_key_is_skipped"),
        pytest.param(["roles.5"], {}, id="index_out_of_range_is_skipped"),
        pytest.param(["roles.x"], {}, id="non_integer_index_is_skipped"),
        pytest.param(
            ["project.name"], {}, id="indexing_into_a_scalar_is_skipped"
        ),
        pytest.param(
            ["project", "missing", "roles.1"],
            {"project": "test-project", "roles.1": "user"},
            id="unresolved_path_doesnt_affect_the_others",
        ),
    ],
)
def test_resolve_paths(paths: list[str], expected: dict[str, str]):
    assert resolve_paths(_USER_INFO, paths) == expected


def test_resolve_paths_serializes_null():
    assert resolve_paths({"project": None}, ["project"]) == {"project": "null"}


# ------------------------------------------------------------------ Tags.parse


@pytest.mark.parametrize(
    "config, expected",
    [
        pytest.param({}, Tags([], []), id="empty"),
        pytest.param(
            {"application": "Bedrock.modelId"},
            Tags(["application"], []),
            id="model_id",
        ),
        pytest.param(
            {"employee": "UserInfo.userClaims.email"},
            Tags([], [("employee", "userClaims.email")]),
            id="user_info_path",
        ),
        pytest.param(
            {"application": "Bedrock.modelId", "project": "UserInfo.project"},
            Tags(["application"], [("project", "project")]),
            id="both_sources",
        ),
        pytest.param(
            {"role": "UserInfo.roles.0", "project": "UserInfo.project"},
            Tags([], [("role", "roles.0"), ("project", "project")]),
            id="user_info_order_is_preserved",
        ),
        pytest.param(
            {"a": "Bedrock.modelId", "b": "Bedrock.modelId"},
            Tags(["a", "b"], []),
            id="one_source_under_several_keys",
        ),
        pytest.param(
            {"project": "project"}, Tags([], []), id="unprefixed_path"
        ),
        pytest.param(
            {"project": "Nope.project"}, Tags([], []), id="unknown_prefix"
        ),
        pytest.param(
            {"region": "Bedrock.region"},
            Tags([], []),
            id="unknown_bedrock_field",
        ),
        pytest.param(
            {"project": "UserInfoProject"}, Tags([], []), id="prefix_near_miss"
        ),
        pytest.param(
            {"project": 42}, Tags([], []), id="non_string_value_source"
        ),
        pytest.param(
            {"kept": "UserInfo.project", "dropped": "Nope.project"},
            Tags([], [("kept", "project")]),
            id="unknown_source_doesnt_affect_the_others",
        ),
    ],
)
def test_tags_parse(config: dict[str, Any], expected: Tags):
    assert Tags.parse(config) == expected


@pytest.mark.parametrize(
    "value_source",
    ["project", "Nope.project", "Bedrock.region", "UserInfoProject"],
)
def test_tags_parse_logs_unknown_value_sources(
    caplog: pytest.LogCaptureFixture, value_source: str
):
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        Tags.parse({"tag": value_source})

    assert "unknown value source" in caplog.text
    assert repr(value_source) in caplog.text


@pytest.mark.parametrize(
    "config, expected",
    [
        pytest.param({}, False, id="empty"),
        pytest.param({"a": "Bedrock.modelId"}, False, id="model_id_only"),
        pytest.param({"a": "UserInfo.project"}, True, id="user_info_path"),
        pytest.param(
            {"a": "Bedrock.modelId", "b": "UserInfo.project"},
            True,
            id="both_sources",
        ),
    ],
)
def test_tags_wants_user_info(config: dict[str, str], expected: bool):
    assert Tags.parse(config).wants_user_info is expected


# ------------------------------------------------------ _sanitize_session_tags


def test_sanitize_session_tags_fits_the_key_and_value():
    tags = [_model_tag("k" * 200, "v" * 300)]

    assert _sanitize_session_tags(tags) == [_model_tag("k" * 128, "v" * 256)]


def test_sanitize_session_tags_keeps_one_source_under_every_key():
    tags = [_model_tag("a", _MODEL), _model_tag("b", _MODEL)]

    assert _sanitize_session_tags(tags) == tags


@pytest.mark.parametrize(
    "value, expected",
    [
        pytest.param(
            '["test_user@example.com"]',
            "__test_user@example.com__",
            id="json_serialized_list",
        ),
        pytest.param(
            '["read", "write"]',
            "__read__ _write__",
            id="comma_is_rejected_but_the_space_survives",
        ),
        pytest.param(
            "a_b.c:d/e=f+g-h@i",
            "a_b.c:d/e=f+g-h@i",
            id="allowed_punctuation_passes_through",
        ),
        pytest.param(
            "EPAM / DIAL (prod)",
            "EPAM / DIAL _prod_",
            id="spaces_kept_parens_replaced",
        ),
        pytest.param(
            "Проект-42", "Проект-42", id="non_latin_scripts_pass_through"
        ),
        pytest.param("naïve", "naïve", id="diacritics_pass_through"),
    ],
)
def test_sanitize_session_tags_replaces_disallowed_chars_in_values(
    value: str, expected: str
):
    tags = _sanitize_session_tags([_model_tag("key", value)])

    assert tags == [_model_tag("key", expected)]


def test_sanitize_session_tags_replaces_disallowed_chars_in_keys():
    tags = _sanitize_session_tags([_model_tag('a"b,c', _MODEL)])

    assert tags == [_model_tag("a_b_c", _MODEL)]


def test_sanitize_session_tags_preserves_the_value_length():
    value = '{"a": ["b", "c"]}'

    tags = _sanitize_session_tags([_model_tag("key", value)])

    assert len(tags[0]["Value"]) == len(value)


def test_sanitize_session_tags_caps_at_50_entries():
    tags = _sanitize_session_tags(
        [_model_tag(f"key-{i}", str(i)) for i in range(60)]
    )

    assert len(tags) == 50
    assert tags[0] == _model_tag("key-0", "0")
    assert tags[-1] == _model_tag("key-49", "49")


def test_sanitize_session_tags_logs_the_capped_entries_by_key(
    caplog: pytest.LogCaptureFixture,
):
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        _sanitize_session_tags(
            [_model_tag(f"key-{i}", str(i)) for i in range(52)]
        )

    assert "entry cap reached; omitted 2 configured tag(s)" in caplog.text
    assert "key-50, key-51" in caplog.text


def test_sanitize_session_tags_postfixes_truncated_key_collisions():
    tags = _sanitize_session_tags(
        [
            _model_tag("k" * 128 + "1", "first"),
            _model_tag("k" * 128 + "2", "second"),
            _model_tag("k" * 128 + "3", "third"),
        ]
    )

    assert [tag["Value"] for tag in tags] == ["first", "second", "third"]
    assert [tag["Key"] for tag in tags] == [
        "k" * 128,
        "k" * 126 + "_1",
        "k" * 126 + "_2",
    ]
    assert all(len(tag["Key"]) == 128 for tag in tags)


def test_sanitize_session_tags_postfixes_sanitized_key_collisions():
    tags = _sanitize_session_tags(
        [_model_tag("a,b", "first"), _model_tag("a;b", "second")]
    )

    assert tags == [
        _model_tag("a_b", "first"),
        _model_tag("a_b_1", "second"),
    ]


def test_sanitize_session_tags_logs_postfixed_key_collisions(
    caplog: pytest.LogCaptureFixture,
):
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        _sanitize_session_tags(
            [_model_tag("a,b", "first"), _model_tag("a;b", "second")]
        )

    assert "Postfixed AWS STS session tags" in caplog.text
    assert "a;b" in caplog.text


def test_sanitize_session_tags_drops_empty_keys_but_keeps_empty_values():
    tags = _sanitize_session_tags(
        [_model_tag("", "dropped"), _user_info_tag("kept", "project", "")]
    )

    assert tags == [_user_info_tag("kept", "project", "")]


def test_sanitize_session_tags_logs_the_source_of_an_empty_key(
    caplog: pytest.LogCaptureFixture,
):
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        _sanitize_session_tags([_user_info_tag("", "project", "value")])

    assert "empty key" in caplog.text
    assert "UserInfo.project" in caplog.text


def test_sanitize_session_tags_logs_truncated_keys_and_values(
    caplog: pytest.LogCaptureFixture,
):
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        _sanitize_session_tags([_model_tag("k" * 200, "v" * 300)])

    assert "Sanitized AWS STS session tags key(s)" in caplog.text
    assert "Sanitized AWS STS session tags value(s)" in caplog.text


# --------------------------------------------------------- Tags.to_session_tags


@pytest.fixture
def user_info() -> UserInfo:
    return UserInfo(**_USER_INFO)


def test_to_session_tags_keys_every_tag_by_its_configured_key(
    user_info: UserInfo,
):
    tags = Tags.parse(
        {
            "application": "Bedrock.modelId",
            "role": "UserInfo.roles.0",
            "project": "UserInfo.project",
            "employee": "UserInfo.userClaims.email",
            "groups": "UserInfo.userClaims.groups",
        }
    )

    assert tags.to_session_tags(_MODEL, user_info) == [
        _model_tag("application", _MODEL),
        _user_info_tag("role", "roles.0", "admin"),
        _user_info_tag("project", "project", "test-project"),
        _user_info_tag("employee", "userClaims.email", "user@example.com"),
        # The JSON punctuation of a serialized value is sanitized away.
        _user_info_tag("groups", "userClaims.groups", "__a__ _b__"),
    ]


def test_to_session_tags_serializes_a_missing_project_as_null():
    tags = Tags.parse({"project": "UserInfo.project"})

    assert tags.to_session_tags(None, UserInfo(roles=[])) == [
        _user_info_tag("project", "project", "null")
    ]


@pytest.mark.parametrize(
    "value_source, expected_value",
    [
        pytest.param("Bedrock.modelId", _MODEL, id="model_id"),
        pytest.param("UserInfo.project", "test-project", id="user_info_path"),
    ],
)
def test_to_session_tags_repeats_a_source_under_every_key(
    user_info: UserInfo, value_source: str, expected_value: str
):
    tags = Tags.parse({"a": value_source, "b": value_source})

    assert tags.to_session_tags(_MODEL, user_info) == [
        _tag("a", value_source, expected_value),
        _tag("b", value_source, expected_value),
    ]


def test_to_session_tags_puts_the_model_id_first(user_info: UserInfo):
    tags = Tags.parse(
        {"project": "UserInfo.project", "application": "Bedrock.modelId"}
    )

    assert tags.to_session_tags(_MODEL, user_info) == [
        _model_tag("application", _MODEL),
        _user_info_tag("project", "project", "test-project"),
    ]


def test_to_session_tags_passes_only_the_configured_tags(user_info: UserInfo):
    tags = Tags.parse({"project": "UserInfo.project"})

    assert tags.to_session_tags(_MODEL, user_info) == [
        _user_info_tag("project", "project", "test-project")
    ]


def test_to_session_tags_without_user_info():
    tags = Tags.parse(
        {"application": "Bedrock.modelId", "project": "UserInfo.project"}
    )

    assert tags.to_session_tags(_MODEL, None) == [
        _model_tag("application", _MODEL)
    ]


def test_to_session_tags_without_a_model_id(user_info: UserInfo):
    tags = Tags.parse(
        {"application": "Bedrock.modelId", "project": "UserInfo.project"}
    )

    assert tags.to_session_tags(None, user_info) == [
        _user_info_tag("project", "project", "test-project")
    ]


def test_to_session_tags_without_any_source():
    tags = Tags.parse({"application": "Bedrock.modelId"})

    assert tags.to_session_tags(None, None) == []


def test_to_session_tags_keeps_the_model_id_when_capped():
    user_info = UserInfo(roles=[f"role-{i}" for i in range(60)])
    config = {f"role-{i}": f"UserInfo.roles.{i}" for i in range(60)}
    config["application"] = "Bedrock.modelId"

    tags = Tags.parse(config).to_session_tags(_MODEL, user_info)

    assert len(tags) == 50
    assert tags[0] == _model_tag("application", _MODEL)
    assert tags[-1] == _user_info_tag("role-48", "roles.48", "role-48")


def test_to_session_tags_truncates_long_model_ids():
    tags = Tags.parse({"application": "Bedrock.modelId"}).to_session_tags(
        "m" * 300, None
    )

    assert tags == [_model_tag("application", "m" * 256)]


# ---------------------------------------------------- get_role_session_name


@pytest.mark.parametrize(
    "tags, expected",
    [
        pytest.param(None, "BedrockAccessSession", id="no_tags"),
        pytest.param([], "BedrockAccessSession", id="empty_tags"),
        pytest.param(
            [_model_tag("application", _MODEL)],
            "BedrockAccessSession",
            id="no_project_tag",
        ),
        pytest.param(
            [_tag("project", "project", "epam")],
            "BedrockAccessSession",
            id="unprefixed_value_source_is_not_the_project",
        ),
        pytest.param(
            [_user_info_tag("project", "project", "null")],
            "BedrockAccessSession",
            id="null_project",
        ),
        pytest.param(
            [_user_info_tag("project", "project", "")],
            "BedrockAccessSession",
            id="empty_project",
        ),
        pytest.param(
            [_user_info_tag("anything", "project", "epam")],
            "Project_epam",
            id="the_tag_key_doesnt_matter",
        ),
        pytest.param(
            [
                _model_tag("application", _MODEL),
                _user_info_tag("project", "project", "epam"),
            ],
            "Project_epam",
            id="found_among_other_tags",
        ),
        pytest.param(
            [_user_info_tag("project", "project", "EPAM / DIAL _prod_")],
            "Project_EPAM___DIAL__prod_",
            id="disallowed_chars_are_replaced",
        ),
        pytest.param(
            [_user_info_tag("project", "project", "a+b=c,d.e@f-g_1")],
            "Project_a+b=c,d.e@f-g_1",
            id="allowed_punctuation_passes_through",
        ),
    ],
)
def test_get_role_session_name(tags: list[SessionTag] | None, expected: str):
    assert get_role_session_name(tags) == expected


def test_get_role_session_name_truncates_long_projects():
    tags = [_user_info_tag("project", "project", "p" * 200)]

    name = get_role_session_name(tags)

    assert name == "Project_" + "p" * 56
    assert len(name) == 64


# ------------------------------------------------------- resolve_session_tags


@pytest.fixture
def mock_user_info():
    with respx.mock(
        base_url=_DIAL_URL + "/v1",
        assert_all_called=False,
        assert_all_mocked=True,
    ) as router:
        yield router.get("/user/info")


async def test_resolve_session_tags_is_disabled_by_default():
    assert await resolve_session_tags(_API_KEY, _MODEL) is None


async def test_resolve_session_tags_returns_tags(
    monkeypatch: pytest.MonkeyPatch, mock_user_info: Any
):
    _configure(
        monkeypatch,
        {
            "application": "Bedrock.modelId",
            "role": "UserInfo.roles.0",
            "project": "UserInfo.project",
        },
    )
    mock_user_info.respond(json=_USER_INFO)

    assert await resolve_session_tags(_API_KEY, _MODEL) == [
        _model_tag("application", _MODEL),
        _user_info_tag("role", "roles.0", "admin"),
        _user_info_tag("project", "project", "test-project"),
    ]


async def test_resolve_session_tags_skips_the_user_info_request_when_not_wanted(
    monkeypatch: pytest.MonkeyPatch, mock_user_info: Any
):
    _configure(monkeypatch, {"application": "Bedrock.modelId"})

    assert await resolve_session_tags(_API_KEY, _MODEL) == [
        _model_tag("application", _MODEL)
    ]
    assert not mock_user_info.called


async def test_resolve_session_tags_returns_none_when_nothing_resolves(
    monkeypatch: pytest.MonkeyPatch, mock_user_info: Any
):
    _configure(monkeypatch, {"claim": "UserInfo.userClaims.missing"})
    mock_user_info.respond(json=_USER_INFO)

    assert await resolve_session_tags(_API_KEY, None) is None


async def test_resolve_session_tags_tolerates_a_failing_user_info_request(
    monkeypatch: pytest.MonkeyPatch,
    mock_user_info: Any,
    caplog: pytest.LogCaptureFixture,
):
    _configure(
        monkeypatch,
        {"application": "Bedrock.modelId", "project": "UserInfo.project"},
    )
    mock_user_info.respond(status_code=403)

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        tags = await resolve_session_tags(_API_KEY, _MODEL)

    # The model id is an independent source, so it survives.
    assert tags == [_model_tag("application", _MODEL)]
    assert "failed to fetch DIAL user info" in caplog.text


async def test_resolve_session_tags_tolerates_a_missing_api_key(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    _configure(
        monkeypatch,
        {"application": "Bedrock.modelId", "project": "UserInfo.project"},
    )

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        tags = await resolve_session_tags(None, _MODEL)

    assert tags == [_model_tag("application", _MODEL)]
    assert "carries no DIAL API key" in caplog.text


async def test_resolve_session_tags_tolerates_a_missing_dial_url(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    _configure(
        monkeypatch,
        {"application": "Bedrock.modelId", "project": "UserInfo.project"},
    )
    monkeypatch.setattr(session_tags, "DIAL_URL", None)

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        tags = await resolve_session_tags(_API_KEY, _MODEL)

    assert tags == [_model_tag("application", _MODEL)]
    assert "DIAL_URL env variable is not set" in caplog.text


async def test_resolve_session_tags_requires_a_user_info_tag_for_the_api_key(
    monkeypatch: pytest.MonkeyPatch,
):
    """
    A UserInfo-only configuration yields nothing without the DIAL user info.
    """
    _configure(monkeypatch, {"project": "UserInfo.project"})

    assert await resolve_session_tags(None, _MODEL) is None
