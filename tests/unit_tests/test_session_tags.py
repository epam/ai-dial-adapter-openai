from typing import Any

import pytest
import respx
from aidial_client import UserInfo

from aidial_adapter_openai.utils import session_tags
from aidial_adapter_openai.utils.session_tags import (
    SessionTag,
    from_user_info,
    resolve_paths,
    resolve_session_tags,
    to_session_tags,
)

_DIAL_URL = "http://test-dial-url"
_API_KEY = "test-api-key"

_USER_INFO = {
    "roles": ["admin", "user"],
    "project": "test-project",
    "userClaims": {"email": "user@example.com", "groups": ["a", "b"]},
}


@pytest.fixture(autouse=True)
def dial_url(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(session_tags, "DIAL_URL", _DIAL_URL)


@pytest.fixture(autouse=True)
def no_session_tags_fields(monkeypatch: pytest.MonkeyPatch):
    """
    The environment of whoever runs the tests must not leak into the tests.
    """
    monkeypatch.delenv("AWS_SESSION_TAGS_FIELDS", raising=False)


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


def test_to_session_tags_preserves_the_order():
    flat = {"b": "2", "a": "1"}

    assert to_session_tags(flat) == [
        {"Key": "b", "Value": "2"},
        {"Key": "a", "Value": "1"},
    ]


def test_to_session_tags_truncates_keys_and_values():
    flat = {"k" * 200: "v" * 300}

    assert to_session_tags(flat) == [{"Key": "k" * 128, "Value": "v" * 256}]


def test_to_session_tags_drops_a_key_colliding_after_truncation():
    flat = {"k" * 128 + "1": "first", "k" * 128 + "2": "second"}

    assert to_session_tags(flat) == [{"Key": "k" * 128, "Value": "first"}]


def test_to_session_tags_drops_an_empty_key():
    flat = {"": "value", "kept": "value"}

    assert to_session_tags(flat) == [{"Key": "kept", "Value": "value"}]


def test_to_session_tags_caps_the_number_of_entries():
    flat = {f"key-{i}": str(i) for i in range(60)}

    tags = to_session_tags(flat)

    assert len(tags) == 50
    assert tags[0] == {"Key": "key-0", "Value": "0"}
    assert tags[-1] == {"Key": "key-49", "Value": "49"}


def test_from_user_info():
    user_info = UserInfo(**_USER_INFO)

    assert from_user_info(user_info, ["roles.0", "userClaims.email"]) == [
        {"Key": "roles.0", "Value": "admin"},
        {"Key": "userClaims.email", "Value": "user@example.com"},
    ]


@pytest.fixture
def mock_user_info():
    with respx.mock(
        base_url=_DIAL_URL + "/v1",
        assert_all_called=False,
        assert_all_mocked=True,
    ) as router:
        yield router.get("/user/info")


async def test_resolve_session_tags_is_disabled_by_default():
    assert await resolve_session_tags(_API_KEY) is None


async def test_resolve_session_tags(
    monkeypatch: pytest.MonkeyPatch, mock_user_info: Any
):
    monkeypatch.setenv("AWS_SESSION_TAGS_FIELDS", "roles.0, project")
    mock_user_info.respond(json=_USER_INFO)

    tags: list[SessionTag] | None = await resolve_session_tags(_API_KEY)

    assert tags == [
        {"Key": "roles.0", "Value": "admin"},
        {"Key": "project", "Value": "test-project"},
    ]


async def test_resolve_session_tags_returns_none_when_nothing_resolves(
    monkeypatch: pytest.MonkeyPatch, mock_user_info: Any
):
    monkeypatch.setenv("AWS_SESSION_TAGS_FIELDS", "userClaims.missing")
    mock_user_info.respond(json=_USER_INFO)

    assert await resolve_session_tags(_API_KEY) is None


async def test_resolve_session_tags_tolerates_a_failing_user_info_request(
    monkeypatch: pytest.MonkeyPatch, mock_user_info: Any
):
    monkeypatch.setenv("AWS_SESSION_TAGS_FIELDS", "project")
    mock_user_info.respond(status_code=403)

    assert await resolve_session_tags(_API_KEY) is None


async def test_resolve_session_tags_requires_an_api_key(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("AWS_SESSION_TAGS_FIELDS", "project")

    assert await resolve_session_tags(None) is None


async def test_resolve_session_tags_requires_the_dial_url(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("AWS_SESSION_TAGS_FIELDS", "project")
    monkeypatch.setattr(session_tags, "DIAL_URL", None)

    assert await resolve_session_tags(_API_KEY) is None
