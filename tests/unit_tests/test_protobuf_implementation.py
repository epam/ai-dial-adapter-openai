from google.protobuf.internal import api_implementation


def test_protobuf_uses_the_c_extension():
    assert api_implementation.Type() == "upb", (
        "protobuf fell back to its pure-Python implementation. On musl this "
        "happens when the wheel is used instead of a source build — check "
        'that poetry.toml still sets `no-binary = ["protobuf"]`.'
    )


def test_upb_extension_is_importable():
    """The backing module, imported directly, so the failure names itself."""
    from google._upb import _message  # noqa: F401
