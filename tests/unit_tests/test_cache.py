import pytest

from aidial_adapter_openai.utils.cache import cache


async def test_the_value_is_computed_once_per_key():
    calls: list[int] = []

    @cache()
    def factory(x: int) -> object:
        calls.append(x)
        return object()

    first = factory(1)

    assert factory(1) is first
    assert factory(2) is not first
    assert calls == [1, 2]


async def test_the_key_is_independent_of_the_argument_style():
    @cache()
    def factory(x: int) -> object:
        return [x]

    assert factory(x=1) is factory(x=1)


async def test_an_unbounded_cache_keeps_every_entry():
    @cache()
    def factory(x: int) -> object:
        return [x]

    entries = [factory(x) for x in range(100)]

    assert [factory(x) for x in range(100)] == entries


async def test_maxsize_evicts_the_least_recently_used_entry():
    @cache(maxsize=2)
    def factory(x: int) -> object:
        return [x]

    one, two = factory(1), factory(2)

    factory(1)  # makes 2 the least recently used entry
    factory(3)  # exceeds the bound, so 2 is evicted

    assert factory(1) is one
    assert factory(2) is not two


async def test_clear_closes_every_cached_value():
    closed: list[str] = []

    async def close(value: str) -> None:
        closed.append(value)

    @cache(close)
    def factory(x: int) -> str:
        return f"value-{x}"

    factory(1)
    factory(2)
    await factory.clear()

    assert sorted(closed) == ["value-1", "value-2"]


async def test_a_failing_close_doesnt_break_the_cache():
    async def close(_value: str) -> None:
        raise RuntimeError("boom")

    calls: list[int] = []

    @cache(close)
    def factory(x: int) -> str:
        calls.append(x)
        return f"value-{x}"

    factory(1)
    await factory.clear()
    factory(1)

    assert calls == [1, 1]


def test_a_bounded_cache_rejects_a_close_callback():
    """An evicted value is dropped as is, so it must own nothing."""

    async def close(_value: object) -> None: ...

    with pytest.raises(ValueError):
        cache(close, maxsize=1)
