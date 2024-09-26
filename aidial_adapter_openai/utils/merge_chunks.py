from typing import TypeVar

from aidial_sdk.utils.merge_chunks import merge

_Chunk = TypeVar("_Chunk", bound=dict)


def merge_chunks(*chunks: _Chunk) -> _Chunk:
    """
    The recursive merging procedure that avoids merging top-level atomic fields
    (e.g. "id", "created", "model", "object", "system_fingerprint") and
    instead chooses an _override_ merging strategy for such fields.
    Non-atomic fields (e.g. "choice", "usage") are merged following
    the standard recursive merging procedure.
    """

    assert len(chunks) > 0, "At least one chunk must be provided"

    target = chunks[0]

    def _generator():
        yield target
        for source in chunks[1:]:
            source = source.copy()
            for key, value in list(source.items()):
                if not isinstance(value, (list, dict)) and value is not None:
                    target[key] = value
                    del source[key]
            yield source

    return merge(*list(_generator()))
