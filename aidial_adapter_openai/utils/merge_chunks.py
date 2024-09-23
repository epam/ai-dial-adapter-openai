from aidial_sdk.utils.merge_chunks import merge


def merge_chunks(target: dict, source: dict) -> dict:
    """
    The recursive merging procedure that avoids merging top-level atomic fields
    (e.g. "id", "created", "model", "object", "system_fingerprint") and
    instead chooses an _override_ merging strategy for such fields.

    Non-atomic field (e.g. "choice", "usage") are merged following
    the standard merging procedure.
    """
    source = source.copy()

    for key, value in list(source.items()):
        if not isinstance(value, (list, dict)) and value is not None:
            target[key] = value
            del source[key]

    return merge(target, source)
