def format_ordinal(idx: int) -> str:
    return f"{idx}{get_ordinal_suffix(idx)}"


def get_ordinal_suffix(idx: int) -> str:
    idx = abs(idx)
    if 11 <= (idx % 100) <= 13:
        return "th"
    else:
        return ["th", "st", "nd", "rd", "th"][min(idx % 10, 4)]
