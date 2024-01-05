def get_ordinal_suffix(idx: int) -> str:
    match idx % 10:
        case 1:
            return "st"
        case 2:
            return "nd"
        case 3:
            return "rd"
        case _:
            return "th"


def format_ordinal(idx: int) -> str:
    return f"{idx}{get_ordinal_suffix(idx)}"
