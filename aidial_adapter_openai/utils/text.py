def decapitalize(message: str) -> str:
    if not message:
        return message
    return message[0].lower() + message[1:]


def truncate_string(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[:n] + "..."
