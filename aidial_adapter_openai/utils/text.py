def decapitalize(message: str) -> str:
    if not message:
        return message
    return message[0].lower() + message[1:]
