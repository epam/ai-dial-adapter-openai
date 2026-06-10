def normalize_audio_type(audio_type: str) -> str:
    match audio_type.lower().strip():
        case "audio/x-m4a" | "audio/m4a":
            return "audio/mp4"
        case _:
            return audio_type
