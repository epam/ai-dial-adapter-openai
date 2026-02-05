from openai._types import FileTypes

from aidial_adapter_openai.video_generation.prompt import VideoGenPrompt


def get_last_file(self: VideoGenPrompt) -> FileTypes | None:
    for resource in reversed(self.resources):
        mime_type = resource.type
        return ("file", resource.data, mime_type)

    return None
