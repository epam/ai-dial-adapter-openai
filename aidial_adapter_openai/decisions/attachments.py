from aidial_adapter_openai.responses.request import AttachmentRule

# Decisions rejects files, so only images are inlined
DECISIONS_ATTACHMENT_RULES = [
    AttachmentRule(
        "input[?type == null || type == 'message'].content[] | [?type == 'input_image']",
        "image_url",
    ),
    AttachmentRule(
        "input[?type == 'custom_tool_call_output' || type == 'function_call_output'].output[] | [?type == 'input_image']",
        "image_url",
    ),
    AttachmentRule(
        "questions[].instructions[] | [?type == 'input_image']",
        "image_url",
    ),
]
