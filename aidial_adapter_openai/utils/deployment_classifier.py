def is_text_to_image_deployment(deployment_id: str):
    return deployment_id.lower() == "dalle3"


def is_image_to_text_deployment(deployment_id: str):
    return deployment_id.lower() == "gpt-4-vision-review"
