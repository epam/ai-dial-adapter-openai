def is_text_to_image_deployment(deployment_id: str):
    return deployment_id.lower() == "dalle3"
