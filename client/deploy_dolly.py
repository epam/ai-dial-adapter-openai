from google.cloud.aiplatform import Endpoint, Model, init

from utils.env import get_env

PROJECT_ID = get_env("GCP_PROJECT_ID")
REGION = get_env("DEFAULT_REGION")

init(project=PROJECT_ID, location=REGION)

SERVE_DOCKER_URI = "us-docker.pkg.dev/vertex-ai-restricted/vertex-vision-model-garden-dockers/pytorch-dolly-v2-serve"


def deploy_model(model_id):
    """Uploads and deploys the model to Vertex AI endpoint for prediction."""
    model_name = "dolly_v2_7b"
    endpoint = Endpoint.create(display_name=f"{model_name}-endpoint")
    serving_env = {
        "MODEL_ID": model_id,
    }
    model: Model = (
        Model.upload(
            display_name=model_name,
            serving_container_image_uri=SERVE_DOCKER_URI,
            serving_container_ports=[7080],
            serving_container_predict_route="/predictions/transformers_serving",
            serving_container_health_route="/ping",
            serving_container_environment_variables=serving_env,
            artifact_uri=None,
        ),
    )  # type: ignore

    model.deploy(
        endpoint=endpoint,
        machine_type="a2-highgpu-1g",
        accelerator_type="NVIDIA_TESLA_A100",
        accelerator_count=1,
        deploy_request_timeout=1800,
    )
    return model, endpoint


print("Deploying model...")
model, endpoint = deploy_model(model_id="databricks/dolly-v2-7b")
