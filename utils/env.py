import os


def get_env(name: str) -> str:
    if name in os.environ:
        val = os.environ.get(name)
        if val is not None:
            return val

    raise Exception(f"{name} env variable is not set")


def get_project_id() -> str:
    return os.environ.get("PROJECT_ID", "vertex-ai-391108")
