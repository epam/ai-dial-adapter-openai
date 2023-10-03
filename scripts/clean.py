import glob
import os
import shutil
from typing import List


def remove_dir(directory_path: str):
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
        print("Removed: " + directory_path)


def remove_recursively(pattern: str, exclude_dirs: List[str] = []):
    files = glob.glob(f"**/{pattern}", recursive=True)

    def is_excluded(file: str) -> bool:
        return any(dir in file for dir in exclude_dirs)

    for file in files:
        if not is_excluded(file):
            remove_dir(file)


def main():
    remove_dir(".nox")
    remove_dir("dist")
    remove_recursively("__pycache__", exclude_dirs=[".venv"])
    remove_recursively(".pytest_cache", exclude_dirs=[".venv"])


if __name__ == "__main__":
    main()
