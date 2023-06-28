from enum import Enum
from typing import List, Type, TypeVar

import inquirer

V = TypeVar("V")


def select_option(title: str, options: List[V]) -> V:
    questions = [
        inquirer.List(
            "option",
            message=title,
            choices=[(str(option), option) for option in options],
            carousel=True,
        ),
    ]
    return inquirer.prompt(questions)["option"]  # type: ignore


T = TypeVar("T", bound=Enum)


def select_enum(title: str, enum: Type[T]) -> T:
    questions = [
        inquirer.List(
            "option",
            message=title,
            choices=[(option.value, option) for option in enum],
            carousel=True,
        ),
    ]
    return inquirer.prompt(questions)["option"]  # type: ignore
