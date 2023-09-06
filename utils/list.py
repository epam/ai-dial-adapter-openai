from typing import List, Tuple, TypeVar

T = TypeVar("T")


def list_to_tuples(lst: List[T]) -> List[Tuple[T, T]]:
    return [(lst[i], lst[i + 1]) for i in range(0, len(lst), 2)]
