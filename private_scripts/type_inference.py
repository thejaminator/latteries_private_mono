from typing import TypeVar, Callable

A = TypeVar("A")
B = TypeVar("B")


class MyList(list[A]):
    def map(self, f: Callable[[A], B]) -> "MyList[B]":
        return MyList(f(x) for x in self)


def to_int(item: str) -> int:
    return int(item)


lambda_to_int = lambda x: to_int(x)

my_list_int = MyList(["1", "2", "3"]).map(to_int)
my_list_lambda_int = MyList(["1", "2", "3"]).map(lambda x: x)
