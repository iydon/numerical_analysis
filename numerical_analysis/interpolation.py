__all__ = ['Lagrange1D', 'Newton1D']


import numbers

from abc import ABCMeta, abstractmethod
from typing import List, Optional, Tuple, Union

from sympy.core.expr import Expr
from sympy.core.function import Function
from sympy.core.mul import prod
from sympy.core.numbers import Number, Zero
from sympy.core.symbol import Symbol


NumberLike = Union[numbers.Number, Expr]


class Wrapper:
    def __init__(
        self,
        method: 'BaseInterpolation1DMethod', x: str = 'x', epsilon: str = 'ε',
    ) -> None:
        self._i = method
        self._x, self._epsilon, self._expression = x, epsilon, None

    def __call__(self, *xs: NumberLike) -> List[Expr]:
        return self.interpolate(xs)

    def __getitem__(self, x: NumberLike) -> Expr:
        return self.interpolate(x)[0]

    def __setitem__(self, x: NumberLike, y: NumberLike) -> None:
        self.add([(x, y)])

    def __repr__(self) -> str:
        name = self._i.__class__.__name__
        return f'<{name}: {self.expression}>'

    @property
    def expression(self) -> Expr:
        if self._expression is None:
            self._expression = self._i.expression(self._x)
        return self._expression

    @property
    def xs(self) -> Tuple[Expr, ...]:
        return self._xs

    @property
    def ys(self) -> Tuple[Expr, ...]:
        return self._ys

    def add(self, data: List[Tuple[NumberLike, NumberLike]]) -> 'Wrapper':
        self._expression = None
        self._i.add(data)
        return self

    def interpolate(self, *xs: List[NumberLike]) -> List[Expr]:
        return self._i.interpolate(xs)  # list(xs)

    def remainder(self, f: Function, x: Optional[Symbol] = None):
        x = x or f.args[0]  # assert len(f.args)==1
        return self._i.remainder(f, x or f.args[0], self._epsilon)


class BaseInterpolation1DMethod(metaclass=ABCMeta):
    '''
    - abstract method:
        - expression：拟合公式的表达式
        - init：初始化
        - interpolate：对指定数据进行插值
        - remainder：余项
        - update：更新拟合公式
    '''
    def __init__(
        self,
        data: Optional[List[Tuple[NumberLike, NumberLike]]] = None,
    ) -> None:
        '''
        - Argument:
            - data: [(x1, y1), (x2, y2), ..., (xn, yn)]
        '''
        self._ = Symbol('_')
        self._xs, self._ys = tuple(), tuple()
        self.init()
        if data is not None:
            self.add(data)

    def add(self, data: List[Tuple[NumberLike, NumberLike]]) -> None:
        f = lambda x: x if isinstance(x, Expr) else Number(x)
        xs, ys = (tuple(map(f, x)) for x in zip(*data))
        self._xs += xs
        self._ys += ys
        self.update(xs, ys)

    def wrap(self, *args, **kwargs) -> Wrapper:
        return Wrapper(self, *args, **kwargs)

    @abstractmethod
    def expression(self, name: str = 'x') -> Expr:
        pass

    @abstractmethod
    def init(self) -> None:
        pass

    @abstractmethod
    def interpolate(self, xs: List[NumberLike]) -> List[Expr]:
        pass

    @abstractmethod
    def remainder(self, f: Function, x: Symbol, name: str = 'ε') -> Expr:
        pass

    @abstractmethod
    def update(self, xs: List[NumberLike], ys: List[NumberLike]) -> None:
        pass


class Lagrange1D(BaseInterpolation1DMethod):
    def expression(self, name='x'):
        return self._f.subs(self._, name)

    def init(self):
        self._f = Zero()

    def interpolate(self, xs):
        return [self._f.subs(self._, x) for x in xs]

    def remainder(self, f, x, name='ε'):
        return f.diff(x, len(self._xs)).subs(x, name) * \
            prod((x-xk)/(i+1) for i, xk in enumerate(self._xs))

    def update(self, *_):  # 每次重新计算拟合公式
        self._f = sum(
            (y*self._lagrange(ith) for ith, y in enumerate(self._ys)), Zero()
        )  # .simplify()

    def _lagrange(self, kth: int) -> NumberLike:
        # assert 0 <= kth < len(self._xs)
        return prod(
            (self._-x) / (self._xs[kth]-x)
            for ith, x in enumerate(self._xs) if ith!=kth
        )


class Newton1D(BaseInterpolation1DMethod):
    def expression(self, name='x'):
        raise NotImplementedError

    def init(self):
        raise NotImplementedError

    def interpolate(self, xs):
        raise NotImplementedError

    def remainder(self, f, x, name='ε'):
        raise NotImplementedError

    def update(self, xs, ys):
        raise NotImplementedError


if __name__ == '__main__':
    from sympy import symbols, sin
    from sympy.abc import x

    f = sin(x)
    for xs in ((0.32, 0.34, 0.36), symbols('x:3')):
        i = Lagrange1D([(x0, f.subs(x, x0)) for x0 in xs]).wrap()
        remainder = i.remainder(f).subs(x, 0.3367)
        print(f'|{i[0.3367]}-{f.subs(x, 0.3367)}|≤{remainder}')
