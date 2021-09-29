__all__ = ['Lagrange1D']


import numbers

from abc import ABCMeta, abstractmethod
from typing import List, Optional, Tuple, Union

from sympy.core.expr import Expr
from sympy.core.function import Function
from sympy.core.mul import prod
from sympy.core.numbers import Number, Zero
from sympy.core.symbol import Symbol


NumberLike = Union[numbers.Number, Expr]


class BaseInterpolation1DMethod(metaclass=ABCMeta):
    '''
    - abstract method:
        - add：添加拟合数据
        - expression：拟合公式的表达式
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
        self._xs, self._ys = (tuple(), tuple()) if data is None else self._xys(data)
        self.update()

    def __call__(self, *xs: NumberLike) -> List[Expr]:
        return self.interpolate(xs)

    def __getitem__(self, x: NumberLike) -> Expr:
        return self.interpolate([x])[0]

    def __setitem__(self, x: NumberLike, y: NumberLike) -> None:
        self.add([(x, y)])
        self.update()

    def __repr__(self) -> str:
        name = self.__class__.__name__
        return f'<{name}: {self.expression()}>'

    @property
    def xs(self) -> Tuple[Expr, ...]:
        return self._xs

    @property
    def ys(self) -> Tuple[Expr, ...]:
        return self._ys

    @abstractmethod
    def add(
        self, data: List[Tuple[NumberLike, NumberLike]]
    ) -> 'BaseInterpolation1DMethod':
        pass

    @abstractmethod
    def expression(self, name: str = 'x') -> Expr:
        pass

    @abstractmethod
    def interpolate(self, xs: List[NumberLike]) -> List[Expr]:
        pass

    @abstractmethod
    def remainder(
        self, f: Function, x: Optional[Symbol] = None, name: str = 'ε'
    ) -> Expr:
        pass

    @abstractmethod
    def update(self) -> 'BaseInterpolation1DMethod':
        pass

    def _xys(
        self, data: List[Tuple[NumberLike, NumberLike]]
    ) -> Tuple[Tuple[Expr], ...]:
        f = lambda x: x if isinstance(x, Expr) else Number(x)
        return tuple(tuple(map(f, x)) for x in zip(*data))


class Lagrange1D(BaseInterpolation1DMethod):
    def add(self, data):
        xs, ys = self._xys(data)
        self._xs += xs
        self._ys += ys
        return self

    def expression(self, name='x'):
        return self._f.subs(self._, name)

    def interpolate(self, xs):
        return [self._f.subs(self._, x) for x in xs]

    def remainder(self, f, x=None, name='ε'):
        x = x or f.args[0]  # assert len(f.args)==1
        return f.diff(x, len(self._xs)).subs(x, name) * \
            prod((x-xk)/(i+1) for i, xk in enumerate(self._xs))

    def update(self):
        self._f = sum(
            (y*self._lagrange(ith) for ith, y in enumerate(self._ys)), Zero()
        )  # .simplify()
        return self

    def _lagrange(self, kth: int) -> NumberLike:
        # assert 0 <= kth < len(self._xs)
        xk = self._xs[kth]
        return prod(
            (self._-x)/(xk-x) for ith, x in enumerate(self._xs) if ith!=kth
        )


if __name__ == '__main__':
    from sympy import symbols, sin
    from sympy.abc import x


    f = sin(x)
    for xs in ((0.32, 0.34, 0.36), symbols('x:3')):
        i = Lagrange1D([(x0, f.subs(x, x0)) for x0 in xs])
        remainder = i.remainder(f).subs(x, 0.3367)
        print(f'|{i[0.3367]}-{f.subs(x, 0.3367)}|≤{remainder}')
