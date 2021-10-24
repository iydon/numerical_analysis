__all__ = [
    'Lagrange', 'NewtonBackward', 'NewtonForward', 'Hermite',
    'PiecewiseLinearInterpolation', 'PiecewiseHermite', 'Spline',
]


import numbers

from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from sympy.core.expr import Expr
from sympy.core.function import Function
from sympy.core.mul import prod
from sympy.core.numbers import Number, One, Zero
from sympy.core.symbol import Symbol
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.special.bsplines import interpolating_spline


NumberLike = Union[numbers.Number, Expr]

one, zero = One(), Zero()


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

    def __delitem__(self, x: NumberLike) -> None:
        index = self._i._xs.index(x)
        self._i._xs = self._i._xs[:index] + self._i._xs[index+1:]
        self._i._ys = self._i._ys[:index] + self._i._ys[index+1:]

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
        return self._i._xs

    @property
    def ys(self) -> Tuple[Expr, ...]:
        return self._i._ys

    def add(self, data: List[Tuple[NumberLike, NumberLike]]) -> 'Wrapper':
        self._expression = None
        self._i.add(data)
        return self

    def attr(self, name: str, *args, **kwargs) -> Any:
        attr = getattr(self._i, name)
        if isinstance(attr, Callable):
            attr = attr(*args, **kwargs)
        return attr

    def interpolate(self, *xs: List[NumberLike]) -> List[Expr]:
        return self._i.interpolate(xs)  # list(xs)

    def remainder(self, f: Function, x: Optional[Symbol] = None):
        x = x or f.free_symbols.pop()  # len(f.free_symbols)==1
        return self._i.remainder(f, x, self._epsilon)


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
        assert len(self._xs)==len(set(self._xs))
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


class Lagrange(BaseInterpolation1DMethod):
    def expression(self, name='x'):
        return self._f.subs(self._, name)

    def init(self):
        self._f = zero

    def interpolate(self, xs):
        return [self._f.subs(self._, x) for x in xs]

    def remainder(self, f, x, name='ε'):
        return f.diff(x, len(self._xs)).subs(x, name) * \
            prod((x-xk)/(i+1) for i, xk in enumerate(self._xs))

    def update(self, *_):  # 每次重新计算拟合公式
        self._f = sum(
            (y*self._lagrange(ith) for ith, y in enumerate(self._ys)), zero
        )  # .simplify()

    def _lagrange(self, kth: int) -> NumberLike:
        assert 0 <= kth < len(self._xs)
        return prod(
            (self._-x) / (self._xs[kth]-x)
            for ith, x in enumerate(self._xs) if ith!=kth
        )


class NewtonBackward(BaseInterpolation1DMethod):
    def expression(self, name='x'):
        return self._f.subs(self._, name)

    def init(self):
        self._x = self._f = self._fs = None

    def interpolate(self, xs):
        return [self._f.subs(self._, x) for x in xs]

    def remainder(self, f, x, name='ε'):
        n = len(self._xs)
        return f.diff(x, n).subs(x, name) * \
            self._x.subs(self._, x) / factorial(n)

    def update(self, xs, ys):
        # 设置初始差商表
        if self._fs is None:
            (x, *xs), (y, *ys) = xs, ys
            self._x, self._f, self._fs = self._-x, y, [[y]]
        # 计算差商表
        total_number, current_number = len(self._xs), len(xs)
        self._fs += [list() for _ in range(current_number)]
        self._fs[0] += list(ys)
        # for ith, (pf, cf) in enumerate(zip(self._fs[:-1], self._fs[1:]), start=1):
        for ith in range(1, total_number):
            pf, cf = self._fs[ith-1], self._fs[ith]
            self._fs[ith] += [
                (pf[-jth-1]-pf[-jth-2]) / (self._xs[-jth-1]-self._xs[-jth-ith-1])
                for jth in range(total_number-len(cf)-ith)
            ][::-1]
        # 根据差商表计算插值表达式
        for ith, x in enumerate(xs):
            self._f += self._fs[-current_number+ith][0] * self._x
            self._x *= self._ - x

    def table(self, precision=5, width=10):  # divided_difference_table
        length = len(self._fs)
        for ith in range(length):
            print('|', end=' ')
            for jth in range(ith+1):
                number = round(float(self._fs[jth][ith-jth]), precision)
                print(f'{number: >{width}}', end=' | ')
            print()


class NewtonForward(NewtonBackward):
    def update(self, xs, ys):
        # 设置初始差商表
        if self._fs is None:
            (*xs, x), (*ys, y) = xs, ys
            self._x, self._f, self._fs = self._-x, y, [[y]]
        # 计算差商表
        total_number, current_number = len(self._xs), len(xs)
        self._fs += [list() for _ in range(current_number)]
        self._fs[0] = list(ys) + self._fs[0]
        for ith in range(1, total_number):
            pf, cf = self._fs[ith-1], self._fs[ith]
            self._fs[ith] = [
                (pf[-jth-1]-pf[-jth-2]) / (self._xs[-jth-1]-self._xs[-jth-ith-1])
                for jth in range(total_number-len(cf)-ith)
            ][::-1] + self._fs[ith]
        # 根据差商表计算插值表达式
        for ith, x in enumerate(reversed(xs)):
            self._f += self._fs[-current_number+ith][-1] * self._x
            self._x *= self._ - x

    def table(self, precision=5, width=10):  # divided_difference_table
        length = len(self._fs)
        for ith in range(length-1, -1, -1):
            print('|', end=' ')
            for jth in range(ith+1):
                number = round(float(self._fs[jth][jth-ith-1]), precision)
                print(f'{number: >{width}}', end=' | ')
            print()


class Hermite(Lagrange):
    def add(
        self,
        data: List[Tuple[NumberLike, Tuple[NumberLike, NumberLike]]],
    ) -> None:
        xs, ys = zip(*data)
        assert all(isinstance(y, tuple) and len(y)==2 for y in ys)
        self._xs += xs
        self._ys += ys
        assert len(self._xs)==len(set(self._xs))
        self.update(xs, ys)

    def remainder(self, f, x, name='ε'):
        return f.diff(x, 2*len(self._xs)).subs(x, name) * \
            prod((x-xk)**2/(2*i+1)/(2*i+2) for i, xk in enumerate(self._xs))

    def update(self, *_):  # 每次重新计算拟合公式
        self._f = sum(
            (y*self._alphaL(ith)+m*self._betaL(ith)) * self._lagrange(ith)**2
            for ith, (y, m) in enumerate(self._ys)
        )  # .simplify()

    def _alphaL(self, jth: int) -> NumberLike:
        assert 0 <= jth < len(self._xs)
        return 1 - 2*(self._-self._xs[jth])*sum(
            one / (self._xs[jth]-x)
            for ith, x in enumerate(self._xs) if ith!=jth
        )

    def _betaL(self, jth: int) -> NumberLike:
        0 <= jth < len(self._xs)
        return self._ - self._xs[jth]


class PiecewiseLinearInterpolation(BaseInterpolation1DMethod):
    '''
    - TODO:
        - `from sympy import Piecewise`
    '''
    def expression(self, name: str = 'x') -> Dict[Tuple[Number, Number], Expr]:
        return {
            key: value.expression(name) for key, value in self._fs.items()
        }

    def init(self) -> None:
        self._fs: Dict[Tuple[Number, Number], BaseInterpolation1DMethod] = dict()

    def interpolate(self, xs: List[Number]) -> List[Expr]:
        return [self._which(x).interpolate([x])[0] for x in xs]

    def remainder(
        self,
        f: Function, x: Symbol, name: str = 'ε',
    ) -> Dict[Tuple[Number, Number], Expr]:
        return {
            key: value.remainder(f, x, name) for key, value in self._fs.items()
        }

    def update(self, *_) -> None:
        data = dict(zip(self._xs, self._ys))
        keys = sorted(data)
        for left, right in zip(keys[:-1], keys[1:]):
            self._fs[(left, right)] = Lagrange(
                [(left, data[left]), (right, data[right])]
            )

    def _which(self, x: Number) -> BaseInterpolation1DMethod:
        for (left, right), interpolation in self._fs.items():
            if left <= x <= right:
                return interpolation
        raise NotImplementedError


class PiecewiseHermite(PiecewiseLinearInterpolation):
    def add(self, data: List[Tuple[Number, Tuple[Number, Number]]]) -> None:
        xs, ys = zip(*data)
        assert all(isinstance(y, tuple) and len(y)==2 for y in ys)
        self._xs += xs
        self._ys += ys
        assert len(self._xs)==len(set(self._xs))
        self.update(xs, ys)

    def update(self, *_) -> None:
        data = dict(zip(self._xs, self._ys))
        keys = sorted(data)
        for left, right in zip(keys[:-1], keys[1:]):
            self._fs[(left, right)] = Hermite(
                [(left, data[left]), (right, data[right])]
            )


class Spline(Lagrange):
    '''
    - TODO
        - re-write `interpolating_spline`
    '''
    def remainder(self, f, x, name='ε'):
        raise NotImplementedError

    def update(self, *_) -> None:
        data = dict(zip(self._xs, self._ys))
        keys = sorted(data)
        values = [data[key] for key in keys]
        self._f = interpolating_spline(3, self._, keys, values)


if __name__ == '__main__':
    from sympy import symbols, sin, cos
    from sympy.abc import x

    f = sin(x)
    for xs in ((0.32, 0.34, 0.36), symbols('x:3')):
        i = Lagrange([(x0, f.subs(x, x0)) for x0 in xs]).wrap()
        remainder = i.remainder(f).subs(x, 0.3367)
        print(f'|{i[0.3367]}-{f.subs(x, 0.3367)}|≤{remainder}')

    for newton in (NewtonBackward, NewtonForward):
        i = newton(
            [
                (0.4, 0.41075), (0.55, 0.57815), (0.65, 0.69675),
                (0.8, 0.88811), (0.9, 1.02652), (1.05, 1.25382),
            ]
        ).wrap()
        print(newton.__name__, i[0.596])
        i.attr('table')

    f, df = sin(x), cos(x)
    i = Hermite(
        [
            (x0, (f.subs(x, x0), df.subs(x, x0)))
            for x0 in (0.32, 0.34, 0.36)
        ]
    ).wrap()
    print(i.expression.simplify())

    i = PiecewiseLinearInterpolation(
        [
            (5, 26), (4, 17), (3, 10), (2, 5), (1, 2),
        ]
    ).wrap()
    print(i.expression)

    i = PiecewiseHermite(
        [
            (1, (2, 2)), (2, (5, 4)), (3, (10, 6)),
        ]
    ).wrap()
    print(i.expression, i.remainder(x**2+1))

    i = Spline(
        [
            (5, 26), (4, 17), (3, 10), (2, 5), (1, 2),
        ]
    ).wrap()
    print(i.expression)
