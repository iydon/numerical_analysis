import textwrap

from typing import Dict, Optional, Tuple


InputType = Tuple[str, Tuple[Tuple[Optional[int], bool], ...]]


class Coder:
    '''MATLAB Coder Generator

    - Reference:
        - https://www.mathworks.com/help/coder/ref/codegen.html
        - https://www.mathworks.com/help/coder/ug/define-input-properties-programmatically-in-the-matlab-file.html
    '''
    __template__ = '''
function [{output}] = {name}({input})  %#codegen
    %{{
{docstring}
    %}}
    % pre-conditioning statements
{annotation}
    % function logic part
{body}
end
    '''.strip()

    def __init__(
        self,
        name: str = 'test',
        input: Dict[str, InputType] = dict(),
        output: Tuple[str, ...] = tuple(),
        body: str = '% TODO',
    ) -> None:
        '''
        - Arguments:
            - name: function name
            - input: {name: (type, ((dimension, variable-size?), ...))}
            - output: (name, ...)
        '''
        # pre-processing
        for arg, (type, properties) in input.items():
            # dim, inf != None, False
            assert all(dim is not None or inf for dim, inf in properties)
            # all dim (is None or >= 1)
            assert all(d is None or d>=1 for d in next(zip(*properties)))
            # len(properties) >= 2
            assert len(properties) >= 1
            if len(properties) == 1:
                input[arg] = (type, ((1, False), properties[0]))
        # private variables
        self._name = name
        self._in, self._out, self._body = input, output, body

    def render(self) -> str:
        '''Render generated MATLAB code
        '''
        indent = lambda s: textwrap.indent(s, 4*' ')
        return self.__template__.format(
            name=self._name, body=indent(self._body),
            input=', '.join(self._in), output=', '.join(self._out),
            docstring=indent('\n'.join(self._docstring())),
            annotation=indent('\n'.join(self._annotation())),
        )

    def save(self, encoding: str = 'utf-8') -> None:
        with open(self._name+'.m', 'w', encoding=encoding) as f:
            f.write(self.render())

    def _docstring(self) -> str:
        '''
        - Example:
            ```
            x = coder.typeof(double(0), [1, 7], [false, true]);
            y = coder.typeof(double(0), [1, 7], [false, true]);
            codegen test ...
                -args {x,y} -config:mex -lang:c -jit ...
                -O enable:inline -O enable:openmp ...
                -test test(rand(1, 2), rand(1, 2))
            ```
        '''
        template = '{name} = coder.typeof({type}(0), [{dim}], [{inf}]);'
        tests = list()
        for arg, (type, properties) in self._in.items():
            dim, inf = zip(*properties)
            tests.append(f'rand({", ".join(str(d or 7) for d in dim)})')
            yield template.format(
                name=arg, type=type,
                dim=', '.join(map(str, dim)).replace('None', 'Inf'),
                inf=', '.join(str(bool(i)).lower() for i in inf),
            )
        yield f'''
codegen {self._name} ...
    -args {{{','.join(self._in)}}} -config:mex -lang:c -jit ...
    -O enable:inline -O enable:openmp ...
    -test {self._name}({", ".join(tests)})
        '''.strip()

    def _annotation(self) -> str:
        '''
        - Example:
            ```
            assert(isa(x, 'double') && size(x, 1)==1 && size(x, 2)<=7);
            assert(isa(y, 'double') && size(y, 1)==1 && size(y, 2)<=7);
            ```
        '''
        for arg, (type, properties) in self._in.items():
            parts = [f'isa({arg}, {repr(type)})']
            for ith, (dim, inf) in enumerate(properties):
                rel = ('>=' if dim is None else '<=') if inf else '=='
                dim = '0' if dim is None else str(dim)
                parts.append(
                    f'size({arg}, {ith+1}){rel}{dim}'
                )
            yield f'assert({" && ".join(parts)});'


class type:
    '''MATLAB Coder type wrapper
    '''
    @staticmethod
    def double(*dims: Tuple[int]) -> InputType:
        '''Double with dims

        - Note:
            - match dim: 0->None, +=>exact, -=>unbounded
        '''
        return (
            'double', tuple(
                (abs(dim), dim<0) if dim!=0 else (None, True)
                for dim in (dims or (1, ))
            )
        )


if __name__ == '__main__':
    coder = Coder(
        name='test',
        input={
            'x': type.double(-7),
            'y': type.double(-7),
            'z': type.double(-7, -7),
        },
        output=('t', ),
        body='''
t = sum(sum(x'*y*z));
        '''.strip()
    )
    print(coder.render())
