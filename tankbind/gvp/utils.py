import inspect
import re
import sys
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Type, Union, Tuple

from torch import Tensor
import torch
import tankbind.gvp.typing as typing
import functools
import torch_scatter
import warnings
from typing import Literal

class Parameter(NamedTuple):
    name: str
    type: Type
    type_repr: str
    default: Any


class Signature(NamedTuple):
    param_dict: Dict[str, Parameter]
    return_type: Type
    return_type_repr: str


class Inspector:
    r"""Inspects a given class and collects information about its instance
    methods.

    Args:
        cls (Type): The class to inspect.
    """
    def __init__(self, cls: Type):
        self._cls = cls
        self._signature_dict: Dict[str, Signature] = {}
        self._source_dict: Dict[str, str] = {}

    @property
    def _globals(self) -> Dict[str, Any]:
        return sys.modules[self._cls.__module__].__dict__

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self._cls.__name__})'

    def eval_type(self, value: Any) -> Type:
        r"""Returns the type hint of a string."""
        return eval_type(value, self._globals)

    def type_repr(self, obj: Any) -> str:
        r"""Returns the type hint representation of an object."""
        return type_repr(obj, self._globals)

    def implements(self, func_name: str) -> bool:
        r"""Returns :obj:`True` in case the inspected class implements the
        :obj:`func_name` method.

        Args:
            func_name (str): The function name to check for existence.
        """
        func = getattr(self._cls, func_name, None)
        if not callable(func):
            return False
        return not getattr(func, '__isabstractmethod__', False)

    # Inspecting Method Signatures ############################################

    def inspect_signature(
        self,
        func: Union[Callable, str],
        exclude: Optional[List[Union[str, int]]] = None,
    ) -> Signature:
        r"""Inspects the function signature of :obj:`func` and returns a tuple
        of parameter types and return type.

        Args:
            func (callabel or str): The function.
            exclude (list[int or str]): A list of parameters to exclude, either
                given by their name or index. (default: :obj:`None`)
        """
        if isinstance(func, str):
            func = getattr(self._cls, func)
        assert callable(func)

        if func.__name__ in self._signature_dict:
            return self._signature_dict[func.__name__]

        signature = inspect.signature(func)
        params = [p for p in signature.parameters.values() if p.name != 'self']

        param_dict: Dict[str, Parameter] = {}
        for i, param in enumerate(params):
            if exclude is not None and (i in exclude or param.name in exclude):
                continue

            param_type = param.annotation
            # Mimic TorchScript to auto-infer `Tensor` on non-present types:
            param_type = Tensor if param_type is inspect._empty else param_type

            param_dict[param.name] = Parameter(
                name=param.name,
                type=self.eval_type(param_type),
                type_repr=self.type_repr(param_type),
                default=param.default,
            )

        return_type = signature.return_annotation
        # Mimic TorchScript to auto-infer `Tensor` on non-present types:
        return_type = Tensor if return_type is inspect._empty else return_type

        self._signature_dict[func.__name__] = Signature(
            param_dict=param_dict,
            return_type=self.eval_type(return_type),
            return_type_repr=self.type_repr(return_type),
        )

        return self._signature_dict[func.__name__]

    def get_signature(
        self,
        func: Union[Callable, str],
        exclude: Optional[List[str]] = None,
    ) -> Signature:
        r"""Returns the function signature of the inspected function
        :obj:`func`.

        Args:
            func (callabel or str): The function.
            exclude (list[str], optional): The parameter names to exclude.
                (default: :obj:`None`)
        """
        func_name = func if isinstance(func, str) else func.__name__
        signature = self._signature_dict.get(func_name)
        if signature is None:
            raise IndexError(f"Could not access signature for function "
                             f"'{func_name}'. Did you forget to inspect it?")

        if exclude is None:
            return signature

        param_dict = {
            name: param
            for name, param in signature.param_dict.items()
            if name not in exclude
        }
        return Signature(
            param_dict=param_dict,
            return_type=signature.return_type,
            return_type_repr=signature.return_type_repr,
        )

    def remove_signature(
        self,
        func: Union[Callable, str],
    ) -> Optional[Signature]:
        r"""Removes the inspected function signature :obj:`func`.

        Args:
            func (callabel or str): The function.
        """
        func_name = func if isinstance(func, str) else func.__name__
        return self._signature_dict.pop(func_name, None)

    def get_param_dict(
        self,
        func: Union[Callable, str],
        exclude: Optional[List[str]] = None,
    ) -> Dict[str, Parameter]:
        r"""Returns the parameters of the inspected function :obj:`func`.

        Args:
            func (str or callable): The function.
            exclude (list[str], optional): The parameter names to exclude.
                (default: :obj:`None`)
        """
        return self.get_signature(func, exclude).param_dict

    def get_params(
        self,
        func: Union[Callable, str],
        exclude: Optional[List[str]] = None,
    ) -> List[Parameter]:
        r"""Returns the parameters of the inspected function :obj:`func`.

        Args:
            func (str or callable): The function.
            exclude (list[str], optional): The parameter names to exclude.
                (default: :obj:`None`)
        """
        return list(self.get_param_dict(func, exclude).values())

    def get_flat_param_dict(
        self,
        funcs: List[Union[Callable, str]],
        exclude: Optional[List[str]] = None,
    ) -> Dict[str, Parameter]:
        r"""Returns the union of parameters of all inspected functions in
        :obj:`funcs`.

        Args:
            funcs (list[str or callable]): The functions.
            exclude (list[str], optional): The parameter names to exclude.
                (default: :obj:`None`)
        """
        param_dict: Dict[str, Parameter] = {}
        for func in funcs:
            params = self.get_params(func, exclude)
            for param in params:
                expected = param_dict.get(param.name)
                if expected is not None and param.type != expected.type:
                    raise ValueError(f"Found inconsistent types for argument "
                                     f"'{param.name}'. Expected type "
                                     f"'{expected.type}' but found type "
                                     f"'{param.type}'.")

                if expected is not None and param.default != expected.default:
                    if (param.default is not inspect._empty
                            and expected.default is not inspect._empty):
                        raise ValueError(f"Found inconsistent defaults for "
                                         f"argument '{param.name}'. Expected "
                                         f"'{expected.default}'  but found "
                                         f"'{param.default}'.")

                    default = expected.default
                    if default is inspect._empty:
                        default = param.default

                    param_dict[param.name] = Parameter(
                        name=param.name,
                        type=param.type,
                        type_repr=param.type_repr,
                        default=default,
                    )

                if expected is None:
                    param_dict[param.name] = param

        return param_dict

    def get_flat_params(
        self,
        funcs: List[Union[Callable, str]],
        exclude: Optional[List[str]] = None,
    ) -> List[Parameter]:
        r"""Returns the union of parameters of all inspected functions in
        :obj:`funcs`.

        Args:
            funcs (list[str or callable]): The functions.
            exclude (list[str], optional): The parameter names to exclude.
                (default: :obj:`None`)
        """
        return list(self.get_flat_param_dict(funcs, exclude).values())

    def get_param_names(
        self,
        func: Union[Callable, str],
        exclude: Optional[List[str]] = None,
    ) -> List[str]:
        r"""Returns the parameter names of the inspected function :obj:`func`.

        Args:
            func (str or callable): The function.
            exclude (list[str], optional): The parameter names to exclude.
                (default: :obj:`None`)
        """
        return list(self.get_param_dict(func, exclude).keys())

    def get_flat_param_names(
        self,
        funcs: List[Union[Callable, str]],
        exclude: Optional[List[str]] = None,
    ) -> List[str]:
        r"""Returns the union of parameter names of all inspected functions in
        :obj:`funcs`.

        Args:
            funcs (list[str or callable]): The functions.
            exclude (list[str], optional): The parameter names to exclude.
                (default: :obj:`None`)
        """
        return list(self.get_flat_param_dict(funcs, exclude).keys())

    def collect_param_data(
        self,
        func: Union[Callable, str],
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        r"""Collects the input data of the inspected function :obj:`func`
        according to its function signature from a data blob.

        Args:
            func (callabel or str): The function.
            kwargs (dict[str, Any]): The data blob which may serve as inputs.
        """
        out_dict: Dict[str, Any] = {}
        for param in self.get_params(func):
            if param.name not in kwargs:
                if param.default is inspect._empty:
                    raise TypeError(f"Parameter '{param.name}' is required")
                out_dict[param.name] = param.default
            else:
                out_dict[param.name] = kwargs[param.name]
        return out_dict

    # Inspecting Method Bodies ################################################

    @property
    def can_read_source(self) -> bool:
        r"""Returns :obj:`True` if able to read the source file of the
        inspected class.
        """
        try:
            inspect.getfile(self._cls)
            return True
        except Exception:
            return False

    def get_source(self, cls: Optional[Type] = None) -> str:
        r"""Returns the source code of :obj:`cls`."""
        cls = cls or self._cls
        if cls.__name__ in self._source_dict:
            return self._source_dict[cls.__name__]
        try:
            source = inspect.getsource(cls)
        except Exception:
            source = ''
        self._source_dict[cls.__name__] = source
        return source

    def get_params_from_method_call(
        self,
        func: Union[Callable, str],
        exclude: Optional[List[Union[int, str]]] = None,
    ) -> Dict[str, Parameter]:
        r"""Parses a method call of :obj:`func` and returns its keyword
        arguments.

        .. note::
            The method is required to be called via keyword arguments in case
            type annotations are not found.

        Args:
            func (callabel or str): The function.
            exclude (list[int or str]): A list of parameters to exclude, either
                given by their name or index. (default: :obj:`None`)
        """
        func_name = func if isinstance(func, str) else func.__name__
        param_dict: Dict[str, Parameter] = {}

        # Three ways to specify the parameters of an unknown function header:
        # 1. Defined as class attributes in `{func_name}_type`.
        # 2. Defined via type annotations in `# {func_name}_type: (...)`.
        # 3. Defined via parsing of the function call.

        # (1) Find class attribute:
        if hasattr(self._cls, f'{func_name}_type'):
            type_dict = getattr(self._cls, f'{func_name}_type')
            if not isinstance(type_dict, dict):
                raise ValueError(f"'{func_name}_type' is expected to be a "
                                 f"dictionary (got '{type(type_dict)}')")

            for name, param_type in type_dict.items():
                param_dict[name] = Parameter(
                    name=name,
                    type=self.eval_type(param_type),
                    type_repr=self.type_repr(param_type),
                    default=inspect._empty,
                )
            return param_dict

        # (2) Find type annotation:
        for cls in self._cls.__mro__:
            source = self.get_source(cls)
            match = find_parenthesis_content(source, f'{func_name}_type:')
            if match is not None:
                for arg in split(match, sep=','):
                    name_and_type_repr = re.split(r'\s*:\s*', arg)
                    if len(name_and_type_repr) != 2:
                        raise ValueError(f"Could not parse argument '{arg}' "
                                         f"of '{func_name}_type' annotation")

                    name, type_repr = name_and_type_repr
                    param_dict[name] = Parameter(
                        name=name,
                        type=self.eval_type(type_repr),
                        type_repr=type_repr,
                        default=inspect._empty,
                    )
                return param_dict

        # (3) Parse the function call:
        for cls in self._cls.__mro__:
            source = self.get_source(cls)
            match = find_parenthesis_content(source, f'self.{func_name}')
            if match is not None:
                for i, kwarg in enumerate(split(match, sep=',')):
                    if exclude is not None and i in exclude:
                        continue

                    name_and_content = re.split(r'\s*=\s*', kwarg)
                    if len(name_and_content) != 2:
                        raise ValueError(f"Could not parse keyword argument "
                                         f"'{kwarg}' in 'self.{func_name}()'")

                    name, _ = name_and_content

                    if exclude is not None and name in exclude:
                        continue

                    param_dict[name] = Parameter(
                        name=name,
                        type=Tensor,
                        type_repr=self.type_repr(Tensor),
                        default=inspect._empty,
                    )
                return param_dict

        return {}  # (4) No function call found:


def eval_type(value: Any, _globals: Dict[str, Any]) -> Type:
    r"""Returns the type hint of a string."""
    if isinstance(value, str):
        value = typing.ForwardRef(value)
    return typing._eval_type(value, _globals, None)  # type: ignore


def type_repr(obj: Any, _globals: Dict[str, Any]) -> str:
    r"""Returns the type hint representation of an object."""
    def _get_name(name: str, module: str) -> str:
        return name if name in _globals else f'{module}.{name}'

    if isinstance(obj, str):
        return obj

    if obj is type(None):
        return 'None'

    if obj is ...:
        return '...'

    if obj.__module__ == 'typing':  # Special logic for `typing.*` types:
        name = obj._name
        if name is None:  # In some cases, `_name` is not populated.
            name = str(obj.__origin__).split('.')[-1]

        args = getattr(obj, '__args__', None)
        if args is None or len(args) == 0:
            return _get_name(name, obj.__module__)
        if all(isinstance(arg, typing.TypeVar) for arg in args):
            return _get_name(name, obj.__module__)

        # Convert `Union[*, None]` to `Optional[*]`.
        # This is only necessary for old Python versions, e.g. 3.8.
        # TODO Only convert to `Optional` if `Optional` is importable.
        if (name == 'Union' and len(args) == 2
                and any([arg is type(None) for arg in args])):
            name = 'Optional'

        if name == 'Optional':  # Remove `None` from `Optional` arguments:
            args = [arg for arg in obj.__args__ if arg is not type(None)]

        args_repr = ', '.join([type_repr(arg, _globals) for arg in args])
        return f'{_get_name(name, obj.__module__)}[{args_repr}]'

    if obj.__module__ == 'builtins':
        return obj.__qualname__

    return _get_name(obj.__qualname__, obj.__module__)


def find_parenthesis_content(source: str, prefix: str) -> Optional[str]:
    r"""Returns the content of :obj:`{prefix}.*(...)` within :obj:`source`."""
    match = re.search(prefix, source)
    if match is None:
        return None

    offset = source[match.start():].find('(')
    if offset < 0:
        return None

    source = source[match.start() + offset:]

    depth = 0
    for end, char in enumerate(source):
        if char == '(':
            depth += 1
        if char == ')':
            depth -= 1
        if depth == 0:
            content = source[1:end]
            # Properly handle line breaks and multiple white-spaces:
            content = content.replace('\n', ' ')
            content = content.replace('#', ' ')
            content = re.sub(' +', ' ', content)
            content = content.strip()
            return content

    return None


def split(content: str, sep: str) -> List[str]:
    r"""Splits :obj:`content` based on :obj:`sep`.
    :obj:`sep` inside parentheses or square brackets are ignored.
    """
    assert len(sep) == 1
    outs: List[str] = []

    start = depth = 0
    for end, char in enumerate(content):
        if char == '[' or char == '(':
            depth += 1
        elif char == ']' or char == ')':
            depth -= 1
        elif char == sep and depth == 0:
            outs.append(content[start:end].strip())
            start = end + 1
    if start != len(content):  # Respect dangling `sep`:
        outs.append(content[start:].strip())
    return outs



def is_compiling() -> bool:
    r"""Returns :obj:`True` in case :pytorch:`PyTorch` is compiling via
    :meth:`torch.compile`.
    """
    if typing.WITH_PT21:
        return torch._dynamo.is_compiling()
    return False  # pragma: no cover

Options = Optional[Union[str, List[str]]]
__experimental_flag__: Dict[str, bool] = {
    'disable_dynamic_shapes': False,
}

def get_options(options: Options) -> List[str]:
    if options is None:
        options = list(__experimental_flag__.keys())
    if isinstance(options, str):
        options = [options]
    return options

def is_experimental_mode_enabled(options: Options = None) -> bool:
    r"""Returns :obj:`True` if the experimental mode is enabled. See
    :class:`torch_geometric.experimental_mode` for a list of (optional)
    options.
    """
    if torch.jit.is_scripting() or torch.jit.is_tracing():
        return False
    options = get_options(options)
    return all([__experimental_flag__[option] for option in options])

def disable_dynamic_shapes(required_args: List[str]) -> Callable:
    r"""A decorator that disables the usage of dynamic shapes for the given
    arguments, i.e., it will raise an error in case :obj:`required_args` are
    not passed and needs to be automatically inferred.
    """
    def decorator(func: Callable) -> Callable:
        spec = inspect.getfullargspec(func)

        required_args_pos: Dict[str, int] = {}
        for arg_name in required_args:
            if arg_name not in spec.args:
                raise ValueError(f"The function '{func}' does not have a "
                                 f"'{arg_name}' argument")
            required_args_pos[arg_name] = spec.args.index(arg_name)

        num_args = len(spec.args)
        num_default_args = 0 if spec.defaults is None else len(spec.defaults)
        num_positional_args = num_args - num_default_args

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not is_experimental_mode_enabled('disable_dynamic_shapes'):
                return func(*args, **kwargs)

            for required_arg in required_args:
                index = required_args_pos[required_arg]

                value: Optional[Any] = None
                if index < len(args):
                    value = args[index]
                elif required_arg in kwargs:
                    value = kwargs[required_arg]
                elif num_default_args > 0:
                    assert spec.defaults is not None
                    value = spec.defaults[index - num_positional_args]

                if value is None:
                    raise ValueError(f"Dynamic shapes disabled. Argument "
                                     f"'{required_arg}' needs to be set")

            return func(*args, **kwargs)

        return wrapper

    return decorator



def warn(message: str) -> None:
    if is_compiling():
        return

    warnings.warn(message)


def filterwarnings(
    action: Literal['default', 'error', 'ignore', 'always', 'module', 'once'],
    message: str,
) -> None:
    if is_compiling():
        return

    warnings.filterwarnings(action, message)

def broadcast(src: Tensor, ref: Tensor, dim: int) -> Tensor:
    size = ((1, ) * dim) + (-1, ) + ((1, ) * (ref.dim() - dim - 1))
    return src.view(size).expand_as(ref)

if typing.WITH_PT112:  # pragma: no cover

    warnings.filterwarnings('ignore', '.*is in beta and the API may change.*')

    def scatter(
        src: Tensor,
        index: Tensor,
        dim: int = 0,
        dim_size: Optional[int] = None,
        reduce: str = 'sum',
    ) -> Tensor:
        r"""Reduces all values from the :obj:`src` tensor at the indices
        specified in the :obj:`index` tensor along a given dimension
        :obj:`dim`. See the `documentation
        <https://pytorch-scatter.readthedocs.io/en/latest/functions/
        scatter.html>`__ of the :obj:`torch_scatter` package for more
        information.

        Args:
            src (torch.Tensor): The source tensor.
            index (torch.Tensor): The index tensor.
            dim (int, optional): The dimension along which to index.
                (default: :obj:`0`)
            dim_size (int, optional): The size of the output tensor at
                dimension :obj:`dim`. If set to :obj:`None`, will create a
                minimal-sized output tensor according to
                :obj:`index.max() + 1`. (default: :obj:`None`)
            reduce (str, optional): The reduce operation (:obj:`"sum"`,
                :obj:`"mean"`, :obj:`"mul"`, :obj:`"min"` or :obj:`"max"`,
                :obj:`"any"`). (default: :obj:`"sum"`)
        """
        if isinstance(index, Tensor) and index.dim() != 1:
            raise ValueError(f"The `index` argument must be one-dimensional "
                             f"(got {index.dim()} dimensions)")

        dim = src.dim() + dim if dim < 0 else dim

        if isinstance(src, Tensor) and (dim < 0 or dim >= src.dim()):
            raise ValueError(f"The `dim` argument must lay between 0 and "
                             f"{src.dim() - 1} (got {dim})")

        if dim_size is None:
            dim_size = int(index.max()) + 1 if index.numel() > 0 else 0

        # For now, we maintain various different code paths, based on whether
        # the input requires gradients and whether it lays on the CPU/GPU.
        # For example, `torch_scatter` is usually faster than
        # `torch.scatter_reduce` on GPU, while `torch.scatter_reduce` is faster
        # on CPU.
        # `torch.scatter_reduce` has a faster forward implementation for
        # "min"/"max" reductions since it does not compute additional arg
        # indices, but is therefore way slower in its backward implementation.
        # More insights can be found in `test/utils/test_scatter.py`.

        size = src.size()[:dim] + (dim_size, ) + src.size()[dim + 1:]

        # For "any" reduction, we use regular `scatter_`:
        if reduce == 'any':
            index = broadcast(index, src, dim)
            return src.new_zeros(size).scatter_(dim, index, src)

        # For "sum" and "mean" reduction, we make use of `scatter_add_`:
        if reduce == 'sum' or reduce == 'add':
            index = broadcast(index, src, dim)
            return src.new_zeros(size).scatter_add_(dim, index, src)

        if reduce == 'mean':
            count = src.new_zeros(dim_size)
            count.scatter_add_(0, index, src.new_ones(src.size(dim)))
            count = count.clamp(min=1)

            index = broadcast(index, src, dim)
            out = src.new_zeros(size).scatter_add_(dim, index, src)

            return out / broadcast(count, out, dim)

        # For "min" and "max" reduction, we prefer `scatter_reduce_` on CPU or
        # in case the input does not require gradients:
        if reduce in ['min', 'max', 'amin', 'amax']:
            if (not typing.WITH_TORCH_SCATTER
                    or is_compiling() or not src.is_cuda
                    or not src.requires_grad):

                if src.is_cuda and src.requires_grad and not is_compiling():
                    warnings.warn(f"The usage of `scatter(reduce='{reduce}')` "
                                  f"can be accelerated via the 'torch-scatter'"
                                  f" package, but it was not found")

                index = broadcast(index, src, dim)
                return src.new_zeros(size).scatter_reduce_(
                    dim, index, src, reduce=f'a{reduce[-3:]}',
                    include_self=False)

            return torch_scatter.scatter(src, index, dim, dim_size=dim_size,
                                         reduce=reduce[-3:])

        # For "mul" reduction, we prefer `scatter_reduce_` on CPU:
        if reduce == 'mul':
            if (not typing.WITH_TORCH_SCATTER
                    or is_compiling() or not src.is_cuda):

                if src.is_cuda and not is_compiling():
                    warnings.warn(f"The usage of `scatter(reduce='{reduce}')` "
                                  f"can be accelerated via the 'torch-scatter'"
                                  f" package, but it was not found")

                index = broadcast(index, src, dim)
                # We initialize with `one` here to match `scatter_mul` output:
                return src.new_ones(size).scatter_reduce_(
                    dim, index, src, reduce='prod', include_self=True)

            return torch_scatter.scatter(src, index, dim, dim_size=dim_size,
                                         reduce='mul')

        raise ValueError(f"Encountered invalid `reduce` argument '{reduce}'")

else:  # pragma: no cover

    def scatter(
        src: Tensor,
        index: Tensor,
        dim: int = 0,
        dim_size: Optional[int] = None,
        reduce: str = 'sum',
    ) -> Tensor:
        r"""Reduces all values from the :obj:`src` tensor at the indices
        specified in the :obj:`index` tensor along a given dimension
        :obj:`dim`. See the `documentation
        <https://pytorch-scatter.readthedocs.io/en/latest/functions/
        scatter.html>`_ of the :obj:`torch_scatter` package for more
        information.

        Args:
            src (torch.Tensor): The source tensor.
            index (torch.Tensor): The index tensor.
            dim (int, optional): The dimension along which to index.
                (default: :obj:`0`)
            dim_size (int, optional): The size of the output tensor at
                dimension :obj:`dim`. If set to :obj:`None`, will create a
                minimal-sized output tensor according to
                :obj:`index.max() + 1`. (default: :obj:`None`)
            reduce (str, optional): The reduce operation (:obj:`"sum"`,
                :obj:`"mean"`, :obj:`"mul"`, :obj:`"min"` or :obj:`"max"`,
                :obj:`"any"`). (default: :obj:`"sum"`)
        """
        if reduce == 'any':
            dim = src.dim() + dim if dim < 0 else dim

            if dim_size is None:
                dim_size = int(index.max()) + 1 if index.numel() > 0 else 0

            size = src.size()[:dim] + (dim_size, ) + src.size()[dim + 1:]

            index = broadcast(index, src, dim)
            return src.new_zeros(size).scatter_(dim, index, src)

        if not typing.WITH_TORCH_SCATTER:
            raise ImportError("'scatter' requires the 'torch-scatter' package")

        if reduce == 'amin' or reduce == 'amax':
            reduce = reduce[-3:]

        return torch_scatter.scatter(src, index, dim, dim_size=dim_size,
                                     reduce=reduce)


def cumsum(x: Tensor, dim: int = 0) -> Tensor:
    r"""Returns the cumulative sum of elements of :obj:`x`.
    In contrast to :meth:`torch.cumsum`, prepends the output with zero.

    Args:
        x (torch.Tensor): The input tensor.
        dim (int, optional): The dimension to do the operation over.
            (default: :obj:`0`)

    Example:
        >>> x = torch.tensor([2, 4, 1])
        >>> cumsum(x)
        tensor([0, 2, 6, 7])

    """
    size = x.size()[:dim] + (x.size(dim) + 1, ) + x.size()[dim + 1:]
    out = x.new_empty(size)

    out.narrow(dim, 0, 1).zero_()
    torch.cumsum(x, dim=dim, out=out.narrow(dim, 1, x.size(dim)))

    return out

@disable_dynamic_shapes(required_args=['batch_size', 'max_num_nodes'])
def to_dense_batch(
    x: Tensor,
    batch: Optional[Tensor] = None,
    fill_value: float = 0.0,
    max_num_nodes: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    r"""Given a sparse batch of node features
    :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}` (with
    :math:`N_i` indicating the number of nodes in graph :math:`i`), creates a
    dense node feature tensor
    :math:`\mathbf{X} \in \mathbb{R}^{B \times N_{\max} \times F}` (with
    :math:`N_{\max} = \max_i^B N_i`).
    In addition, a mask of shape :math:`\mathbf{M} \in \{ 0, 1 \}^{B \times
    N_{\max}}` is returned, holding information about the existence of
    fake-nodes in the dense representation.

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. Must be ordered. (default: :obj:`None`)
        fill_value (float, optional): The value for invalid entries in the
            resulting dense output tensor. (default: :obj:`0`)
        max_num_nodes (int, optional): The size of the output node dimension.
            (default: :obj:`None`)
        batch_size (int, optional): The batch size. (default: :obj:`None`)

    :rtype: (:class:`Tensor`, :class:`BoolTensor`)

    Examples:
        >>> x = torch.arange(12).view(6, 2)
        >>> x
        tensor([[ 0,  1],
                [ 2,  3],
                [ 4,  5],
                [ 6,  7],
                [ 8,  9],
                [10, 11]])

        >>> out, mask = to_dense_batch(x)
        >>> mask
        tensor([[True, True, True, True, True, True]])

        >>> batch = torch.tensor([0, 0, 1, 2, 2, 2])
        >>> out, mask = to_dense_batch(x, batch)
        >>> out
        tensor([[[ 0,  1],
                [ 2,  3],
                [ 0,  0]],
                [[ 4,  5],
                [ 0,  0],
                [ 0,  0]],
                [[ 6,  7],
                [ 8,  9],
                [10, 11]]])
        >>> mask
        tensor([[ True,  True, False],
                [ True, False, False],
                [ True,  True,  True]])

        >>> out, mask = to_dense_batch(x, batch, max_num_nodes=4)
        >>> out
        tensor([[[ 0,  1],
                [ 2,  3],
                [ 0,  0],
                [ 0,  0]],
                [[ 4,  5],
                [ 0,  0],
                [ 0,  0],
                [ 0,  0]],
                [[ 6,  7],
                [ 8,  9],
                [10, 11],
                [ 0,  0]]])

        >>> mask
        tensor([[ True,  True, False, False],
                [ True, False, False, False],
                [ True,  True,  True, False]])
    """
    if batch is None and max_num_nodes is None:
        mask = torch.ones(1, x.size(0), dtype=torch.bool, device=x.device)
        return x.unsqueeze(0), mask

    if batch is None:
        batch = x.new_zeros(x.size(0), dtype=torch.long)

    if batch_size is None:
        batch_size = int(batch.max()) + 1

    num_nodes = scatter(batch.new_ones(x.size(0)), batch, dim=0,
                        dim_size=batch_size, reduce='sum')
    cum_nodes = cumsum(num_nodes)

    filter_nodes = False
    dynamic_shapes_disabled = is_experimental_mode_enabled(
        'disable_dynamic_shapes')

    if max_num_nodes is None:
        max_num_nodes = int(num_nodes.max())
    elif not dynamic_shapes_disabled and num_nodes.max() > max_num_nodes:
        filter_nodes = True

    tmp = torch.arange(batch.size(0), device=x.device) - cum_nodes[batch]
    idx = tmp + (batch * max_num_nodes)
    if filter_nodes:
        mask = tmp < max_num_nodes
        x, idx = x[mask], idx[mask]

    size = [batch_size * max_num_nodes] + list(x.size())[1:]
    out = torch.as_tensor(fill_value, device=x.device)
    out = out.to(x.dtype).repeat(size)
    out[idx] = x
    out = out.view([batch_size, max_num_nodes] + list(x.size())[1:])

    mask = torch.zeros(batch_size * max_num_nodes, dtype=torch.bool,
                       device=x.device)
    mask[idx] = 1
    mask = mask.view(batch_size, max_num_nodes)

    return out, mask

def segment(src: Tensor, ptr: Tensor, reduce: str = 'sum') -> Tensor:
    r"""Reduces all values in the first dimension of the :obj:`src` tensor
    within the ranges specified in the :obj:`ptr`. See the `documentation
    <https://pytorch-scatter.readthedocs.io/en/latest/functions/
    segment_csr.html>`__ of the :obj:`torch_scatter` package for more
    information.

    Args:
        src (torch.Tensor): The source tensor.
        ptr (torch.Tensor): A monotonically increasing pointer tensor that
            refers to the boundaries of segments such that :obj:`ptr[0] = 0`
            and :obj:`ptr[-1] = src.size(0)`.
        reduce (str, optional): The reduce operation (:obj:`"sum"`,
            :obj:`"mean"`, :obj:`"mul"`, :obj:`"min"` or :obj:`"max"`).
            (default: :obj:`"sum"`)
    """
    if not typing.WITH_TORCH_SCATTER or is_compiling():
        raise ImportError("'segment' requires the 'torch-scatter' package")
    return torch_scatter.segment_csr(src, ptr, reduce=reduce)


class Aggregation(torch.nn.Module):
    r"""An abstract base class for implementing custom aggregations.

    Aggregation can be either performed via an :obj:`index` vector, which
    defines the mapping from input elements to their location in the output:

    |

    .. image:: https://raw.githubusercontent.com/rusty1s/pytorch_scatter/
            master/docs/source/_figures/add.svg?sanitize=true
        :align: center
        :width: 400px

    |

    Notably, :obj:`index` does not have to be sorted (for most aggregation
    operators):

    .. code-block::

       # Feature matrix holding 10 elements with 64 features each:
       x = torch.randn(10, 64)

       # Assign each element to one of three sets:
       index = torch.tensor([0, 0, 1, 0, 2, 0, 2, 1, 0, 2])

       output = aggr(x, index)  #  Output shape: [3, 64]

    Alternatively, aggregation can be achieved via a "compressed" index vector
    called :obj:`ptr`. Here, elements within the same set need to be grouped
    together in the input, and :obj:`ptr` defines their boundaries:

    .. code-block::

       # Feature matrix holding 10 elements with 64 features each:
       x = torch.randn(10, 64)

       # Define the boundary indices for three sets:
       ptr = torch.tensor([0, 4, 7, 10])

       output = aggr(x, ptr=ptr)  #  Output shape: [4, 64]

    Note that at least one of :obj:`index` or :obj:`ptr` must be defined.

    Shapes:
        - **input:**
          node features :math:`(*, |\mathcal{V}|, F_{in})` or edge features
          :math:`(*, |\mathcal{E}|, F_{in})`,
          index vector :math:`(|\mathcal{V}|)` or :math:`(|\mathcal{E}|)`,
        - **output:** graph features :math:`(*, |\mathcal{G}|, F_{out})` or
          node features :math:`(*, |\mathcal{V}|, F_{out})`
    """
    def forward(
        self,
        x: Tensor,
        index: Optional[Tensor] = None,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
        max_num_elements: Optional[int] = None,
    ) -> Tensor:
        r"""Forward pass.

        Args:
            x (torch.Tensor): The source tensor.
            index (torch.Tensor, optional): The indices of elements for
                applying the aggregation.
                One of :obj:`index` or :obj:`ptr` must be defined.
                (default: :obj:`None`)
            ptr (torch.Tensor, optional): If given, computes the aggregation
                based on sorted inputs in CSR representation.
                One of :obj:`index` or :obj:`ptr` must be defined.
                (default: :obj:`None`)
            dim_size (int, optional): The size of the output tensor at
                dimension :obj:`dim` after aggregation. (default: :obj:`None`)
            dim (int, optional): The dimension in which to aggregate.
                (default: :obj:`-2`)
            max_num_elements: (int, optional): The maximum number of elements
                within a single aggregation group. (default: :obj:`None`)
        """
        pass

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        pass

    @disable_dynamic_shapes(required_args=['dim_size'])
    def __call__(
        self,
        x: Tensor,
        index: Optional[Tensor] = None,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
        **kwargs,
    ) -> Tensor:

        if dim >= x.dim() or dim < -x.dim():
            raise ValueError(f"Encountered invalid dimension '{dim}' of "
                             f"source tensor with {x.dim()} dimensions")

        if index is None and ptr is None:
            index = x.new_zeros(x.size(dim), dtype=torch.long)

        if ptr is not None:
            if dim_size is None:
                dim_size = ptr.numel() - 1
            elif dim_size != ptr.numel() - 1:
                raise ValueError(f"Encountered invalid 'dim_size' (got "
                                 f"'{dim_size}' but expected "
                                 f"'{ptr.numel() - 1}')")

        if index is not None and dim_size is None:
            dim_size = int(index.max()) + 1 if index.numel() > 0 else 0

        try:
            return super().__call__(x, index=index, ptr=ptr, dim_size=dim_size,
                                    dim=dim, **kwargs)
        except (IndexError, RuntimeError) as e:
            if index is not None:
                if index.numel() > 0 and dim_size <= int(index.max()):
                    raise ValueError(f"Encountered invalid 'dim_size' (got "
                                     f"'{dim_size}' but expected "
                                     f">= '{int(index.max()) + 1}')")
            raise e

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

    # Assertions ##############################################################

    def assert_index_present(self, index: Optional[Tensor]):
        # TODO Currently, not all aggregators support `ptr`. This assert helps
        # to ensure that we require `index` to be passed to the computation:
        if index is None:
            raise NotImplementedError(
                "Aggregation requires 'index' to be specified")

    def assert_sorted_index(self, index: Optional[Tensor]):
        if index is not None and not torch.all(index[:-1] <= index[1:]):
            raise ValueError("Can not perform aggregation since the 'index' "
                             "tensor is not sorted. Specifically, if you use "
                             "this aggregation as part of 'MessagePassing`, "
                             "ensure that 'edge_index' is sorted by "
                             "destination nodes, e.g., by calling "
                             "`data.sort(sort_by_row=False)`")

    def assert_two_dimensional_input(self, x: Tensor, dim: int):
        if x.dim() != 2:
            raise ValueError(f"Aggregation requires two-dimensional inputs "
                             f"(got '{x.dim()}')")

        if dim not in [-2, 0]:
            raise ValueError(f"Aggregation needs to perform aggregation in "
                             f"first dimension (got '{dim}')")

    # Helper methods ##########################################################

    def reduce(self, x: Tensor, index: Optional[Tensor] = None,
               ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
               dim: int = -2, reduce: str = 'sum') -> Tensor:

        if (ptr is not None and typing.WITH_TORCH_SCATTER
                and not is_compiling()):
            ptr = expand_left(ptr, dim, dims=x.dim())
            return segment(x, ptr, reduce=reduce)

        if index is None:
            raise NotImplementedError(
                "Aggregation requires 'index' to be specified")
        return scatter(x, index, dim, dim_size, reduce)

    def to_dense_batch(
        self,
        x: Tensor,
        index: Optional[Tensor] = None,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
        fill_value: float = 0.0,
        max_num_elements: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:

        # TODO Currently, `to_dense_batch` can only operate on `index`:
        self.assert_index_present(index)
        self.assert_sorted_index(index)
        self.assert_two_dimensional_input(x, dim)

        return to_dense_batch(
            x,
            index,
            batch_size=dim_size,
            fill_value=fill_value,
            max_num_nodes=max_num_elements,
        )

def expand_left(ptr: Tensor, dim: int, dims: int) -> Tensor:
    for _ in range(dims + dim if dim < 0 else dim):
        ptr = ptr.unsqueeze(0)
    return ptr

