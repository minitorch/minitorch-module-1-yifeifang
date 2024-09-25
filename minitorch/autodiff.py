from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

import copy
import queue

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # Convert vals to a list for easier manipulation
    vals_list = list(vals)

    # Create two copies of the input values
    vals_plus = copy.deepcopy(vals_list)
    vals_minus = copy.deepcopy(vals_list)

    # Modify the argument of interest by epsilon
    vals_plus[arg] += epsilon
    vals_minus[arg] -= epsilon

    # Compute f(x + epsilon) and f(x - epsilon)
    f_plus = f(*vals_plus)
    f_minus = f(*vals_minus)

    # Compute the central difference approximation
    derivative = (f_plus - f_minus) / (2 * epsilon)

    return derivative


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    visited = set()
    result = []

    def dfs(node):
        if node.unique_id in visited:
            return
        visited.add(node.unique_id)
        
        for parent in node.parents:
            dfs(parent)
        
        result.append(node)

    dfs(variable)
    return list(reversed(result))


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    sorted_list = topological_sort(variable)
    var_dict = {}
    var_dict[sorted_list[0].unique_id] = deriv
    for node in sorted_list:
        if node.is_leaf():
            node.accumulate_derivative(var_dict[node.unique_id])
            continue
        for d,p in zip(node.chain_rule(var_dict[node.unique_id]), node.parents):
            if p.unique_id not in var_dict:
                var_dict[p.unique_id] = 0.0
            var_dict[p.unique_id] += d[1]

@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
