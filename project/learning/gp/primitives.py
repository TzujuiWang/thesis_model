import numpy as np
from abc import ABC
from typing import List, Callable, Dict, Any


class Primitive(ABC):
    name: str
    arity: int

    def __repr__(self): return self.name


class Function(Primitive):
    def __init__(self, name: str, arity: int, func: Callable):
        self.name = name
        self.arity = arity
        self.func = func

    def apply(self, args: List[float]) -> float:
        return self.func(*args)


class Terminal(Primitive):
    def __init__(self, name: str):
        self.name = name
        self.arity = 0


# --- Protected Operations (Configurable) ---

class Ops:
    """静态容器，用于注入配置参数"""
    DIV_EPSILON = 1e-9
    SQRT_ROBUST = True  # Default to True (safe)

    @staticmethod
    def configure(config: Dict[str, Any]):
        if 'div_epsilon' in config:
            Ops.DIV_EPSILON = float(config['div_epsilon'])
        # [Fix A] 真正读取配置
        if 'sqrt_robust' in config:
            Ops.SQRT_ROBUST = bool(config['sqrt_robust'])

def _protected_sqrt(a: float) -> float:
    # [Fix A] 根据配置决定行为
    if Ops.SQRT_ROBUST:
        return np.sqrt(abs(a))
    else:
        # Non-robust: 负数将导致 RuntimeWarning (NaN) 或 Error，
        # 这有助于在调试阶段发现 GP 树是否产生了非法值
        return np.sqrt(a)


def _protected_div(a: float, b: float) -> float:
    if abs(b) < Ops.DIV_EPSILON:
        return 1.0
    return a / b

def _ifelse(cond: float, true_val: float, false_val: float) -> float:
    return true_val if cond >= 0 else false_val


# --- Registry ---

class PrimitiveRegistry:
    _FUNCTIONS = {
        "+": Function("+", 2, lambda x, y: x + y),
        "-": Function("-", 2, lambda x, y: x - y),
        "*": Function("*", 2, lambda x, y: x * y),
        "/": Function("/", 2, _protected_div),
        "abs": Function("abs", 1, abs),
        "sin": Function("sin", 1, np.sin),
        "cos": Function("cos", 1, np.cos),
        "sqrt": Function("sqrt", 1, _protected_sqrt),
        "ifelse": Function("ifelse", 3, _ifelse),

    }

    @staticmethod
    def get_function(name: str) -> Function:
        if name not in PrimitiveRegistry._FUNCTIONS:
            raise ValueError(f"Unknown or Invalid GP function: {name} (Note: 'con' is not a function)")
        return PrimitiveRegistry._FUNCTIONS[name]

    @staticmethod
    def get_terminal(name: str) -> Terminal:
        return Terminal(name)