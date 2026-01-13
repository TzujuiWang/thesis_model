import numpy as np
import copy
from typing import List, Dict, Union, Optional, Tuple, Any
from .primitives import Primitive, Function, Terminal, PrimitiveRegistry


# --- 节点定义 ---

class GPNode:
    """GP 树节点基类"""
    depth: int = 0
    parent: Optional['FunctionNode'] = None  # 用于反向追溯 (可选，但在变异时很有用)

    def evaluate(self, context: Dict[str, float]) -> float:
        raise NotImplementedError

    def get_depth(self) -> int:
        raise NotImplementedError

    def size(self) -> int:
        """子树节点总数 (包括自身)"""
        raise NotImplementedError

    def clone(self) -> 'GPNode':
        """深拷贝"""
        raise NotImplementedError

    def get_all_nodes(self) -> List['GPNode']:
        """获取子树下所有节点的列表 (用于随机选择交叉点)"""
        raise NotImplementedError


class FunctionNode(GPNode):
    def __init__(self, primitive: Function, children: List[GPNode]):
        self.primitive = primitive
        self.children = children
        for child in children:
            child.parent = self  # 维护父指针

    def evaluate(self, context: Dict[str, float]) -> float:
        # [Performance Note] 在生产环境中，此处可以用 list comprehension 或 map 加速
        vals = [c.evaluate(context) for c in self.children]
        return self.primitive.apply(vals)

    def get_depth(self) -> int:
        if not self.children: return 1
        return 1 + max(c.get_depth() for c in self.children)

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)

    def clone(self) -> 'FunctionNode':
        # 递归深拷贝
        new_children = [c.clone() for c in self.children]
        return FunctionNode(self.primitive, new_children)

    def get_all_nodes(self) -> List[GPNode]:
        nodes = [self]
        for c in self.children:
            nodes.extend(c.get_all_nodes())
        return nodes

    def replace_child(self, old_child: GPNode, new_child: GPNode):
        """[Step 2 Utility] 用于交叉/变异操作中替换子树"""
        for i, child in enumerate(self.children):
            if child is old_child:
                self.children[i] = new_child
                new_child.parent = self
                return
        raise ValueError("Child node not found in replace_child")

    def __repr__(self):
        args = ", ".join([repr(c) for c in self.children])
        return f"{self.primitive.name}({args})"


class TerminalNode(GPNode):
    def __init__(self, primitive: Terminal):
        self.primitive = primitive

    def evaluate(self, context: Dict[str, float]) -> float:
        # [Fix C] 严禁静默失败。
        # 如果 context 缺变量，必须立刻炸掉，否则 GP 进化会基于错误数据。
        try:
            return context[self.primitive.name]
        except KeyError:
            raise KeyError(
                f"GP Evaluation Critical Error: Variable '{self.primitive.name}' "
                f"missing in context. Available keys: {list(context.keys())}"
            )

    def get_depth(self) -> int: return 1

    def size(self) -> int: return 1

    def clone(self) -> 'TerminalNode': return TerminalNode(self.primitive)

    def get_all_nodes(self) -> List[GPNode]: return [self]

    def __repr__(self): return self.primitive.name


class ConstantNode(GPNode):
    def __init__(self, value: float):
        self.value = value

    def evaluate(self, context: Dict[str, float]) -> float: return self.value

    def get_depth(self) -> int: return 1

    def size(self) -> int: return 1

    def clone(self) -> 'ConstantNode': return ConstantNode(self.value)

    def get_all_nodes(self) -> List[GPNode]: return [self]

    def __repr__(self): return f"{self.value:.4f}"


# --- 工厂 ---

class GPTreeFactory:
    """
    负责 GP 树的随机生成。
    处理了 ERC (Ephemeral Random Constant) 的特殊逻辑。
    """

    def __init__(self,
                 function_set_names: List[str],
                 terminal_set_names: List[str],
                 config: Dict[str, Any],  # 注入 learning_gp.config.init_params
                 rng: np.random.Generator):

        # 1. 过滤掉 "con" 和 "R"，它们不是真正的 Function
        #    [Alignment] Yeh & Yang 用 "R", Dai 用 "con"
        self.func_names = [n for n in function_set_names if n not in ("con", "R")]
        self.funcs = [PrimitiveRegistry.get_function(n) for n in self.func_names]

        self.terminals = [PrimitiveRegistry.get_terminal(n) for n in terminal_set_names]
        self.rng = rng

        # 2. 识别是否启用 ERC
        self.use_const = ("con" in function_set_names) or ("R" in function_set_names)

        # 3. 读取配置 (不硬编码)
        self.const_prob = config.get('const_prob', 0.2)
        self.grow_terminate_prob = config.get('grow_terminate_prob', 0.4)
        c_range = config.get('const_range', [-10.0, 10.0])
        self.const_range = tuple(c_range)

    def create_random_tree(self, max_depth: int, method: str = "grow") -> GPNode:
        """
        :param max_depth: 最大深度 (0-indexed). e.g., 4 means 5 levels (root=0 to leaf=4)
        """
        return self._grow_recursive(0, max_depth, method)

    def _grow_recursive(self, current_depth: int, max_depth: int, method: str) -> GPNode:
        must_terminate = (current_depth >= max_depth)

        # Grow 模式下，非根节点有一定概率提前终止
        can_terminate = (method == 'grow') and (current_depth > 0)

        should_terminate = must_terminate or \
                           (can_terminate and self.rng.random() < self.grow_terminate_prob)

        if should_terminate:
            return self._create_terminal_or_const()
        else:
            # 随机选择一个 Function
            func = self.rng.choice(self.funcs)
            children = [
                self._grow_recursive(current_depth + 1, max_depth, method)
                for _ in range(func.arity)
            ]
            return FunctionNode(func, children)

    def _create_terminal_or_const(self) -> GPNode:
        # 决定生成常数还是变量
        if self.use_const and self.rng.random() < self.const_prob:
            val = self.rng.uniform(*self.const_range)
            return ConstantNode(val)
        else:
            term = self.rng.choice(self.terminals)
            return TerminalNode(term)