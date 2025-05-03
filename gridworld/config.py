"""
配置模块 - 集中所有可调常量与枚举

此模块负责集中定义所有环境的配置参数，包括网格大小、智能体数量、
奖励/惩罚值等，避免魔法数字散落各处。
"""

from enum import Enum, auto
from typing import Tuple
import collections.abc

# 网格与代理相关常量
# -------------------------
# 网格大小，固定为5x5
GRID_SIZE = 5
# 智能体数量，固定为4
N_AGENTS = 4

# 坐标类型定义
# -------------------------
# 定义位置类型为二元整数元组 (行,列)
Pos = Tuple[int, int]

# 方向枚举
# -------------------------
class Dir(Enum):
    """
    方向枚举类，提供四个基本方向的向量表示
    """
    # 北、南、东、西四个方向
    N = auto()  # 北 (-1, 0)
    S = auto()  # 南 (1, 0)
    E = auto()  # 东 (0, 1)
    W = auto()  # 西 (0, -1)
    
    @property
    def vector(self) -> Tuple[int, int]:
        """返回方向对应的(dx,dy)向量"""
        if self == Dir.N:
            return (-1, 0)
        elif self == Dir.S:
            return (1, 0)
        elif self == Dir.E:
            return (0, 1)
        elif self == Dir.W:
            return (0, -1)

# 奖励/惩罚常量
# -------------------------
# 成功运送一个物品的奖励值
REWARD_DELIVER = 100
# 发生碰撞的惩罚值
PENALTY_COLLISION = -100
# 每走一步的小惩罚(鼓励更短的路径)
STEP_PENALTY = -1

# 选项开关
# -------------------------
# 是否启用反向占用传感器（对应老师的"购物单"选项）
SENSOR_OPPOSITE = True
# 是否使用中央时钟调度（对应购物单选项）
CENTRAL_CLOCK = True

# 随机种子
# -------------------------
# 随机种子(可通过CLI覆盖)
SEED = 42
