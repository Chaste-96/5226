"""
传感器模块 - 实现额外的观测功能

此模块提供智能体的额外传感能力，特别是"反向占用"传感器，
用于检测周围8个方向的格子中是否有朝相反方向移动的智能体。
"""

from typing import Tuple, List
from .config import Dir, Pos


def neighbour_coords(pos: Pos) -> List[Pos]:
    """
    获取指定位置周围8个相邻格子的坐标
    
    参数:
        pos: 中心位置坐标
        
    返回:
        周围8个格子的坐标列表，顺序为: NW, N, NE, W, E, SW, S, SE
    """
    x, y = pos
    # 按照 NW, N, NE, W, E, SW, S, SE 的顺序返回坐标
    return [
        (x-1, y-1),  # 西北
        (x-1, y),    # 北
        (x-1, y+1),  # 东北
        (x, y-1),    # 西
        (x, y+1),    # 东
        (x+1, y-1),  # 西南
        (x+1, y),    # 南
        (x+1, y+1),  # 东南
    ]


def opp_dir_mask(env, agent_id: int) -> Tuple[bool, bool, bool, bool, bool, bool, bool, bool]:
    """
    返回长度为8的布尔元组，表示周围8个方向的格子是否有反向移动的智能体
    
    参数:
        env: 环境实例，包含智能体状态
        agent_id: 当前智能体ID
        
    返回:
        8位布尔元组，按照 NW, N, NE, W, E, SW, S, SE 的顺序，
        True 表示对应格子被"反向移动方向"的其他智能体占用
    """
    # 从环境中获取智能体状态
    agent_states = env.agent_states
    
    # 获取当前智能体的位置和方向标志
    current_pos = agent_states[agent_id]['position']
    current_dir_flag = agent_states[agent_id]['direction']  # 0: A→B, 1: B→A
    
    # 获取当前位置周围的8个格子坐标
    neighbours = neighbour_coords(current_pos)
    
    # 初始化结果元组
    mask = [False] * 8
    
    # 网格大小
    grid_size = env.grid_size
    
    # 检查每个相邻格子
    for i, (nx, ny) in enumerate(neighbours):
        # 检查坐标是否在网格范围内
        if not (0 <= nx < grid_size and 0 <= ny < grid_size):
            # 如果超出网格边界，标记为False（或者与注释中说的一致，返回False）
            mask[i] = False
            continue
        
        # 检查该位置是否有其他智能体
        for other_id, other_state in agent_states.items():
            if other_id == agent_id:
                continue  # 跳过当前智能体自身
                
            other_pos = other_state['position']
            other_dir_flag = other_state['direction']  # 0: A→B, 1: B→A
            
            # 如果该位置有其他智能体，并且方向与当前智能体相反
            if other_pos == (nx, ny) and other_dir_flag != current_dir_flag:
                mask[i] = True
                break
    
    # 返回8位布尔元组
    return tuple(mask)


# 导出主要的函数
__all__ = ['opp_dir_mask', 'neighbour_coords']

