"""
gridworld 模块 - 实现多智能体运输任务的网格世界环境

这个模块提供了一个网格世界环境，用于多智能体协调运输任务的强化学习训练。
环境中多个智能体需要在 A 点和 B 点之间运送物品，同时避免碰撞。
"""

import numpy as np
import random
from enum import IntEnum
from typing import List, Tuple, Dict, Optional, Set

# 动作枚举，北南西东四个方向
class Action(IntEnum):
    NORTH = 0  # 向上移动
    SOUTH = 1  # 向下移动
    WEST = 2   # 向左移动
    EAST = 3   # 向右移动


# 配置常量
GRID_SIZE = 5  # 5x5 网格
NUM_AGENTS = 4  # 4个智能体
COLLISION_PENALTY = -10.0  # 碰撞惩罚
DELIVERY_REWARD = 1.0  # 完成一次运送的奖励


class GridWorldEnv:
    """网格世界环境，实现多智能体的运输任务"""
    
    def __init__(self, 
                 grid_size: int = GRID_SIZE, 
                 num_agents: int = NUM_AGENTS,
                 collision_penalty: float = COLLISION_PENALTY,
                 delivery_reward: float = DELIVERY_REWARD):
        """
        初始化网格世界环境
        
        参数:
            grid_size: 网格的大小 (默认 5x5)
            num_agents: 智能体数量 (默认 4)
            collision_penalty: 碰撞惩罚值 (默认 -10.0)
            delivery_reward: 成功运送的奖励值 (默认 1.0)
        """
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.collision_penalty = collision_penalty
        self.delivery_reward = delivery_reward
        
        # 环境重置
        self.reset()
    
    def reset(self) -> Dict:
        """
        重置环境到初始状态
        
        返回:
            包含初始观察的字典
        """
        # 随机放置 A 和 B 的位置
        self.location_A = self._random_position()
        self.location_B = self._random_position()
        
        # 确保 A 和 B 不在同一位置
        while self.location_A == self.location_B:
            self.location_B = self._random_position()
        
        # 初始化智能体位置，默认都在 A 位置
        self.agent_positions = [self.location_A.copy() for _ in range(self.num_agents)]
        
        # 智能体是否携带物品 (0: 未携带, 1: 携带)
        self.carrying = [0] * self.num_agents
        
        # 智能体的方向 (0: A→B, 1: B→A)
        self.directions = [0] * self.num_agents  # 初始都是从 A 到 B
        
        # 累计步数和奖励
        self.steps = 0
        self.total_reward = 0
        self.total_collisions = 0
        self.total_deliveries = 0
        
        # 返回初始观察
        return self._get_observations()
    
    def step(self, actions: List[int]) -> Tuple[Dict, List[float], bool, Dict]:
        """
        执行环境中的一步。
        
        参数:
            actions: 每个智能体的动作列表
            
        返回:
            (observations, rewards, done, info)
        """
        assert len(actions) == self.num_agents, "动作数量必须等于智能体数量"
        
        # 增加步数计数
        self.steps += 1
        
        # 计算下一个位置
        next_positions = []
        for agent_idx, action in enumerate(actions):
            next_pos = self._get_next_position(self.agent_positions[agent_idx], action)
            next_positions.append(next_pos)
        
        # 检测碰撞
        rewards = [0.0] * self.num_agents
        collisions = self._detect_collisions(next_positions)
        
        # 更新位置和奖励
        for agent_idx, next_pos in enumerate(next_positions):
            if agent_idx in collisions:
                # 发生碰撞，不更新位置，给予惩罚
                rewards[agent_idx] += self.collision_penalty
                self.total_collisions += 1
            else:
                # 没有碰撞，更新位置
                self.agent_positions[agent_idx] = next_pos
                
                # 检查是否到达目标点
                if self._is_at_location_A(agent_idx) and self.directions[agent_idx] == 1:
                    # 从 B 到 A，获取新物品
                    self.carrying[agent_idx] = 1
                    self.directions[agent_idx] = 0  # 现在方向是 A→B
                
                elif self._is_at_location_B(agent_idx) and self.directions[agent_idx] == 0:
                    # 从 A 到 B，递送物品
                    if self.carrying[agent_idx] == 1:
                        self.carrying[agent_idx] = 0
                        rewards[agent_idx] += self.delivery_reward
                        self.total_deliveries += 1
                        self.directions[agent_idx] = 1  # 现在方向是 B→A
        
        # 更新总奖励
        self.total_reward += sum(rewards)
        
        # 获取新的观察
        observations = self._get_observations()
        
        # 检查是否结束 (这个任务是无限的，所以通常不会结束)
        done = False
        
        # 额外信息
        info = {
            'steps': self.steps,
            'total_reward': self.total_reward,
            'total_collisions': self.total_collisions,
            'total_deliveries': self.total_deliveries
        }
        
        return observations, rewards, done, info
    
    def _random_position(self) -> List[int]:
        """生成随机位置"""
        return [random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1)]
    
    def _get_next_position(self, current_pos: List[int], action: int) -> List[int]:
        """计算给定动作后的下一个位置"""
        next_pos = current_pos.copy()
        
        if action == Action.NORTH and next_pos[0] > 0:
            next_pos[0] -= 1
        elif action == Action.SOUTH and next_pos[0] < self.grid_size - 1:
            next_pos[0] += 1
        elif action == Action.WEST and next_pos[1] > 0:
            next_pos[1] -= 1
        elif action == Action.EAST and next_pos[1] < self.grid_size - 1:
            next_pos[1] += 1
        
        return next_pos
    
    def _detect_collisions(self, next_positions: List[List[int]]) -> Set[int]:
        """
        检测碰撞的智能体
        
        根据任务要求:
        1. 如果两个智能体一个从A→B，一个从B→A，且进入同一个格子，则发生对向碰撞
        2. 如果所有进入同一格子的智能体方向相同，则不算碰撞
        3. A和B位置不检测碰撞
        """
        collisions = set()
        
        # 建立位置到智能体的映射
        pos_to_agents = {}
        for agent_idx, pos in enumerate(next_positions):
            pos_tuple = tuple(pos)
            if pos_tuple not in pos_to_agents:
                pos_to_agents[pos_tuple] = []
            pos_to_agents[pos_tuple].append(agent_idx)
        
        # 检查每个有多个智能体的位置
        for pos_tuple, agents in pos_to_agents.items():
            # 如果位置是A或B，不检测碰撞
            if (pos_tuple == tuple(self.location_A) or 
                pos_tuple == tuple(self.location_B)):
                continue
            
            # 如果只有一个智能体，不会碰撞
            if len(agents) <= 1:
                continue
            
            # 检查是否有方向不同的智能体
            has_A_to_B = False
            has_B_to_A = False
            
            for agent_idx in agents:
                if self.directions[agent_idx] == 0:  # A→B
                    has_A_to_B = True
                else:  # B→A
                    has_B_to_A = True
            
            # 如果同时有A→B和B→A的智能体，则发生碰撞
            if has_A_to_B and has_B_to_A:
                collisions.update(agents)
        
        return collisions
    
    def _is_at_location_A(self, agent_idx: int) -> bool:
        """检查智能体是否在位置A"""
        return self.agent_positions[agent_idx] == self.location_A
    
    def _is_at_location_B(self, agent_idx: int) -> bool:
        """检查智能体是否在位置B"""
        return self.agent_positions[agent_idx] == self.location_B
    
    def _get_observations(self) -> Dict:
        """
        获取所有智能体的观察
        
        每个智能体可以观察:
        1. 自己的位置
        2. A和B的位置
        3. 是否携带物品
        4. 自己的方向(A→B或B→A)
        """
        observations = {}
        
        for agent_idx in range(self.num_agents):
            # 基本观察
            obs = {
                'position': self.agent_positions[agent_idx],
                'location_A': self.location_A,
                'location_B': self.location_B,
                'carrying': self.carrying[agent_idx],
                'direction': self.directions[agent_idx]
            }
            
            observations[agent_idx] = obs
        
        return observations
    
    def render(self):
        """打印环境的当前状态"""
        grid = [[' ' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        # 标记A和B位置
        a_row, a_col = self.location_A
        b_row, b_col = self.location_B
        grid[a_row][a_col] = 'A'
        grid[b_row][b_col] = 'B'
        
        # 标记智能体位置
        for i, (row, col) in enumerate(self.agent_positions):
            # 如果格子已经有A或B，在后面加上智能体编号
            if grid[row][col] in ['A', 'B']:
                grid[row][col] += str(i)
            else:
                # 标记携带状态和方向
                marker = str(i)
                if self.carrying[i]:
                    marker += '+'  # 携带物品
                if self.directions[i] == 0:
                    marker += '→'  # A→B方向
                else:
                    marker += '←'  # B→A方向
                grid[row][col] = marker
        
        # 打印网格
        print(f"Step: {self.steps}, Deliveries: {self.total_deliveries}, Collisions: {self.total_collisions}")
        print("-" * (self.grid_size * 4 + 1))
        for row in grid:
            print("| " + " | ".join(f"{cell:<3}" for cell in row) + " |")
            print("-" * (self.grid_size * 4 + 1))


# 导出主要的类和常量
__all__ = ['GridWorldEnv', 'Action', 'GRID_SIZE', 'NUM_AGENTS', 
           'COLLISION_PENALTY', 'DELIVERY_REWARD']
