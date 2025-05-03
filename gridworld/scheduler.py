"""
调度器模块 - 实现智能体的中央时钟调度

此模块提供中央时钟（CentralClock）调度方式，以轮询方式调度智能体。
使用中央时钟可以让智能体按照固定顺序执行，有助于减少碰撞。
"""

import itertools
import random
from typing import List, Optional


class CentralClock:
    """
    中央时钟调度器

    按照固定的轮询顺序（Round-Robin）调度智能体。
    可以通过shuffle方法重置调度顺序。

    用法:
        clock = CentralClock(agent_ids=[0,1,2,3])
        for next_id in clock:
            yield next_id      # 0,1,2,3,0,1,...
    """

    def __init__(self, agent_ids: List[int]):
        """
        初始化中央时钟调度器
        
        参数:
            agent_ids: 需要调度的智能体ID列表
        """
        self.agent_ids = agent_ids.copy()
        self._cycle = itertools.cycle(self.agent_ids)
    
    def __iter__(self) -> 'CentralClock':
        """
        实现迭代器协议
        
        返回:
            调度器自身
        """
        return self
    
    def __next__(self) -> int:
        """
        获取下一个要执行的智能体ID
        
        返回:
            下一个智能体ID
        """
        return next(self._cycle)
    
    def shuffle(self, new_order: Optional[List[int]] = None) -> None:
        """
        重新设置调度顺序
        
        参数:
            new_order: 新的智能体ID顺序，如果为None则随机打乱
        """
        if new_order is not None:
            # 验证新顺序中包含所有的智能体
            assert set(new_order) == set(self.agent_ids), "新顺序必须包含所有的智能体ID"
            self.agent_ids = new_order.copy()
        else:
            # 随机打乱当前顺序
            random.shuffle(self.agent_ids)
        
        # 重置迭代器循环
        self._cycle = itertools.cycle(self.agent_ids)


class RoundRobin:
    """
    Round-Robin调度器（CentralClock的别名）
    
    为了保持API命名一致性而提供
    """
    
    def __init__(self, agent_ids: List[int]):
        self.clock = CentralClock(agent_ids)
    
    def __iter__(self):
        return self.clock.__iter__()
    
    def __next__(self):
        return self.clock.__next__()
    
    def shuffle(self, new_order: Optional[List[int]] = None) -> None:
        self.clock.shuffle(new_order)
