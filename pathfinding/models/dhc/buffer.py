import numpy as np
import torch
from collections import deque
from typing import List, Dict, Tuple, Any, Optional

class ExperienceBuffer:
    """经验回放缓冲区
    
    为无标签化任务分配系统提供经验储存和采样功能
    """
    def __init__(self, 
                capacity: int = 10000, 
                batch_size: int = 64):
        """初始化经验回放缓冲区
        
        Args:
            capacity: 缓冲区容量
            batch_size: 批次大小
        """
        self.capacity = capacity
        self.batch_size = batch_size
        
        # 存储容器
        self.observations = []  # [B, N, 6, 9, 9]
        self.actions = []      # [B, N]
        self.rewards = []      # [B, N]
        self.next_observations = []  # [B, N, 6, 9, 9]
        self.dones = []        # [B]
        self.hidden_states = []  # [B*N, latent_dim]
        self.comm_masks = []   # [B, N, N]
        self.agent_states = []  # [B, N, 2]
        self.task_features = []  # [B, N, 4]
        
        # 指针和大小
        self.position = 0
        self.size = 0

    def add(self, 
           observation: np.ndarray, 
           action: np.ndarray,
           reward: np.ndarray,
           next_observation: np.ndarray,
           done: np.ndarray,
           hidden_state: np.ndarray = None,
           comm_mask: np.ndarray = None,
           agent_states: np.ndarray = None,
           task_features: np.ndarray = None):
        """添加经验到缓冲区
        
        Args:
            observation: 观察 [B, N, 6, 9, 9]
            action: 动作 [B, N]
            reward: 奖励 [B, N]
            next_observation: 下一个观察 [B, N, 6, 9, 9]
            done: 是否完成 [B]
            hidden_state: 隐藏状态 [B*N, latent_dim]
            comm_mask: 通信掩码 [B, N, N]
            agent_states: 智能体状态 [B, N, 2]
            task_features: 任务特征 [B, N, 4]
        """
        if len(self.observations) < self.capacity:
            self.observations.append(observation)
            self.actions.append(action)
            self.rewards.append(reward)
            self.next_observations.append(next_observation)
            self.dones.append(done)
            if hidden_state is not None:
                self.hidden_states.append(hidden_state)
            if comm_mask is not None:
                self.comm_masks.append(comm_mask)
            if agent_states is not None:
                self.agent_states.append(agent_states)
            if task_features is not None:
                self.task_features.append(task_features)
        else:
            self.observations[self.position] = observation
            self.actions[self.position] = action
            self.rewards[self.position] = reward
            self.next_observations[self.position] = next_observation
            self.dones[self.position] = done
            if hidden_state is not None:
                self.hidden_states[self.position] = hidden_state
            if comm_mask is not None:
                self.comm_masks[self.position] = comm_mask
            if agent_states is not None:
                self.agent_states[self.position] = agent_states
            if task_features is not None:
                self.task_features[self.position] = task_features
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int = None) -> Dict[str, np.ndarray]:
        """从缓冲区采样经验批次
        Args:
            batch_size: 批次大小，如果为None则使用默认值
        Returns:
            Dict[str, np.ndarray]: 包含采样数据的字典
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        indices = np.random.randint(0, self.size, size=batch_size)
        
        batch = {
            'observations': np.array([self.observations[idx] for idx in indices]),
            'actions': np.array([self.actions[idx] for idx in indices]),
            'rewards': np.array([self.rewards[idx] for idx in indices]),
            'next_observations': np.array([self.next_observations[idx] for idx in indices]),
            'dones': np.array([self.dones[idx] for idx in indices])
        }
        
        if self.hidden_states:
            batch['hidden_states'] = np.array([self.hidden_states[idx] for idx in indices])
        if self.comm_masks:
            batch['comm_masks'] = np.array([self.comm_masks[idx] for idx in indices])
        if self.agent_states:
            batch['agent_states'] = np.array([self.agent_states[idx] for idx in indices])
        if self.task_features:
            batch['task_features'] = np.array([self.task_features[idx] for idx in indices])
            
        return batch
    
    def get_size(self) -> int:
        """获取当前缓冲区大小
        
        Returns:
            int: 当前缓冲区中的经验数量
        """
        return self.size 