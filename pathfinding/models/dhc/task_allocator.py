import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple
from scipy.optimize import linear_sum_assignment
from .agent import Agent
from .task import Task, TaskStatus, TaskPriority

class TaskAllocator:
    """任务分配器类
    
    负责智能体与任务之间的匹配和分配
    支持无标签化的协作和通信机制
    """
    def __init__(self, 
                 num_agents: int,
                 map_size: int,
                 communication_range: float = 10.0):
        """初始化任务分配器
        
        Args:
            num_agents: 智能体数量
            map_size: 地图大小
            communication_range: 通信范围
        """
        self.num_agents = num_agents
        self.map_size = map_size
        self.communication_range = communication_range
        
        # 智能体和任务管理
        self.agents: Dict[str, Agent] = {}
        self.tasks: Dict[str, Task] = {}
        self.charging_stations: List[np.ndarray] = []
        
        # 分配历史记录
        self.assignment_history: List[Dict[str, Any]] = []
        
        # 当前时间步
        self.current_step = 0
        
        # 统计信息
        self.statistics = {
            'total_assignments': 0,
            'successful_assignments': 0,
            'failed_assignments': 0,
            'average_assignment_time': 0,
            'communication_overhead': 0
        }
        
        # 添加障碍物地图属性
        self.obstacle_map = None
        
        # 添加可用位置列表属性
        self.available_positions = []
        
        # 添加探索区域地图
        self.explored_area = np.zeros((map_size, map_size), dtype=bool)
        
    def set_obstacle_map(self, obstacle_map):
        """设置障碍物地图
        
        Args:
            obstacle_map: 障碍物地图，二维numpy数组，1表示障碍物，0表示空地
        """
        self.obstacle_map = obstacle_map
        # 初始化可用位置列表（非障碍物位置）
        self.initialize_available_positions()
        
    def initialize_available_positions(self):
        """初始化可用位置列表（非障碍物位置）"""
        if self.obstacle_map is None:
            return
            
        self.available_positions = []
        for i in range(self.map_size):
            for j in range(self.map_size):
                if self.obstacle_map[i, j] == 0:  # 非障碍物位置
                    self.available_positions.append((i, j))
        
    def add_agent(self, agent: Agent):
        """添加智能体
        
        Args:
            agent: 要添加的智能体
        """
        self.agents[agent.id] = agent
        agent.task_allocator = self  # 设置智能体的任务分配器
        
    def add_task(self, task: Task):
        """添加任务
        
        Args:
            task: 要添加的任务
        """
        self.tasks[task.id] = task
        
    def set_charging_stations(self, stations: List[np.ndarray]):
        """设置充电站位置
        
        Args:
            stations: 充电站位置列表
        """
        self.charging_stations = stations
        
    def find_nearest_charging_station(self, position: np.ndarray) -> Tuple[np.ndarray, float]:
        """找到最近的充电站
        
        Args:
            position: 当前位置
            
        Returns:
            Tuple[np.ndarray, float]: (充电站位置, 距离)
        """
        if not self.charging_stations:
            return None, float('inf')
            
        distances = [np.linalg.norm(position - station) 
                    for station in self.charging_stations]
        min_idx = np.argmin(distances)
        return self.charging_stations[min_idx], distances[min_idx]
        
    def build_agent_features(self) -> np.ndarray:
        """构建智能体特征矩阵
        
        Returns:
            np.ndarray: 智能体特征矩阵
        """
        features = []
        for agent in self.agents.values():
            # 基础特征
            agent_features = [
                agent.current_battery / agent.max_battery,  # 电量比例
                agent.state['experience'] / 10.0,          # 经验值
                agent.state['idle_time'] / 20.0,          # 空闲时间
                agent.state['total_distance'] / 100.0      # 累计移动距离
            ]
            
            # 如果有当前任务，添加任务相关特征
            if agent.current_task:
                task = agent.current_task
                agent_features.extend([
                    task.priority.value / 4.0,            # 任务优先级
                    task.features['urgency'],             # 任务紧急程度
                    task.features['complexity']           # 任务复杂度
                ])
            else:
                agent_features.extend([0.0, 0.0, 0.0])
                
            features.append(agent_features)
            
        return np.array(features)
        
    def build_task_features(self) -> np.ndarray:
        """构建任务特征矩阵
        
        Returns:
            np.ndarray: 任务特征矩阵
        """
        features = []
        for task in self.tasks.values():
            if task.status != TaskStatus.PENDING:
                continue
                
            # 任务特征
            task_features = [
                task.priority.value / 4.0,           # 优先级
                task.features['urgency'],            # 紧急程度
                task.features['complexity'],         # 复杂度
                task.features['reward'] / 50.0,      # 奖励值
                task.estimated_energy / 50.0         # 预计能量消耗
            ]
            
            features.append(task_features)
            
        return np.array(features) if features else np.zeros((0, 5))
        
    def compute_compatibility_matrix(self, 
                                   agent_features=None,
                                   task_features=None,
                                   model=None) -> np.ndarray:
        """计算智能体和任务之间的兼容性矩阵
        
        Args:
            agent_features: 智能体特征矩阵
            task_features: 任务特征矩阵
            model: 可选的深度学习模型
            
        Returns:
            np.ndarray: 兼容性矩阵
        """
        if agent_features is None:
            agent_features = self.build_agent_features()
            
        if task_features is None:
            task_features = self.build_task_features()
            
        # 使用基于规则的兼容性计算
        compatibility = np.zeros((len(self.agents), len(self.tasks)))
        
        for i, agent in enumerate(self.agents.values()):
            for j, task in enumerate(self.tasks.values()):
                if task.status != TaskStatus.PENDING:
                    continue
                    
                # 检查电量是否足够
                if agent.current_battery < task.estimated_energy:
                    compatibility[i, j] = -float('inf')
                    continue
                    
                # 计算距离得分（使用曼哈顿距离）
                distance = abs(agent.pos[0] - task.start_pos[0]) + abs(agent.pos[1] - task.start_pos[1])
                distance_score = 1.0 - min(1.0, distance / (self.map_size * 0.5))
                
                # 计算电量得分（考虑任务能耗）
                battery_score = (agent.current_battery - task.estimated_energy) / agent.max_battery
                
                # 计算经验得分（考虑任务复杂度）
                experience_score = min(1.0, agent.state['experience'] / 10.0)
                complexity_factor = 1.0 - task.features['complexity'] * 0.5
                
                # 计算空闲时间得分（考虑任务紧急程度）
                idle_score = min(1.0, agent.state['idle_time'] / 20.0)
                urgency_factor = 1.0 + task.features['urgency'] * 0.5
                
                # 计算任务优先级得分
                priority_score = task.priority.value / 4.0
                
                # 计算任务紧急程度得分
                urgency_score = task.features.get('urgency', 0.5)
                
                # 计算任务奖励得分
                reward_score = task.features['reward'] / 50.0
                
                # 计算综合得分（使用加权平均）
                weights = {
                    'distance': 2.0,      # 距离权重
                    'battery': 2.5,       # 电量权重
                    'experience': 1.0,    # 经验权重
                    'idle': 1.5,          # 空闲时间权重
                    'priority': 2.0,      # 优先级权重
                    'urgency': 2.5,       # 紧急程度权重
                    'reward': 1.5         # 奖励权重
                }
                
                total_weight = sum(weights.values())
                
                compatibility[i, j] = (
                    distance_score * weights['distance'] +
                    battery_score * weights['battery'] +
                    experience_score * complexity_factor * weights['experience'] +
                    idle_score * urgency_factor * weights['idle'] +
                    priority_score * weights['priority'] +
                    urgency_score * weights['urgency'] +
                    reward_score * weights['reward']
                ) / total_weight
                
                # 添加随机扰动，避免完全相同的得分
                compatibility[i, j] += np.random.uniform(-0.01, 0.01)
                
        return compatibility
        
    def hungarian_assignment(self, compatibility: np.ndarray) -> List[Tuple[int, int]]:
        """使用匈牙利算法进行任务分配
        
        Args:
            compatibility: 兼容性矩阵
            
        Returns:
            List[Tuple[int, int]]: 分配结果列表 (agent_idx, task_idx)
        """
        # 将兼容性矩阵转换为成本矩阵 (取负值)
        cost_matrix = -compatibility
        
        # 使用匈牙利算法求解
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # 返回分配结果
        return list(zip(row_ind, col_ind))
        
    def decentralized_assignment(self, compatibility: np.ndarray) -> List[Tuple[int, int]]:
        """去中心化分配算法（无标签化协作）
        
        Args:
            compatibility: 兼容性矩阵
            
        Returns:
            List[Tuple[int, int]]: 分配结果列表 (agent_idx, task_idx)
        """
        assignments = []
        available_agents = set(range(len(self.agents)))
        available_tasks = set(range(len(self.tasks)))
        
        while available_agents and available_tasks:
            # 创建当前可用智能体的列表副本以避免在迭代过程中修改集合
            current_available_agents = list(available_agents)
            
            # 每个智能体选择最适合的任务
            for agent_idx in current_available_agents:
                if not available_tasks:
                    break
                    
                # 获取当前智能体对所有可用任务的评分
                task_scores = []
                for task_idx in available_tasks:
                    if compatibility[agent_idx, task_idx] > -float('inf'):
                        task_scores.append((task_idx, compatibility[agent_idx, task_idx]))
                        
                if not task_scores:
                    continue
                    
                # 选择得分最高的任务
                best_task_idx = max(task_scores, key=lambda x: x[1])[0]
                
                # 检查是否有其他智能体也选择了这个任务
                conflict = False
                for other_agent_idx in available_agents:
                    if other_agent_idx != agent_idx:
                        if compatibility[other_agent_idx, best_task_idx] > compatibility[agent_idx, best_task_idx]:
                            conflict = True
                            break
                            
                if not conflict:
                    assignments.append((agent_idx, best_task_idx))
                    available_agents.remove(agent_idx)
                    available_tasks.remove(best_task_idx)
                    
        return assignments
        
    def execute_assignments(self, assignments: List[Tuple[int, int]]):
        """执行任务分配
        
        Args:
            assignments: 任务分配列表，每一项是(智能体索引, 任务索引)
        """
        for agent_idx, task_idx in assignments:
            if agent_idx in self.agents and task_idx in self.tasks:
                agent = self.agents[agent_idx]
                task = self.tasks[task_idx]
                
                # 检查任务是否可分配
                if task.status == TaskStatus.PENDING:
                    # 分配任务
                    agent.assign_task(task)
                    task.assign_to_agent(agent.id)
                    
                    # 记录分配历史
                    self.assignment_history.append({
                        'step': self.current_step,
                        'agent_id': agent.id,
                        'task_id': task.id,
                        'distance': np.linalg.norm(agent.pos - task.start_pos)
                    })
                    
                    # 更新统计信息
                    self.statistics['total_assignments'] += 1
                    
                    # 记录信息
                    print(f"Step {self.current_step}: Task {task.id} assigned to Agent {agent.id}")
            
    def get_available_tasks(self, agent: Agent) -> List[Task]:
        """获取可用任务列表
        
        Args:
            agent: 智能体
            
        Returns:
            List[Task]: 可用任务列表
        """
        # 返回所有待分配的任务
        return [task for task in self.tasks.values() if task.status == TaskStatus.PENDING]
    
    def assign_task(self, agent: Agent, task: Task):
        """分配任务给智能体
        
        Args:
            agent: 智能体
            task: 任务
        """
        # 检查任务是否可分配
        if task.status == TaskStatus.PENDING:
            # 分配任务
            agent.assign_task(task)
            task.assign_to_agent(agent.id)
            
            # 记录分配历史
            self.assignment_history.append({
                'step': self.current_step,
                'agent_id': agent.id,
                'task_id': task.id,
                'distance': np.linalg.norm(agent.pos - task.start_pos)
            })
            
            # 更新统计信息
            self.statistics['total_assignments'] += 1
            
            # 记录信息
            print(f"Step {self.current_step}: 任务{task.id} (优先级: {task.priority.name}) 分配给智能体{agent.id}")
            
    def generate_communication_mask(self):
        """生成智能体之间的通信掩码
        
        Returns:
            np.ndarray: 通信掩码，值为1表示可以通信，0表示不能通信
        """
        num_agents = len(self.agents)
        comm_mask = np.zeros((num_agents, num_agents))
        
        # 获取所有智能体
        agents_list = list(self.agents.values())
        
        # 检查每对智能体之间的通信能力
        for i in range(num_agents):
            for j in range(num_agents):
                if i == j:
                    # 智能体可以与自身通信
                    comm_mask[i, j] = 1
                else:
                    # 检查距离是否在通信范围内
                    agent_i = agents_list[i]
                    agent_j = agents_list[j]
                    
                    distance = np.linalg.norm(agent_i.pos - agent_j.pos)
                    if distance <= self.communication_range:
                        comm_mask[i, j] = 1
        
        return comm_mask
        
    def update(self):
        """更新任务分配器状态并计算奖励"""
        self.current_step += 1
        
        # 检查并处理超时任务
        self.check_and_handle_task_timeouts()
        
        # 更新统计信息
        if self.statistics['total_assignments'] > 0:
            self.statistics['average_assignment_time'] = (
                self.statistics['average_assignment_time'] * 0.9 +
                self.current_step * 0.1
            )
        
        # 更新智能体状态并计算奖励
        rewards = []
        
        # 检测智能体之间的碰撞
        collision_pairs = set()
        for i, agent1 in self.agents.items():
            for j, agent2 in self.agents.items():
                if i != j and np.array_equal(agent1.pos, agent2.pos):
                    collision_pairs.add(frozenset([i, j]))
        
        for agent_id, agent in self.agents.items():
            reward = 0.0
            
            if agent.current_task:
                task = agent.current_task
                current_pos = np.array(agent.pos)
                
                # 1. 任务完成奖励（最高优先级）
                if np.array_equal(current_pos, task.goal_pos):
                    reward += 50.0 * task.priority.value
                    continue
                
                # 2. 路径规划奖励
                if not hasattr(task, 'reached_start'):
                    # 2.1 到达起点阶段
                    target_pos = task.start_pos
                    if np.array_equal(current_pos, target_pos):
                        task.reached_start = True
                        task_reward = 10.0 * task.priority.value  # 到达起点奖励
                        reward += task_reward
                        # 添加到达起点提示信息
                        print(f"智能体{agent.id}到达任务{task.id}起点，额外奖励：{task_reward:.1f}")
                    else:
                        # 使用A*算法获取到起点的路径
                        path = self._get_astar_path(current_pos, target_pos)
                        if path and len(path) > 1:
                            next_pos = path[1]
                            # 计算实际移动方向与规划路径方向的一致性
                            if hasattr(task, 'last_pos') and not np.array_equal(current_pos, task.last_pos):
                                actual_movement = current_pos - task.last_pos
                                planned_movement = next_pos - current_pos
                                
                                # 归一化向量
                                if np.linalg.norm(actual_movement) > 0 and np.linalg.norm(planned_movement) > 0:
                                    actual_movement = actual_movement / np.linalg.norm(actual_movement)
                                    planned_movement = planned_movement / np.linalg.norm(planned_movement)
                                    
                                    # 计算方向一致性
                                    direction_alignment = np.dot(actual_movement, planned_movement)
                                    # 即使方向不完全一致，只要大致正确也给予奖励
                                    if direction_alignment > -0.5:  # 允许更大的角度偏差
                                        reward += (direction_alignment + 0.5) * 0.5
                            
                            # 路径长度奖励
                            remaining_steps = len(path) - 1
                            if not hasattr(task, 'last_start_steps'):
                                task.last_start_steps = remaining_steps
                            elif remaining_steps < task.last_start_steps:
                                reward += (task.last_start_steps - remaining_steps) * 1.0
                            task.last_start_steps = remaining_steps
                else:
                    # 2.2 到达终点阶段
                    target_pos = task.goal_pos
                    if np.array_equal(current_pos, target_pos):
                        task_reward = 50.0 * task.priority.value  # 完成任务奖励
                        reward += task_reward
                        # 添加完成任务提示信息
                        print(f"--------智能体{agent.id}完成任务{task.id}，额外奖励：{task_reward:.1f}--------")
                        agent.complete_task()
                    else:
                        # 使用A*算法获取到终点的路径
                        path = self._get_astar_path(current_pos, target_pos)
                        if path and len(path) > 1:
                            next_pos = path[1]
                            # 计算实际移动方向与规划路径方向的一致性
                            if hasattr(task, 'last_pos') and not np.array_equal(current_pos, task.last_pos):
                                actual_movement = current_pos - task.last_pos
                                planned_movement = next_pos - current_pos
                                
                                # 归一化向量
                                if np.linalg.norm(actual_movement) > 0 and np.linalg.norm(planned_movement) > 0:
                                    actual_movement = actual_movement / np.linalg.norm(actual_movement)
                                    planned_movement = planned_movement / np.linalg.norm(planned_movement)
                                    
                                    # 计算方向一致性
                                    direction_alignment = np.dot(actual_movement, planned_movement)
                                    # 即使方向不完全一致，只要大致正确也给予奖励
                                    if direction_alignment > -0.5:  # 允许更大的角度偏差
                                        reward += (direction_alignment + 0.5) * 1.0
                            
                            # 路径长度奖励
                            remaining_steps = len(path) - 1
                            if not hasattr(task, 'last_goal_steps'):
                                task.last_goal_steps = remaining_steps
                            elif remaining_steps < task.last_goal_steps:
                                reward += (task.last_goal_steps - remaining_steps) * 1.0
                            task.last_goal_steps = remaining_steps
                
                # 3. 安全移动奖励
                if self.is_valid_position(current_pos):
                    reward += 0.5  # 奖励安全移动
                
                # 4. 基础行为奖励
                if hasattr(agent, 'last_position'):
                    last_pos = agent.last_position
                    if not np.array_equal(current_pos, last_pos):
                        reward += 0  
                    else:
                        reward -= 0.05  # 轻微惩罚原地停留
                
                # 更新位置记录
                agent.last_position = current_pos.copy()
                if not hasattr(task, 'last_pos'):
                    task.last_pos = current_pos.copy()
                else:
                    task.last_pos = current_pos.copy()
            else:
                # 5. 未分配任务惩罚
                reward -= 0  # 轻微惩罚未分配任务
                
                # 尝试分配任务
                available_tasks = self.get_available_tasks(agent)
                if available_tasks:
                    best_task = None
                    best_score = float('-inf')
                    
                    for task in available_tasks:
                        distance = np.linalg.norm(agent.pos - task.start_pos)
                        score = task.priority.value - (distance / self.map_size)
                        
                        if score > best_score:
                            best_score = score
                            best_task = task
                    
                    if best_task and best_score > 0:
                        self.assign_task(agent, best_task)
            
            # 检查是否发生与其他智能体的碰撞 - 降低惩罚
            for pair in collision_pairs:
                if agent_id in pair:
                    # 碰撞惩罚 - 降低碰撞惩罚
                    reward -= 3.0  # 原来是5.0，降低惩罚
                    break
            
            # 检查是否尝试进入障碍物区域 - 降低惩罚
            if hasattr(agent, 'attempted_invalid_move') and agent.attempted_invalid_move:
                # 尝试进入障碍物或地图边界的惩罚 - 使用固定值而非累加
                reward -= 2.0  # 原来是2.0，改为更小的固定惩罚
                
                # 记录碰撞但不累加惩罚
                if hasattr(agent, 'consecutive_obstacle_collisions'):
                    agent.consecutive_obstacle_collisions += 1
                    # 限制最大连续碰撞惩罚，避免reward崩溃
                    max_collision_penalty = 1.5
                    collision_penalty = min(max_collision_penalty, 0.2 * agent.consecutive_obstacle_collisions)
                    reward -= collision_penalty  # 使用衰减的碰撞惩罚
                else:
                    agent.consecutive_obstacle_collisions = 1
                
                # 标记碰撞，但重置标志，避免重复惩罚
                agent.attempted_invalid_move = False
            else:
                # 重置连续碰撞计数
                if hasattr(agent, 'consecutive_obstacle_collisions'):
                    agent.consecutive_obstacle_collisions = 0
            
            # 4. 充电行为奖励 - 增强充电奖励
            for station in self.charging_stations:
                if np.array_equal(agent.pos, station):
                    if agent.current_battery < agent.max_battery:
                        # 低电量时充电的额外奖励
                        if agent.current_battery < 0.3:
                            reward += 2.0  # 原来是1.0，提高低电量充电奖励
                        else:
                            reward += 1.0  # 原来是0.5，提高普通充电奖励
                        
                        # 充电
                        agent.update_battery(5.0)  # 充电5%
                        
                        # 添加标记
                        agent.state['charging'] = True
                        break
            else:
                if hasattr(agent.state, 'charging') and agent.state['charging']:
                    agent.state['charging'] = False
            
            rewards.append(reward)
            
        # 为每个空闲的智能体尝试分配任务
        self.try_assign_unassigned_tasks()
            
        return rewards
        
    def try_assign_unassigned_tasks(self):
        """尝试为未分配任务的智能体分配任务"""
        # 获取所有待处理任务
        pending_tasks = [task for task in self.tasks.values() if task.status == TaskStatus.PENDING]
        if not pending_tasks:
            return
            
        # 获取没有任务的智能体
        idle_agents = [agent for agent in self.agents.values() if agent.current_task is None]
        if not idle_agents:
            return
            
        # 计算兼容性矩阵
        compatibility = np.zeros((len(idle_agents), len(pending_tasks)))
        
        # 获取每个智能体的任务完成历史
        agent_task_counts = {}
        for agent in idle_agents:
            agent_id = agent.id
            # 初始化计数器
            if not hasattr(agent, 'completed_tasks_by_priority'):
                agent.completed_tasks_by_priority = {1: 0, 2: 0, 3: 0, 4: 0}
            
            # 记录完成的任务数
            agent_task_counts[agent_id] = sum(agent.completed_tasks_by_priority.values())
        
        for i, agent in enumerate(idle_agents):
            for j, task in enumerate(pending_tasks):
                # 计算距离因子 (距离越近越好)
                distance = np.linalg.norm(agent.pos - task.start_pos)
                distance_factor = 1.0 - min(1.0, distance / self.map_size)
                
                # 计算电量因子 (电量越多越好)
                battery_factor = agent.current_battery / agent.max_battery
                
                # 计算任务优先级因子
                priority_factor = task.priority.value / 4.0
                
                # 计算任务分配均衡因子 (优先分配给完成该优先级任务较少的智能体)
                priority_value = task.priority.value
                balance_factor = 1.0
                
                # 如果智能体之前完成过这个优先级的任务，降低分数
                if hasattr(agent, 'completed_tasks_by_priority') and agent.completed_tasks_by_priority.get(priority_value, 0) > 0:
                    # 越是高优先级任务，越需要平衡分配
                    priority_multiplier = priority_value * 0.25  # 高优先级有更大的平衡系数
                    # 完成任务数越多，平衡因子越低
                    balance_factor = 1.0 - min(0.5, agent.completed_tasks_by_priority.get(priority_value, 0) * 0.1 * priority_multiplier)
                
                # 计算综合得分
                score = (distance_factor * 2.5 + 
                         battery_factor * 2.0 + 
                         priority_factor * 1.5 + 
                         balance_factor * 2.0)
                
                compatibility[i, j] = score
        
        # 贪心分配 - 每次选择最高得分的(智能体,任务)对
        while len(idle_agents) > 0 and len(pending_tasks) > 0:
            # 找到最大得分
            flat_idx = np.argmax(compatibility)
            i, j = np.unravel_index(flat_idx, compatibility.shape)
            
            # 如果得分太低，不分配
            if compatibility[i, j] <= 0:
                break
                
            # 分配任务
            agent = idle_agents[i]
            task = pending_tasks[j]
            
            self.assign_task(agent, task)
            
            # 更新剩余智能体和任务
            compatibility[i, :] = -np.inf  # 该智能体不再可用
            compatibility[:, j] = -np.inf  # 该任务不再可用
            
            # 从列表中移除
            idle_agents.pop(i)
            pending_tasks.pop(j)
            
            # 更新兼容性矩阵形状
            compatibility = np.delete(compatibility, i, axis=0)
            compatibility = np.delete(compatibility, j, axis=1)
        
    def _discovered_new_area(self, agent):
        """检查是否发现新区域
        
        Args:
            agent: 智能体对象
            
        Returns:
            bool: 是否发现新区域
        """
        x, y = agent.pos
        obs_radius = 4
        discovered = False
        
        for dx in range(-obs_radius, obs_radius + 1):
            for dy in range(-obs_radius, obs_radius + 1):
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < self.map_size and 0 <= new_y < self.map_size:
                    if not self.explored_area[new_x, new_y]:
                        self.explored_area[new_x, new_y] = True
                        discovered = True
        
        return discovered
        
    def _helped_other_agent(self, agent):
        """检查是否帮助了其他智能体
        
        Args:
            agent: 智能体对象
            
        Returns:
            bool: 是否帮助了其他智能体
        """
        # 检查是否在充电站附近有其他低电量智能体
        if any(np.array_equal(agent.pos, station) for station in self.charging_stations):
            for other_agent in self.agents.values():
                if other_agent.id != agent.id and other_agent.current_battery < 20:
                    return True
        
        # 检查是否帮助其他智能体完成任务
        if agent.current_task and agent.current_task.status == TaskStatus.COMPLETED:
            for other_agent in self.agents.values():
                if other_agent.id != agent.id and other_agent.current_task == agent.current_task:
                    return True
        
        return False
        
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息
        
        Returns:
            Dict[str, Any]: 统计信息字典
        """
        return self.statistics
        
    def reset(self):
        """重置任务分配器"""
        self.tasks.clear()
        self.assignment_history.clear()
        self.current_step = 0
        
        # 重置统计信息
        self.statistics = {
            'total_assignments': 0,
            'successful_assignments': 0,
            'failed_assignments': 0,
            'average_assignment_time': 0,
            'communication_overhead': 0
        }
        
    def to_dict(self) -> Dict[str, Any]:
        """将任务分配器转换为字典格式
        
        Returns:
            Dict[str, Any]: 任务分配器字典
        """
        return {
            'num_agents': self.num_agents,
            'map_size': self.map_size,
            'communication_range': self.communication_range,
            'charging_stations': [station.tolist() for station in self.charging_stations],
            'tasks': [task.to_dict() for task in self.tasks.values()],
            'assignment_history': self.assignment_history,
            'current_step': self.current_step,
            'statistics': self.statistics
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskAllocator':
        """从字典创建任务分配器
        
        Args:
            data: 任务分配器数据字典
            
        Returns:
            TaskAllocator: 新任务分配器实例
        """
        allocator = cls(
            num_agents=data['num_agents'],
            map_size=data['map_size'],
            communication_range=data['communication_range']
        )
        
        # 恢复充电站
        allocator.charging_stations = [
            np.array(station) for station in data['charging_stations']
        ]
        
        # 恢复任务列表
        allocator.tasks = {
            task['id']: Task.from_dict(task)
            for task in data['tasks']
        }
        
        # 恢复其他属性
        allocator.assignment_history = data['assignment_history']
        allocator.current_step = data['current_step']
        allocator.statistics = data['statistics']
        
        return allocator
        
    def is_done(self):
        """检查是否所有任务都已完成
        
        Returns:
            bool: 如果所有任务都已完成则返回True
        """
        # 检查是否所有任务都已完成或失败
        for task in self.tasks.values():
            if task.status not in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                return False
                
        return True

    def handle_agent_failure(self, agent_id: str, failure_reason: str):
        """处理智能体故障
        
        Args:
            agent_id: 故障智能体ID
            failure_reason: 故障原因
        """
        if agent_id not in self.agents:
            return
        
        agent = self.agents[agent_id]
        
        # 如果智能体有正在执行的任务，需要重新分配
        if agent.current_task:
            task = agent.current_task
            task.status = TaskStatus.PENDING
            task.assigned_agent_id = None
            task.assigned_time = None
            
            # 记录故障信息
            self.statistics['failed_assignments'] += 1
            
            # 尝试重新分配任务
            self.redistribute_task(task)
        
        # 从活跃智能体列表中移除
        del self.agents[agent_id]

    def redistribute_task(self, task: Task):
        """重新分配任务
        
        Args:
            task: 需要重新分配的任务
        """
        # 更新任务状态
        task.status = TaskStatus.PENDING
        task.assigned_agent_id = None
        task.assigned_time = None
        
        # 计算新的兼容性矩阵
        compatibility = self.compute_compatibility_matrix()
        
        # 获取任务索引
        task_idx = list(self.tasks.keys()).index(task.id)
        
        # 找到最适合的智能体
        best_agent_idx = np.argmax(compatibility[:, task_idx])
        best_score = compatibility[best_agent_idx, task_idx]
        
        if best_score > 0:  # 确保有合适的智能体
            agent_id = list(self.agents.keys())[best_agent_idx]
            agent = self.agents[agent_id]
            
            # 检查智能体是否已经有任务
            if agent.current_task:
                # 如果新任务优先级更高，则替换当前任务
                if task.priority.value > agent.current_task.priority.value:
                    old_task = agent.current_task
                    self.redistribute_task(old_task)  # 递归处理被替换的任务
                    self.assign_task_to_agent(task, agent_id)
            else:
                self.assign_task_to_agent(task, agent_id)

    def assign_task_to_agent(self, task: Task, agent_id: str):
        """将任务分配给智能体
        
        Args:
            task: 要分配的任务
            agent_id: 智能体ID
        """
        if agent_id not in self.agents:
            return
        
        agent = self.agents[agent_id]
        
        # 检查智能体是否能够执行任务
        if agent.current_battery < task.estimated_energy:
            return
        
        # 分配任务
        task.assigned_agent_id = agent_id
        task.status = TaskStatus.ASSIGNED
        task.assigned_time = self.current_step
        
        agent.current_task = task
        
        # 更新统计信息
        self.statistics['successful_assignments'] += 1

    def check_and_handle_task_timeouts(self):
        """检查并处理超时任务"""
        current_time = self.current_step
        
        for task in self.tasks.values():
            if task.status == TaskStatus.ASSIGNED and task.deadline is not None:
                if current_time > task.deadline:
                    # 任务超时，重新分配
                    self.redistribute_task(task)
                    task.fail("任务超时", current_time) 

    def _get_astar_path(self, start_pos: np.ndarray, goal_pos: np.ndarray) -> List[np.ndarray]:
        """使用A*算法获取从起点到终点的路径
        
        Args:
            start_pos: 起点位置
            goal_pos: 终点位置
            
        Returns:
            List[np.ndarray]: 路径点列表，如果找不到路径则返回空列表
        """
        # 检查是否超出地图边界
        if (start_pos < 0).any() or start_pos[0] >= self.map_size or start_pos[1] >= self.map_size:
            return []
        if (goal_pos < 0).any() or goal_pos[0] >= self.map_size or goal_pos[1] >= self.map_size:
            return []
        
        # 检查起点和终点是否是障碍物
        if self.obstacle_map[start_pos[0], start_pos[1]] == 1 or self.obstacle_map[goal_pos[0], goal_pos[1]] == 1:
            return []
        
        # 初始化开放列表和关闭列表
        open_list = []
        closed_list = set()
        
        # 初始化起点
        start_node = {
            'pos': tuple(start_pos),
            'g_cost': 0,  # 从起点到当前节点的实际代价
            'h_cost': np.linalg.norm(goal_pos - start_pos),  # 从当前节点到终点的估计代价
            'parent': None
        }
        start_node['f_cost'] = start_node['g_cost'] + start_node['h_cost']
        
        # 将起点加入开放列表
        open_list.append(start_node)
        
        # 定义四个方向的移动
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 右、下、左、上
        
        while open_list:
            # 找到f_cost最小的节点
            current_node = min(open_list, key=lambda x: x['f_cost'])
            
            # 如果到达终点，构建路径并返回
            if np.array_equal(current_node['pos'], tuple(goal_pos)):
                path = []
                while current_node:
                    path.append(np.array(current_node['pos']))
                    current_node = current_node['parent']
                return path[::-1]  # 反转路径，从起点到终点
            
            # 将当前节点从开放列表移到关闭列表
            open_list.remove(current_node)
            closed_list.add(current_node['pos'])
            
            # 检查相邻节点
            for dx, dy in directions:
                neighbor_pos = (current_node['pos'][0] + dx, current_node['pos'][1] + dy)
                
                # 检查位置是否有效（考虑障碍物）
                if not self.is_valid_position(np.array(neighbor_pos)):
                    continue
                    
                # 如果节点在关闭列表中，跳过
                if neighbor_pos in closed_list:
                    continue
                    
                # 计算新的g_cost
                new_g_cost = current_node['g_cost'] + 1
                
                # 检查节点是否已在开放列表中
                neighbor_node = next((node for node in open_list if node['pos'] == neighbor_pos), None)
                
                if neighbor_node is None:
                    # 创建新节点
                    neighbor_node = {
                        'pos': neighbor_pos,
                        'g_cost': new_g_cost,
                        'h_cost': np.linalg.norm(np.array(goal_pos) - np.array(neighbor_pos)),
                        'parent': current_node
                    }
                    neighbor_node['f_cost'] = neighbor_node['g_cost'] + neighbor_node['h_cost']
                    open_list.append(neighbor_node)
                elif new_g_cost < neighbor_node['g_cost']:
                    # 更新已有节点
                    neighbor_node['g_cost'] = new_g_cost
                    neighbor_node['f_cost'] = new_g_cost + neighbor_node['h_cost']
                    neighbor_node['parent'] = current_node
        
        # 如果开放列表为空且未找到路径，返回空列表
        return []

    def is_valid_position(self, position: np.ndarray, ignore_obstacles: bool = False) -> bool:
        """检查位置是否有效
        
        Args:
            position: 要检查的位置
            ignore_obstacles: 是否忽略障碍物检查
            
        Returns:
            bool: 如果位置有效则返回True
        """
        # 检查是否超出地图边界
        if (position < 0).any() or position[0] >= self.map_size or position[1] >= self.map_size:
            return False
        
        # 检查是否是障碍物（除非指定忽略）
        if not ignore_obstacles and self.obstacle_map[position[0], position[1]] == 1:
            return False
        
        # 检查是否与其他智能体位置重叠
        for agent in self.agents.values():
            if np.array_equal(agent.pos, position):
                return False
            
        return True 