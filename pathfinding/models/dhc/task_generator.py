import numpy as np
import random
from typing import List, Optional, Dict, Any, Tuple
from .task import Task, TaskPriority, TaskStatus

class TaskGenerator:
    """任务生成器类
    
    负责生成随机任务或按照指定模式生成任务，支持动态任务生成
    """
    def __init__(self, map_size=40, obstacle_map=None, seed=None):
        """初始化任务生成器
        
        Args:
            map_size: 地图大小
            obstacle_map: 障碍物地图，如果提供，将会避开障碍物生成任务
            seed: 随机种子
        """
        self.map_size = map_size
        self.obstacle_map = obstacle_map
        self.next_task_id = 0
        self.tasks = []  # 当前活跃的任务列表
        self.completed_tasks = []  # 已完成的任务列表
        self.statistics = {
            'total_generated': 0,
            'total_completed': 0,
            'total_failed': 0,
            'total_cancelled': 0,
            'average_completion_time': 0.0,
            'priority_completion_rates': {
                priority: {'total': 0, 'completed': 0}
                for priority in TaskPriority
            }
        }
        
        # 设置随机种子
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # 添加路径查找辅助
        from pathfinding.models.dhc.pathfinder import PathFinder
        self.pathfinder = PathFinder(map_size)
        if obstacle_map is not None:
            self.pathfinder.set_obstacle_map(obstacle_map)
        
    def _get_available_positions(self) -> List[Tuple[int, int]]:
        """获取可用的位置
        
        Returns:
            List[Tuple[int, int]]: 可用位置列表
        """
        if self.obstacle_map is None:
            # 如果没有障碍物地图，所有位置都可用
            return [(i, j) for i in range(self.map_size) for j in range(self.map_size)]
        else:
            # 只返回非障碍物位置
            return [(i, j) for i in range(self.map_size) for j in range(self.map_size) 
                   if self.obstacle_map[i, j] == 0]
    
    def generate_task(self, 
                    priority: Optional[TaskPriority] = None, 
                    min_distance: int = 5,
                    max_distance: Optional[int] = None,
                    required_path: bool = True,
                    zone_bias: bool = True) -> Task:
        """生成新任务
        
        Args:
            priority: 任务优先级，如果为None则随机生成
            min_distance: 起点和终点之间的最小距离
            max_distance: 起点和终点之间的最大距离，如果为None则不限制
            required_path: 是否要求起点和终点之间必须存在有效路径
            zone_bias: 是否对地图进行区域划分以优化任务分布
            
        Returns:
            Task: 生成的任务
        """
        # 如果没有指定优先级，随机生成
        if priority is None:
            # 优先级分布：高(10%)，中(30%)，低(60%)
            priority_choices = [
                TaskPriority.HIGH,
                TaskPriority.MEDIUM,
                TaskPriority.LOW
            ]
            weights = [0.1, 0.3, 0.6]
            priority = np.random.choice(priority_choices, p=weights)
        
        # 设置最大距离
        if max_distance is None:
            max_distance = self.map_size // 2
            
        if not self.available_positions:
            # 如果没有可用位置，重新获取
            self.available_positions = self._get_available_positions()
            if not self.available_positions:
                raise ValueError("没有可用位置生成任务")
                
        # 任务生成最大尝试次数
        max_attempts = 50
        attempts = 0
        
        while attempts < max_attempts:
            # 使用区域偏好生成起点和终点
            if zone_bias:
                start_pos, goal_pos = self._generate_biased_positions(min_distance, max_distance)
            else:
                # 随机选择起点
                start_pos = random.choice(self.available_positions)
                
                # 筛选在有效距离范围内的终点位置
                valid_goal_positions = []
                for pos in self.available_positions:
                    dist = abs(pos[0] - start_pos[0]) + abs(pos[1] - start_pos[1])  # 曼哈顿距离
                    if min_distance <= dist <= max_distance:
                        valid_goal_positions.append(pos)
                
                # 如果没有有效终点，重新选择起点
                if not valid_goal_positions:
                    attempts += 1
                    continue
                
                # 随机选择终点
                goal_pos = random.choice(valid_goal_positions)
            
            # 如果需要确认有效路径
            if required_path:
                if hasattr(self, 'pathfinder'):
                    path = self.pathfinder.find_path(start_pos, goal_pos)
                    if path and len(path) >= min_distance:
                        # 有效路径，创建任务
                        break
                    else:
                        # 无效路径，重试
                        attempts += 1
                        continue
            else:
                # 不需要检查路径，直接创建任务
                break
            
            attempts += 1
                
        if attempts >= max_attempts:
            # 如果超过最大尝试次数，放宽条件
            start_pos = random.choice(self.available_positions)
            goal_pos = random.choice(self.available_positions)
            while start_pos == goal_pos:
                goal_pos = random.choice(self.available_positions)
        
        # 生成任务特征
        task_features = self._generate_task_features(start_pos, goal_pos, priority)
        
        # 创建任务
        task = Task(
            id=self.next_task_id,
            start_pos=start_pos,
            goal_pos=goal_pos,
            priority=priority,
            features=task_features
        )
        
        # 更新任务ID
        self.next_task_id += 1
        
        return task
    
    def _generate_biased_positions(self, min_distance: int, max_distance: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """生成有区域偏好的起点和终点位置
        
        将地图分成4个象限，优先在不同象限之间生成任务，以便更好地分布
        
        Args:
            min_distance: 最小距离
            max_distance: 最大距离
            
        Returns:
            Tuple[Tuple[int, int], Tuple[int, int]]: 起点和终点位置
        """
        # 分区域
        half = self.map_size // 2
        quarters = [
            [(0, 0), (half, half)],  # 左上
            [(0, half), (half, self.map_size)],  # 右上
            [(half, 0), (self.map_size, half)],  # 左下
            [(half, half), (self.map_size, self.map_size)]  # 右下
        ]
        
        # 随机选择两个不同的区域
        zone_indices = random.sample(range(4), 2)
        start_zone = quarters[zone_indices[0]]
        goal_zone = quarters[zone_indices[1]]
        
        # 在各自区域内找到有效位置
        start_positions = [pos for pos in self.available_positions 
                          if start_zone[0][0] <= pos[0] < start_zone[1][0] and 
                             start_zone[0][1] <= pos[1] < start_zone[1][1]]
        
        goal_positions = [pos for pos in self.available_positions 
                         if goal_zone[0][0] <= pos[0] < goal_zone[1][0] and 
                            goal_zone[0][1] <= pos[1] < goal_zone[1][1]]
        
        # 如果某个区域没有有效位置，回退到随机选择
        if not start_positions or not goal_positions:
            return self._generate_random_positions(min_distance, max_distance)
        
        # 随机选择起点和终点
        start_pos = random.choice(start_positions)
        
        # 筛选在有效距离范围内的终点
        valid_goals = []
        for pos in goal_positions:
            dist = abs(pos[0] - start_pos[0]) + abs(pos[1] - start_pos[1])
            if min_distance <= dist <= max_distance:
                valid_goals.append(pos)
        
        # 如果没有有效终点，重新选择
        if not valid_goals:
            return self._generate_random_positions(min_distance, max_distance)
        
        goal_pos = random.choice(valid_goals)
        return start_pos, goal_pos
    
    def _generate_random_positions(self, min_distance: int, max_distance: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """生成随机的起点和终点位置
        
        Args:
            min_distance: 最小距离
            max_distance: 最大距离
            
        Returns:
            Tuple[Tuple[int, int], Tuple[int, int]]: 起点和终点位置
        """
        # 随机选择起点
        start_pos = random.choice(self.available_positions)
        
        # 筛选在有效距离范围内的终点位置
        valid_goal_positions = []
        for pos in self.available_positions:
            dist = abs(pos[0] - start_pos[0]) + abs(pos[1] - start_pos[1])
            if min_distance <= dist <= max_distance:
                valid_goal_positions.append(pos)
        
        # 如果没有有效终点，放宽条件
        if not valid_goal_positions:
            valid_goal_positions = [pos for pos in self.available_positions if pos != start_pos]
        
        # 随机选择终点
        goal_pos = random.choice(valid_goal_positions) if valid_goal_positions else start_pos
        
        return start_pos, goal_pos
    
    def _generate_task_features(self, start_pos: Tuple[int, int], goal_pos: Tuple[int, int], 
                              priority: TaskPriority) -> Dict[str, float]:
        """生成任务特征
        
        Args:
            start_pos: 起点位置
            goal_pos: 终点位置
            priority: 任务优先级
            
        Returns:
            Dict[str, float]: 任务特征字典
        """
        # 计算距离
        distance = abs(goal_pos[0] - start_pos[0]) + abs(goal_pos[1] - start_pos[1])
        
        # 基于距离和优先级计算奖励
        base_reward = distance * 0.5
        priority_multiplier = {
            TaskPriority.LOW: 1.0,
            TaskPriority.MEDIUM: 1.5,
            TaskPriority.HIGH: 2.0
        }
        reward = base_reward * priority_multiplier[priority]
        
        # 计算任务特征
        urgency = 0.0
        complexity = min(1.0, distance / self.map_size)
        estimated_energy = distance * 0.8  # 估计能量消耗
        
        # 根据优先级设置紧急程度
        if priority == TaskPriority.HIGH:
            urgency = random.uniform(0.7, 1.0)
        elif priority == TaskPriority.MEDIUM:
            urgency = random.uniform(0.3, 0.7)
        else:
            urgency = random.uniform(0.0, 0.3)
        
        # 根据障碍物地图调整复杂度（如果有）
        if self.obstacle_map is not None and hasattr(self, 'pathfinder'):
            path = self.pathfinder.find_path(start_pos, goal_pos)
            if path:
                # 真实路径长度与直线距离的比值作为复杂度指标
                path_length = len(path)
                complexity = min(1.0, (path_length - distance) / (2 * distance))
                
                # 更新估计能量消耗
                estimated_energy = path_length * 0.8
        
        return {
            "reward": reward,
            "urgency": urgency,
            "complexity": complexity,
            "estimated_energy": estimated_energy,
            "estimated_time": distance * 1.2
        }
    
    def generate_dynamic_tasks(self, current_step: int, max_tasks: int = 10) -> List[Task]:
        """动态生成任务
        
        Args:
            current_step: 当前时间步
            max_tasks: 最大同时存在的任务数量
            
        Returns:
            List[Task]: 新生成的任务列表
        """
        new_tasks = []
        
        # 清理过期或完成的任务
        self.tasks = [task for task in self.tasks 
                     if task.status != TaskStatus.COMPLETED and 
                     (task.deadline is None or current_step <= task.deadline)]
        
        # 当前任务数量少于最大值时，考虑生成新任务
        while len(self.tasks) < max_tasks:
            # 根据当前步数动态调整生成概率
            generation_prob = 0.1  # 基础生成概率
            if len(self.tasks) < max_tasks * 0.5:  # 当任务较少时提高生成概率
                generation_prob = 0.3
            
            if random.random() < generation_prob:
                # 生成新任务
                task = self.generate_task(current_step=current_step)
                self.tasks.append(task)
                new_tasks.append(task)
        
        return new_tasks
    
    def update_task_status(self, task: Task, status: TaskStatus, completion_time: Optional[int] = None):
        """更新任务状态和统计信息
        
        Args:
            task: 要更新的任务
            status: 新状态
            completion_time: 完成时间（如果有）
        """
        if status == TaskStatus.COMPLETED:
            self.statistics['total_completed'] += 1
            self.statistics['priority_completion_rates'][task.priority]['completed'] += 1
            if completion_time is not None and task.appear_time is not None:
                completion_duration = completion_time - task.appear_time
                # 更新平均完成时间
                total_completed = self.statistics['total_completed']
                current_avg = self.statistics['average_completion_time']
                self.statistics['average_completion_time'] = (
                    (current_avg * (total_completed - 1) + completion_duration) / total_completed
                )
        elif status == TaskStatus.FAILED:
            self.statistics['total_failed'] += 1
        elif status == TaskStatus.CANCELLED:
            self.statistics['total_cancelled'] += 1
        
        task.status = status
    
    def generate_tasks(self, num_tasks: int, 
                     required_path: bool = True,
                     min_distance: int = 5) -> List[Task]:
        """批量生成任务
        
        Args:
            num_tasks: 要生成的任务数量
            required_path: 是否要求起点和终点之间必须存在有效路径
            min_distance: 起点和终点之间的最小距离
            
        Returns:
            List[Task]: 生成的任务列表
        """
        tasks = []
        for _ in range(num_tasks):
            task = self.generate_task(required_path=required_path, min_distance=min_distance)
            tasks.append(task)
        return tasks
    
    def reset(self):
        """重置任务生成器"""
        self.next_task_id = 0
        
    def get_available_tasks(self) -> List[Task]:
        """获取可用任务列表
        
        返回所有处于PENDING状态的任务
        
        Returns:
            List[Task]: 可用任务列表
        """
        return [task for task in self.tasks if task.status == TaskStatus.PENDING]
        
    def get_task_statistics(self) -> Dict[str, Any]:
        """获取任务统计信息
        
        Returns:
            Dict[str, Any]: 任务统计信息
        """
        # 计算任务完成率
        completion_rate = 0.0
        if self.statistics['total_generated'] > 0:
            completion_rate = self.statistics['total_completed'] / self.statistics['total_generated']
            
        # 计算各优先级任务完成率
        priority_rates = {}
        for priority in TaskPriority:
            total = self.statistics['priority_completion_rates'][priority]['total']
            completed = self.statistics['priority_completion_rates'][priority]['completed']
            rate = completed / total if total > 0 else 0.0
            priority_rates[priority.name] = rate
            
        # 返回统计信息
        return {
            'total_generated': self.statistics['total_generated'],
            'total_completed': self.statistics['total_completed'],
            'total_failed': self.statistics['total_failed'],
            'total_cancelled': self.statistics['total_cancelled'],
            'completion_rate': completion_rate,
            'average_completion_time': self.statistics['average_completion_time'],
            'priority_completion_rates': priority_rates
        }
        
    def to_dict(self) -> Dict[str, Any]:
        """将任务生成器转换为字典
        
        返回:
            Dict[str, Any]: 任务生成器字典表示
        """
        return {
            'map_size': self.map_size,
            'task_id_counter': self.next_task_id,
            'tasks': [
                {
                    'id': task.id,
                    'start_pos': task.start_pos.tolist(),
                    'goal_pos': task.goal_pos.tolist(),
                    'priority': task.priority.value,
                    'deadline': task.deadline,
                    'status': task.status.value
                }
                for task in self.tasks
            ]
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskGenerator':
        """从字典创建任务生成器
        
        Args:
            data: 任务生成器字典表示
            
        Returns:
            TaskGenerator: 任务生成器实例
        """
        generator = cls(
            map_size=data['map_size']
        )
        
        generator.next_task_id = data['task_id_counter']
        
        # 恢复任务
        for task_data in data['tasks']:
            priority = TaskPriority(task_data['priority'])
            status = TaskStatus(task_data['status'])
            
            task = Task(
                id=task_data['id'],
                start_pos=np.array(task_data['start_pos']),
                goal_pos=np.array(task_data['goal_pos']),
                priority=priority,
                deadline=task_data['deadline']
            )
            task.status = status
            generator.tasks.append(task)
            
        return generator
    
    def set_obstacle_map(self, obstacle_map: np.ndarray):
        """设置障碍物地图
        
        Args:
            obstacle_map: 障碍物地图，1表示障碍物，0表示空地
        """
        self.obstacle_map = obstacle_map
        self.available_positions = self._get_available_positions()
        
        # 更新路径查找器
        if hasattr(self, 'pathfinder'):
            self.pathfinder.set_obstacle_map(obstacle_map) 