import enum
import numpy as np
from typing import Dict, Any, Optional

class TaskStatus(enum.Enum):
    PENDING = 0
    ASSIGNED = 1
    IN_PROGRESS = 2
    COMPLETED = 3
    FAILED = 4

class TaskPriority(enum.Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4
    
    @classmethod
    def random(cls):
        """随机选择一个优先级
        
        Returns:
            TaskPriority: 随机优先级
        """
        import random
        return random.choice(list(cls))

class Task:
    """任务类
    
    描述智能体需要完成的任务，包括起点、终点、优先级等属性
    """
    def __init__(self, 
                 id: int, 
                 start_pos: np.ndarray, 
                 goal_pos: np.ndarray, 
                 priority: TaskPriority = TaskPriority.MEDIUM, 
                 deadline: Optional[int] = None,
                 estimated_energy: float = 10.0):
        """初始化任务
        
        Args:
            id: 任务ID
            start_pos: 起始位置
            goal_pos: 目标位置
            priority: 优先级
            deadline: 截止时间
            estimated_energy: 预计消耗的能量
        """
        self.id = id
        self.start_pos = np.array(start_pos)
        self.goal_pos = np.array(goal_pos)
        self.priority = priority
        self.deadline = deadline
        self.estimated_energy = estimated_energy
        
        self.status = TaskStatus.PENDING
        self.assigned_agent_id = None
        self.assigned_time = None
        self.completed_time = None
        self.failed_time = None
        self.fail_reason = None
        self.started = False
        self.appear_time = None
        
        # 任务特征
        self.features = {
            'urgency': 0.5,
            'complexity': 0.5,
            'reward': 10.0,
        }
        
        # 根据优先级更新紧急程度
        if priority == TaskPriority.LOW:
            self.features['urgency'] = 0.2
        elif priority == TaskPriority.MEDIUM:
            self.features['urgency'] = 0.5
        elif priority == TaskPriority.HIGH:
            self.features['urgency'] = 0.8
        elif priority == TaskPriority.URGENT:
            self.features['urgency'] = 1.0
    
    def assign_to(self, agent_id: int, current_time: int):
        """将任务分配给智能体
        
        Args:
            agent_id: 智能体ID
            current_time: 当前时间
        """
        self.assigned_agent_id = agent_id
        self.assigned_time = current_time
        self.status = TaskStatus.ASSIGNED
    
    def assign_to_agent(self, agent_id: int):
        """将任务分配给智能体（简化版本，用于可视化）
        
        Args:
            agent_id: 智能体ID
        """
        self.assigned_agent_id = agent_id
        self.status = TaskStatus.ASSIGNED
    
    def complete(self, current_time: int):
        """完成任务
        
        Args:
            current_time: 当前时间
        """
        self.completed_time = current_time
        self.status = TaskStatus.COMPLETED
    
    def fail(self, reason: str, current_time: int):
        """任务失败
        
        Args:
            reason: 失败原因
            current_time: 当前时间
        """
        self.failed_time = current_time
        self.fail_reason = reason
        self.status = TaskStatus.FAILED
    
    def is_overdue(self, current_time: int) -> bool:
        """检查任务是否超期
        
        Args:
            current_time: 当前时间
            
        Returns:
            bool: 是否超期
        """
        if self.deadline is None:
            return False
        return current_time > self.deadline
    
    def to_dict(self) -> Dict[str, Any]:
        """将任务转换为字典
        
        Returns:
            Dict[str, Any]: 任务字典
        """
        return {
            'id': self.id,
            'start_pos': self.start_pos.tolist(),
            'goal_pos': self.goal_pos.tolist(),
            'priority': self.priority.value,
            'deadline': self.deadline,
            'estimated_energy': self.estimated_energy,
            'status': self.status.value,
            'assigned_agent_id': self.assigned_agent_id,
            'assigned_time': self.assigned_time,
            'completed_time': self.completed_time,
            'failed_time': self.failed_time,
            'fail_reason': self.fail_reason,
            'features': self.features
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """从字典创建任务
        
        Args:
            data: 任务字典
            
        Returns:
            Task: 任务对象
        """
        task = cls(
            id=data['id'],
            start_pos=np.array(data['start_pos']),
            goal_pos=np.array(data['goal_pos']),
            priority=TaskPriority(data['priority']),
            deadline=data['deadline'],
            estimated_energy=data['estimated_energy']
        )
        
        task.status = TaskStatus(data['status'])
        task.assigned_agent_id = data['assigned_agent_id']
        task.assigned_time = data['assigned_time']
        task.completed_time = data['completed_time']
        task.failed_time = data['failed_time']
        task.fail_reason = data['fail_reason']
        task.features = data['features']
        
        return task 