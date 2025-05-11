import numpy as np
from typing import Optional, Dict, Any, Tuple
from .task import Task, TaskStatus
import random

class Agent:
    """智能体类
    
    具有位置、电量、任务执行等功能
    """
    def __init__(self, id: int, pos: Tuple[int, int], map_size: int = 40, max_battery: float = 100.0, 
                 communication_range: float = 10.0):
        """初始化智能体
        
        Args:
            id: 智能体ID
            pos: 初始位置
            map_size: 地图大小
            max_battery: 最大电量
            communication_range: 通信范围
        """
        self.id = id
        self.pos = np.array(pos)
        self.map_size = map_size  # 保存地图大小
        self.max_battery = max_battery
        self.current_battery = max_battery
        self.communication_range = communication_range
        
        # 任务相关
        self.current_task: Optional[Task] = None
        self.task_progress = 0.0  # 任务完成进度
        self.task_history = []  # 任务历史记录
        
        # 状态信息
        self.state = {
            'position': np.array(pos),
            'battery': self.current_battery,
            'experience': 0.0,  # 经验值 (0-5)
            'total_distance': 0.0,  # 总移动距离
            'total_tasks_completed': 0,
            'idle_time': 0,  # 空闲时间
            'charging': False,  # 是否在充电
            'exploring': False,  # 是否在探索模式
            'completed_tasks_by_priority': {},  # 按优先级统计完成的任务数量
            'available': True,  # 是否可用
            'task_completion_rate': 0  # 任务完成率
        }
        
        # 初始化路径规划器
        from pathfinding.models.dhc.pathfinder import PathFinder
        self.pathfinder = PathFinder(map_size)
        
        # 添加无效移动标志
        self.attempted_invalid_move = False
        self.attempted_obstacle_collision = False
        self.consecutive_obstacle_collisions = 0  # 连续碰撞计数
        
        # 初始化消息缓冲区
        self.message_buffer = []
        
    # 动作映射字典
    ACTIONS = {
        0: np.array([0, 0]),   # 不动
        1: np.array([0, 1]),   # 上
        2: np.array([1, 0]),   # 右
        3: np.array([0, -1]),  # 下
        4: np.array([-1, 0])   # 左
    }
    
    def select_action(self, action_values, epsilon=0.1):
        """选择动作，协调Q学习和A*算法
        
        Args:
            action_values: Q网络输出的动作值
            epsilon: 探索率，默认0.1
            
        Returns:
            int: 选择的动作
        """
        # 检查是否卡住
        if self._is_stuck():
            print(f"Agent {self.id}: 检测到卡住，尝试特殊逃脱策略")
            return self._handle_stuck_situation(action_values)
        
        # 获取当前位置
        current_pos = tuple(self.pos)
        
        # 创建动作掩码，阻止向障碍物方向移动
        action_mask = np.ones(5)  # 5个动作：停止，上，右，下，左
        
        # 检查每个方向是否有障碍物
        for action_id in range(1, 5):  # 跳过停止动作(0)
            direction = self.ACTIONS[action_id]
            new_pos = self.pos + direction
            
            # 检查是否超出地图边界
            if not (0 <= new_pos[0] < self.map_size and 0 <= new_pos[1] < self.map_size):
                action_mask[action_id] = 0
                continue
            
            # 检查是否有障碍物
            if hasattr(self.pathfinder, 'obstacle_map') and self.pathfinder.obstacle_map is not None:
                if self.pathfinder.obstacle_map[new_pos[0], new_pos[1]] == 1:
                    action_mask[action_id] = 0
                    continue
                
            # 检查是否与其他智能体碰撞（如果有这个信息）
            if hasattr(self, 'other_agents_positions'):
                for other_pos in self.other_agents_positions:
                    if np.array_equal(new_pos, other_pos):
                        action_mask[action_id] = 0
                        break
        
        # 应用掩码到动作值
        masked_action_values = action_values * action_mask
        
        # 1. 检查是否需要充电（最高优先级）
        if self.need_charging():
            nearest_station = self.find_nearest_charging_station()
            if nearest_station:
                path = self.pathfinder.find_path(current_pos, nearest_station)
                if path and len(path) > 1:
                    action = self._get_action_from_path(path)
                    print(f"Agent {self.id}: 电量低，前往充电站，选择动作 {action}")
                    return action
        
        # 2. 检查是否有任务
        if self.current_task:
            # 确定目标位置（起点或终点）
            goal = None
            if not hasattr(self.current_task, 'reached_start') or not self.current_task.reached_start:
                goal = self.current_task.start_pos
                print(f"Agent {self.id}: 前往任务{self.current_task.id}起点 {goal}")
            else:
                goal = self.current_task.goal_pos
                print(f"Agent {self.id}: 前往任务{self.current_task.id}终点 {goal}")
            
            # 生成新的引导层，增强对目标的导向
            if goal is not None:
                # 增强引导层的作用
                self.pathfinder.generate_path_guidance_layer(current_pos, goal)
                
                # 使用A*算法规划路径
                path = self.pathfinder.find_path(current_pos, goal)
                if path and len(path) > 1:
                    # 增加A*算法的权重，从90%提高到97%
                    use_a_star = np.random.random() < 0.97  # 原来是0.9
                    
                    # 计算目标到当前位置的直线距离
                    distance_to_goal = np.linalg.norm(np.array(goal) - self.pos)
                    
                    # 分析路径质量（长度和直接性）
                    path_quality = 1.0  # 默认质量
                    if len(path) > 2:
                        # 如果路径过长，可能不是最优的，降低权重
                        if len(path) > distance_to_goal * 2:
                            path_quality *= 0.8
                        
                        # 检查路径直接性：如果路径有很多转向，可能不是最优的
                        direction_changes = 0
                        for i in range(1, len(path) - 1):
                            dir1 = (path[i][0] - path[i-1][0], path[i][1] - path[i-1][1])
                            dir2 = (path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
                            if dir1 != dir2:
                                direction_changes += 1
                        if direction_changes > len(path) / 3:  # 如果转向次数超过路径长度的1/3
                            path_quality *= 0.9
                    
                    # 动态调整A*的使用比例
                    use_a_star = use_a_star or (path_quality > 0.9 and np.random.random() < 0.99)
                    
                    # 如果处于测试模式（测试输出目前的决策逻辑）
                    if hasattr(self, 'test_mode') and self.test_mode:
                        print(f"A*路径：{path}")
                        print(f"路径质量：{path_quality}")
                        print(f"是否使用A*：{use_a_star}")
                    
                    if use_a_star:
                        action = self._get_action_from_path(path)
                        print(f"Agent {self.id}: 使用A*路径规划，选择动作 {action}")
                        return action
                    
                    # 即使不用A*，使用路径信息来影响探索
                    if len(path) > 1:
                        next_pos = path[1]
                        direction = np.array(next_pos) - self.pos
                        next_action = self._direction_to_action(direction)
                        
                        # 检查是否是有效动作
                        if action_mask[next_action] > 0:
                            # 如果DQN与A*建议不同，有70%概率采纳A*建议
                            if np.argmax(masked_action_values) != next_action:
                                if np.random.random() < 0.7:  # 原来是0.5，增加采纳概率
                                    print(f"Agent {self.id}: DQN与A*不一致，采纳A*建议，选择动作 {next_action}")
                                    return next_action
        
        # 3. 使用强化学习加入随机性决策
        # 一定概率随机探索
        if np.random.random() < epsilon:
            # 基于路径引导向量增强随机探索
            guidance_vector = self.pathfinder.get_guidance_vector(current_pos)
            if guidance_vector != (0, 0):
                # 使用引导向量确定可能的动作，增加按引导方向移动的概率
                possible_actions = []
                
                # 根据引导向量偏向性选择动作
                if guidance_vector[0] > 0.2:  # 向右
                    possible_actions.extend([2] * 3 if action_mask[2] > 0 else [])  # 增加向右移动的概率
                elif guidance_vector[0] < -0.2:  # 向左
                    possible_actions.extend([4] * 3 if action_mask[4] > 0 else [])  # 增加向左移动的概率
                    
                if guidance_vector[1] > 0.2:  # 向上
                    possible_actions.extend([1] * 3 if action_mask[1] > 0 else [])  # 增加向上移动的概率
                elif guidance_vector[1] < -0.2:  # 向下
                    possible_actions.extend([3] * 3 if action_mask[3] > 0 else [])  # 增加向下移动的概率
                
                # 添加停止动作和可行的移动动作
                possible_actions.append(0)  # 停止动作总是可行的
                for i in range(1, 5):
                    if action_mask[i] > 0:
                        possible_actions.append(i)
                    
                if possible_actions:
                    action = np.random.choice(possible_actions)
                    print(f"Agent {self.id}: 随机探索 (有引导), 选择动作 {action}")
                    return action
                else:
                    print(f"Agent {self.id}: 无可行动作，选择停止")
                    return 0  # 如果没有可行动作，则停止
            else:
                # 随机选择一个有效动作
                valid_actions = [i for i in range(5) if action_mask[i] > 0]
                if valid_actions:
                    action = np.random.choice(valid_actions)
                    print(f"Agent {self.id}: 随机探索, 选择动作 {action}")
                    return action
                else:
                    print(f"Agent {self.id}: 无可行动作，选择停止")
                    return 0  # 如果没有可行动作，则停止
                    
        # 4. 根据DQN模型选择动作
        if np.max(masked_action_values) > 0:
            # 动作值矩阵中可能有多个最大值，随机选择一个
            max_indices = np.where(masked_action_values == np.max(masked_action_values))[0]
            action = np.random.choice(max_indices)
            print(f"Agent {self.id}: 使用DQN，选择动作 {action}")
            return action
        else:
            # 所有动作都不可行，选择停止
            print(f"Agent {self.id}: 无可行DQN动作，选择停止")
            return 0
    
    def _direction_to_action(self, direction):
        """将方向向量转换为动作ID
        
        Args:
            direction: 方向向量 [dx, dy]
            
        Returns:
            int: 动作ID (0=停止, 1=上, 2=右, 3=下, 4=左)
        """
        dx, dy = direction
        
        # 映射到动作ID
        if dx == 0 and dy == 0:
            return 0  # 不动
        elif dx == 0 and dy > 0:
            return 1  # 上
        elif dx > 0 and dy == 0:
            return 2  # 右
        elif dx == 0 and dy < 0:
            return 3  # 下
        elif dx < 0 and dy == 0:
            return 4  # 左
        else:
            # 对角线移动，选择x或y方向
            if abs(dx) > abs(dy):
                return 2 if dx > 0 else 4  # 右或左
            else:
                return 1 if dy > 0 else 3  # 上或下
    
    def _check_for_obstacles(self, direction):
        """检查某个方向是否有障碍物
        
        Args:
            direction: (dx, dy) 方向向量
            
        Returns:
            bool: 如果有障碍物返回True，否则False
        """
        dx, dy = direction
        new_pos = (self.pos[0] + dx, self.pos[1] + dy)
        
        # 检查是否超出地图边界
        if not (0 <= new_pos[0] < self.map_size and 0 <= new_pos[1] < self.map_size):
            return True
            
        # 检查是否有障碍物
        if hasattr(self.pathfinder, 'obstacle_map') and self.pathfinder.obstacle_map is not None:
            if self.pathfinder.obstacle_map[new_pos[0], new_pos[1]] == 1:
                return True
                
        return False
    
    def _is_stuck(self):
        """检测智能体是否卡住
        
        如果连续多次碰撞或原地不动，视为卡住
        
        Returns:
            bool: 是否卡住
        """
        if not hasattr(self, 'position_history'):
            self.position_history = []
            self.stuck_counter = 0
            self.oscillation_counter = 0  # 新增震荡计数器
            self.last_movement_direction = None  # 记录上一次移动方向
            return False
            
        # 将当前位置添加到历史记录
        current_pos = tuple(self.pos)
        
        # 检查方向变化
        if len(self.position_history) > 0:
            last_pos = self.position_history[-1]
            current_direction = None
            
            # 计算移动方向
            if current_pos != last_pos:
                dx = current_pos[0] - last_pos[0]
                dy = current_pos[1] - last_pos[1]
                current_direction = (dx, dy)
                
                # 检测方向反转（上下或左右震荡）
                if self.last_movement_direction is not None and current_direction is not None:
                    # 检查是否是反向移动
                    if (current_direction[0] == -self.last_movement_direction[0] and current_direction[0] != 0) or \
                       (current_direction[1] == -self.last_movement_direction[1] and current_direction[1] != 0):
                        self.oscillation_counter += 1
                        print(f"Agent {self.id}: 检测到方向反转，震荡计数: {self.oscillation_counter}")
                    else:
                        # 不是反向移动，重置震荡计数
                        self.oscillation_counter = max(0, self.oscillation_counter - 1)
                
                # 更新上一次移动方向
                self.last_movement_direction = current_direction
        
        self.position_history.append(current_pos)
        
        # 保持历史记录长度
        if len(self.position_history) > 10:
            self.position_history.pop(0)
            
        # 判断是否卡在原地或在小范围区域内来回移动
        if len(self.position_history) >= 5:  # 减少判断所需的历史长度
            # 取最后5个位置
            recent_positions = self.position_history[-5:]
            unique_positions = set(recent_positions)
            
            # 如果只有1-2个不同位置，认为卡住了
            if len(unique_positions) <= 2:
                self.stuck_counter += 1
                print(f"Agent {self.id}: 少量位置变化，卡住计数: {self.stuck_counter}")
                return self.stuck_counter >= 2  # 降低卡住阈值
            
            # 检查是否在几个位置之间来回移动
            position_counts = {}
            for pos in recent_positions:
                position_counts[pos] = position_counts.get(pos, 0) + 1
                
            # 如果某个位置出现次数超过2次，认为卡住了
            for pos, count in position_counts.items():
                if count >= 2:
                    self.stuck_counter += 1
                    print(f"Agent {self.id}: 位置重复出现，卡住计数: {self.stuck_counter}")
                    return self.stuck_counter >= 2  # 降低卡住阈值
            
            # 检查震荡状态
            if self.oscillation_counter >= 2:  # 如果连续震荡2次以上
                print(f"Agent {self.id}: 检测到持续震荡，判定为卡住")
                return True
        
        # 重置卡住计数器
        self.stuck_counter = 0
        return False
    
    def _handle_stuck_situation(self, action_values):
        """处理卡住的情况
        
        采用更激进的策略摆脱卡住状态
        
        Args:
            action_values: 动作值
            
        Returns:
            int: 选择的动作
        """
        print(f"Agent {self.id} is stuck! Trying to escape...")
        
        # 获取当前位置
        x, y = self.pos
        
        # 检测是否是上下震荡或左右震荡
        oscillation_type = None
        if hasattr(self, 'position_history') and len(self.position_history) >= 4:
            # 计算最近几步的移动方向
            directions = []
            for i in range(len(self.position_history) - 1):
                prev = self.position_history[i]
                curr = self.position_history[i + 1]
                dx = curr[0] - prev[0]
                dy = curr[1] - prev[1]
                if dx != 0 or dy != 0:  # 只记录实际移动
                    directions.append((dx, dy))
            
            # 检测是否是上下震荡
            if len(directions) >= 2:
                vertical_oscillation = True
                for i in range(len(directions) - 1):
                    if directions[i][1] != 0 and directions[i+1][1] != 0:
                        if directions[i][1] != -directions[i+1][1]:
                            vertical_oscillation = False
                            break
                    else:
                        vertical_oscillation = False
                        break
                
                # 检测是否是左右震荡
                horizontal_oscillation = True
                for i in range(len(directions) - 1):
                    if directions[i][0] != 0 and directions[i+1][0] != 0:
                        if directions[i][0] != -directions[i+1][0]:
                            horizontal_oscillation = False
                            break
                    else:
                        horizontal_oscillation = False
                        break
                
                if vertical_oscillation:
                    oscillation_type = "vertical"
                    print(f"Agent {self.id}: 检测到上下震荡，尝试水平移动")
                elif horizontal_oscillation:
                    oscillation_type = "horizontal"
                    print(f"Agent {self.id}: 检测到左右震荡，尝试垂直移动")
        
        # 1. 尝试找到一个没有障碍的方向
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 上、右、下、左
        valid_actions = []
        
        # 如果检测到特定类型的震荡，调整优先方向
        if oscillation_type == "vertical":
            # 对于上下震荡，优先考虑水平方向（左右）
            priority_directions = [(1, 0), (-1, 0)]  # 右、左
            for dx, dy in priority_directions:
                new_x, new_y = x + dx, y + dy
                if (0 <= new_x < self.map_size and 
                    0 <= new_y < self.map_size and 
                    not self._check_for_obstacles((dx, dy))):
                    # 将水平方向的动作权重增加
                    action = self._direction_to_action(np.array([dx, dy]))
                    valid_actions.extend([action] * 5)  # 增加权重，使更可能选择水平方向
        elif oscillation_type == "horizontal":
            # 对于左右震荡，优先考虑垂直方向（上下）
            priority_directions = [(0, 1), (0, -1)]  # 上、下
            for dx, dy in priority_directions:
                new_x, new_y = x + dx, y + dy
                if (0 <= new_x < self.map_size and 
                    0 <= new_y < self.map_size and 
                    not self._check_for_obstacles((dx, dy))):
                    # 将垂直方向的动作权重增加
                    action = self._direction_to_action(np.array([dx, dy]))
                    valid_actions.extend([action] * 5)  # 增加权重，使更可能选择垂直方向
        
        # 为所有可行的方向添加基础权重
        for i, (dx, dy) in enumerate(directions, 1):
            new_x, new_y = x + dx, y + dy
            
            # 检查是否在地图内且没有障碍物
            if (0 <= new_x < self.map_size and 
                0 <= new_y < self.map_size and 
                not self._check_for_obstacles((dx, dy))):
                valid_actions.append(i)  # 动作id: 1-4
        
        # 2. 如果有有效动作，使用加权随机选择
        if valid_actions:
            # 结合最近没有选过的动作
            recent_actions = []
            if hasattr(self, 'position_history') and len(self.position_history) >= 2:
                # 查找最近实际执行的动作
                for i in range(len(self.position_history) - 1):
                    prev = self.position_history[i]
                    curr = self.position_history[i + 1]
                    if prev != curr:  # 只考虑实际移动
                        dx = curr[0] - prev[0]
                        dy = curr[1] - prev[1]
                        recent_action = self._direction_to_action(np.array([dx, dy]))
                        recent_actions.append(recent_action)
            
            # 为近期未使用的方向增加权重
            weighted_actions = []
            for action in set(valid_actions):  # 去重
                # 检查是否是最近使用过的动作
                is_recent = action in recent_actions
                
                # 如果不是最近使用过的，给更高权重
                if not is_recent:
                    weight = 3  # 基础权重
                    # 如果与当前任务方向一致，给额外权重
                    if self.current_task:
                        target_pos = None
                        if hasattr(self.current_task, 'reached_start') and self.current_task.reached_start:
                            target_pos = self.current_task.goal_pos
                        else:
                            target_pos = self.current_task.start_pos
                        
                        if target_pos:
                            # 计算目标方向
                            target_dir = np.array([target_pos[0] - x, target_pos[1] - y])
                            if np.linalg.norm(target_dir) > 0:
                                target_dir = target_dir / np.linalg.norm(target_dir)
                                action_dir = self.ACTIONS[action]
                                # 计算方向相似度（点积）
                                similarity = np.dot(target_dir, action_dir)
                                if similarity > 0.5:  # 如果方向基本一致
                                    weight += 2  # 额外权重
                else:
                    weight = 1  # 最近使用过的动作权重较低
                
                weighted_actions.extend([action] * weight)
            
            # 如果没有加权动作（罕见情况），使用原始有效动作
            if not weighted_actions:
                weighted_actions = valid_actions
            
            # 随机选择一个加权后的动作
            if weighted_actions:
                action = np.random.choice(weighted_actions)
                print(f"Agent {self.id}: 脱困选择动作 {action}, 候选动作: {set(weighted_actions)}")
                return action
            
            # 添加随机停止的可能性
            if np.random.random() < 0.1:  # 10%概率选择停止
                print(f"Agent {self.id}: 脱困策略 - 随机选择停止")
                return 0
        
        # 3. 实在没有有效动作，尝试按DQN最高值行动
        return np.argmax(action_values)
    
    def _is_path_safe(self, path):
        """检查路径是否安全
        
        Args:
            path: A*算法规划的路径
            
        Returns:
            bool: 路径是否安全
        """
        if not path:
            return False
        
        # 检查路径上的每个点
        for pos in path:
            # 检查是否在障碍物上
            if hasattr(self.pathfinder, 'obstacle_map') and self.pathfinder.obstacle_map is not None:
                if self.pathfinder.obstacle_map[pos[0], pos[1]] == 1:
                    return False
            
            # 检查是否在未知区域
            if not self.pathfinder.explored_area[pos[0], pos[1]]:
                return False
        
        return True
    
    def find_nearest_charging_station(self):
        """查找最近的充电站
        
        Returns:
            Tuple[int, int]: 最近充电站的位置
        """
        if not hasattr(self, 'task_allocator') or not self.task_allocator:
            return None
            
        nearest_station, distance = self.task_allocator.find_nearest_charging_station(self.pos)
        return tuple(nearest_station) if nearest_station is not None else None
        
    def _is_in_unknown_area(self):
        """检查当前位置是否在未知区域
        
        Returns:
            bool: 是否在未知区域
        """
        if hasattr(self.pathfinder, 'explored_area'):
            return not self.pathfinder.explored_area[self.pos[0], self.pos[1]]
        return False
    
    def check_position_valid(self, new_pos, check_collisions=True):
        """检查新位置是否有效
        
        Args:
            new_pos: 新位置
            check_collisions: 是否检查与其他智能体的碰撞
            
        Returns:
            bool: 位置是否有效
        """
        new_pos = np.array(new_pos)
        
        # 检查是否超出地图边界
        if not (0 <= new_pos[0] < self.map_size and 0 <= new_pos[1] < self.map_size):
            return False
            
        # 检查是否有障碍物
        if hasattr(self.pathfinder, 'obstacle_map') and self.pathfinder.obstacle_map is not None:
            if self.pathfinder.obstacle_map[new_pos[0], new_pos[1]] == 1:
                return False
                
        # 检查与其他智能体的碰撞
        if check_collisions and hasattr(self, 'task_allocator') and self.task_allocator:
            for agent_id, other_agent in self.task_allocator.agents.items():
                if agent_id != self.id and np.array_equal(other_agent.pos, new_pos):
                    return False
                    
        return True
        
    def execute_action(self, action_id: int) -> bool:
        """执行动作
        
        Args:
            action_id: 动作ID (0-4)
            
        Returns:
            bool: 动作是否执行成功
        """
        if action_id not in self.ACTIONS:
            return False
            
        # 添加基础电量消耗，即使不移动也会消耗少量电量
        if action_id == 0:  # 停止动作
            self.update_battery(-0.05)  # 停止也消耗少量电量
            
        # 计算新位置
        action = self.ACTIONS[action_id]
        new_pos = self.pos + action
        
        # 检查新位置是否有效
        # 添加标志记录是否尝试进入无效位置
        self.attempted_invalid_move = False  # 重置标志
        if not self.check_position_valid(new_pos):
            # 记录尝试进入无效位置
            self.attempted_invalid_move = True
            
            # 检查是否是尝试进入障碍物
            if (0 <= new_pos[0] < self.map_size and 0 <= new_pos[1] < self.map_size and
                hasattr(self.pathfinder, 'obstacle_map') and 
                self.pathfinder.obstacle_map is not None and
                self.pathfinder.obstacle_map[new_pos[0], new_pos[1]] == 1):
                # 尝试进入障碍物
                self.attempted_obstacle_collision = True
            
            return False
            
        # 更新位置
        return self.update_position(new_pos)
    
    def update_position(self, new_pos):
        """更新智能体位置
        
        Args:
            new_pos: 新位置
        """
        # 记录旧位置
        old_pos = self.pos.copy()
        
        # 更新位置
        self.pos = np.array(new_pos)
        
        # 计算移动距离
        distance = np.linalg.norm(self.pos - old_pos)
        
        # 更新总移动距离
        self.state['total_distance'] += distance
        
        # 检查是否尝试了无效移动
        if np.array_equal(old_pos, new_pos) and not np.array_equal(old_pos, [0, 0]):
            self.attempted_invalid_move = True
        else:
            self.attempted_invalid_move = False
            
        # 如果移动了，检查是否是无效位置
        if distance > 0:
            # 检查是否有障碍物
            if hasattr(self.pathfinder, 'obstacle_map') and self.pathfinder.obstacle_map is not None:
                if self.pathfinder.obstacle_map[new_pos[0], new_pos[1]] == 1:
                    self.attempted_obstacle_collision = True
                    self.consecutive_obstacle_collisions += 1
                else:
                    self.attempted_obstacle_collision = False
                    self.consecutive_obstacle_collisions = 0
        
        # 记录位置变化，用于判断是否卡住
        if not hasattr(self, 'position_history'):
            self.position_history = []
        self.position_history.append(tuple(new_pos))
        
        # 保持历史记录长度在限制内
        if len(self.position_history) > 20:
            self.position_history.pop(0)
        
        # 更新状态
        self.state['position'] = new_pos
        self.state['idle_time'] = 0
        
        return True
        
    def _calculate_movement_energy_cost(self, distance: float) -> float:
        """计算移动能耗
        
        Args:
            distance: 移动距离
            
        Returns:
            float: 能耗值
        """
        # 基础消耗: 每单位距离消耗0.1）
        base_consumption = distance * 0.1
        
        # 负载因子: 如果有任务，消耗增加20%
        load_factor = 1.2 if self.current_task else 1.0
        
        # 电量效率因子: 电量越低，效率越低，消耗越多
        efficiency_factor = 1.0 + max(0, (0.8 - self.current_battery / self.max_battery)) * 0.5
        
        return base_consumption * load_factor * efficiency_factor
    
    def update_battery(self, amount: float):
        """更新电量
        
        Args:
            amount: 电量变化值（正数为充电，负数为消耗）
        """
        self.current_battery = np.clip(self.current_battery + amount, 0, self.max_battery)
        self.state['battery'] = self.current_battery
        
    def need_charging(self) -> bool:
        """判断是否需要充电
        
        考虑当前任务和未来任务的能耗预测，以及到充电站的距离和路径可行性
        
        Returns:
            bool: 是否需要充电
        """
        # 获取最近的充电站和距离
        if hasattr(self, 'task_allocator'):
            nearest_station, distance = self.task_allocator.find_nearest_charging_station(self.pos)
            if nearest_station is not None:
                # 检查到充电站的路径是否可行
                path_to_station = self.pathfinder.find_path(self.pos, tuple(nearest_station))
                if not path_to_station:
                    # 如果无法到达充电站，需要更早开始寻找充电站
                    return self.current_battery < self.max_battery * 0.5
                
                # 计算到达充电站所需的能量
                energy_to_station = self._calculate_movement_energy_cost(distance)
                
                # 动态电量阈值
                base_threshold = 0.3  # 基础阈值
                if self.current_task:
                    # 根据任务紧急程度调整阈值
                    if self.current_task.priority == TaskPriority.URGENT:
                        base_threshold = 0.2
                    elif self.current_task.priority == TaskPriority.HIGH:
                        base_threshold = 0.25
                    elif self.current_task.priority == TaskPriority.MEDIUM:
                        base_threshold = 0.3
                    else:
                        base_threshold = 0.35
                
                # 考虑到达充电站所需的能量
                effective_threshold = base_threshold + (energy_to_station / self.max_battery)
                
                # 如果电量低于有效阈值，需要充电
                if self.current_battery < self.max_battery * effective_threshold:
                    return True
                
                # 如果非常接近充电站（距离小于2），且电量不满，考虑充电
                if distance < 2 and self.current_battery < self.max_battery * 0.8:
                    return True
        
        return False
        
    def _predict_future_energy_need(self) -> float:
        """预测未来任务的能耗需求
        
        基于历史数据和当前状态预测
        
        Returns:
            float: 预计能耗
        """
        # 基础能耗（每步消耗）
        base_consumption = 0.01
        
        # 估算未来20步的能耗
        future_steps = 20
        
        # 根据历史任务计算平均任务能耗
        avg_task_energy = 0.0
        if hasattr(self, 'state') and 'total_tasks_completed' in self.state:
            completed_tasks = max(1, self.state['total_tasks_completed'])
            total_distance = self.state['total_distance']
            avg_task_energy = (total_distance * 0.1) / completed_tasks  # 平均每个任务的能耗
        
        # 预测未来任务数
        expected_tasks = future_steps / 20  # 假设平均每20步有一个新任务
        
        # 计算总预期能耗
        total_predicted_energy = (
            base_consumption * future_steps +  # 基础能耗
            avg_task_energy * expected_tasks   # 任务能耗
        )
        
        return total_predicted_energy
    
    def estimate_task_energy_cost(self) -> float:
        """估算完成当前任务所需的电量
        
        Returns:
            float: 估算的电量消耗
        """
        if not self.current_task:
            return 0.0
            
        # 计算总距离
        if not self.current_task.started:
            # 还未开始任务，需要先到达起点
            distance_to_start = np.linalg.norm(self.pos - self.current_task.start_pos)
            task_distance = np.linalg.norm(self.current_task.goal_pos - self.current_task.start_pos)
            total_distance = distance_to_start + task_distance
        else:
            # 已经开始任务，只需要计算到终点的距离
            total_distance = np.linalg.norm(self.pos - self.current_task.goal_pos)
            
        return self._calculate_movement_energy_cost(total_distance)
    
    def start_charging(self):
        """开始充电"""
        self.state['charging'] = True
        
    def stop_charging(self):
        """停止充电"""
        self.state['charging'] = False
        
    def charge(self, amount: float = 1.0):
        """充电
        
        Args:
            amount: 充电量
        """
        if self.state['charging']:
            self.update_battery(amount)
            
    def assign_task(self, task: Task) -> bool:
        """分配任务
        
        Args:
            task: 要分配的任务
            
        Returns:
            bool: 是否成功分配
        """
        if self.current_task is not None:
            return False
            
        self.current_task = task
        task.status = TaskStatus.IN_PROGRESS
        self.task_progress = 0.0
        
        # 记录任务分配历史
        self.task_history.append({
            'task_id': task.id,
            'status': 'assigned',
            'start_time': self.state.get('current_step', 0),
            'start_pos': task.start_pos,
            'goal_pos': task.goal_pos,
            'priority': task.priority.value
        })
        
        return True
        
    def update_task_status(self) -> bool:
        """更新任务状态
        
        Returns:
            bool: 任务是否完成
        """
        if not self.current_task:
            return False
            
        # 检查是否到达任务起点
        if not self.current_task.started:
            if np.array_equal(self.pos, self.current_task.start_pos):
                self.current_task.started = True
                self.state['exploring'] = False  # 到达起点后停止探索
                return False
            else:
                # 如果还未到达起点，使用路径规划器导航
                next_pos = self.pathfinder.get_next_move(tuple(self.pos), tuple(self.current_task.start_pos))
                if self.update_position(next_pos):
                    return False
                else:
                    # 如果无法移动，可能需要探索
                    self.state['exploring'] = True
                    return False
            
        # 检查是否完成任务
        if self.current_task.started:
            if np.array_equal(self.pos, self.current_task.goal_pos):
                self.current_task.status = TaskStatus.COMPLETED
                self.state['total_tasks_completed'] += 1
                self.state['experience'] = min(5.0, self.state['experience'] + 0.1)
                self.current_task = None
                self.task_progress = 0.0
                self.state['exploring'] = False
                return True
            else:
                # 使用路径规划器导航到目标点
                next_pos = self.pathfinder.get_next_move(tuple(self.pos), tuple(self.current_task.goal_pos))
                if self.update_position(next_pos):
                    return False
                else:
                    # 如果无法移动，可能需要探索
                    self.state['exploring'] = True
                    return False
            
        return False
        
    def update(self):
        """更新智能体状态"""
        # 更新空闲时间
        if self.current_task is None and not self.state['charging']:
            self.state['idle_time'] += 1
            # 空闲时进行探索
            if not self.state['exploring']:
                exploration_target = self.pathfinder.get_exploration_target(tuple(self.pos))
                if exploration_target:
                    self.state['exploring'] = True
                    next_pos = self.pathfinder.get_next_move(tuple(self.pos), exploration_target)
                    self.update_position(next_pos)
            
        # 自然电量消耗
        if not self.state['charging']:
            self.update_battery(-0.01)  # 每步骤消耗0.01电量
            
        # 更新探索地图
        if hasattr(self, 'observation') and self.observation is not None:
            self.pathfinder.update_explored_area(tuple(self.pos), self.observation)
            
    def get_state(self) -> Dict[str, Any]:
        """获取智能体状态
        
        Returns:
            Dict[str, Any]: 状态信息
        """
        return {
            'id': self.id,
            'position': self.pos.tolist(),
            'battery': self.current_battery,
            'max_battery': self.max_battery,
            'task': self.current_task.id if self.current_task else None,
            'task_progress': self.task_progress,
            'experience': self.state['experience'],
            'total_distance': self.state['total_distance'],
            'total_tasks_completed': self.state['total_tasks_completed'],
            'idle_time': self.state['idle_time'],
            'charging': self.state['charging']
        }

    def move_to(self, new_pos):
        """移动到新位置
        
        简化版的位置更新，主要用于可视化
        
        Args:
            new_pos: 新位置
        """
        self.update_position(new_pos)
        
    def complete_task(self):
        """完成当前任务"""
        if self.current_task is None:
            return
            
        old_task = self.current_task
        self.state['position'] = old_task.goal_pos
        self.state['total_distance'] += np.linalg.norm(old_task.goal_pos - old_task.start_pos)
        self.state['total_tasks_completed'] += 1
        self.state['experience'] = min(5.0, self.state['experience'] + 0.1)
        self.current_task = None
        self.state['available'] = True
        self.state['task_completion_rate'] = 100
        
        # 更新任务历史
        current_step = self.state.get('current_step', 0)
        self.task_history.append({
            'task_id': old_task.id,
            'priority': old_task.priority.value,
            'start_pos': old_task.start_pos.tolist(),
            'goal_pos': old_task.goal_pos.tolist(),
            'completion_time': current_step,  # 使用当前步数作为完成时间
            'total_steps': current_step - (old_task.assigned_time or 0)  # 如果assigned_time存在就使用它，否则用0
        })
        
        # 更新任务对象的完成时间（如果支持）
        if hasattr(old_task, 'complete') and callable(getattr(old_task, 'complete')):
            try:
                old_task.complete(current_step)
            except TypeError:
                # 如果complete方法不接受参数，就不传入
                pass
        
        # 更新完成任务统计
        priority = old_task.priority.value
        if 'completed_tasks_by_priority' not in self.state:
            self.state['completed_tasks_by_priority'] = {}
        self.state['completed_tasks_by_priority'][priority] = self.state['completed_tasks_by_priority'].get(priority, 0) + 1
        
    def update_task_progress(self, progress_increment):
        """更新当前任务进度
        
        Args:
            progress_increment: 进度增量
            
        Returns:
            bool: 任务是否完成
        """
        if not self.current_task:
            return False
            
        current_progress = self.state['task_completion_rate']
        new_progress = min(100, current_progress + progress_increment)
        self.state['task_completion_rate'] = new_progress
        
        # 如果任务完成，更新任务状态
        if new_progress >= 100:
            self.current_task.status = TaskStatus.COMPLETED
            return True
            
        return False
        
    def can_communicate_with(self, other_agent):
        """判断是否可以与另一个智能体通信
        
        考虑通信质量和障碍物影响
        
        Args:
            other_agent: 另一个智能体
            
        Returns:
            bool: 是否可以通信
        """
        if self.id == other_agent.id:
            return False
            
        # 计算距离
        distance = np.linalg.norm(self.pos - other_agent.pos)
        
        # 基础通信范围检查
        if distance > self.communication_range:
            return False
            
        # 计算通信质量衰减
        quality_factor = 1.0 - (distance / self.communication_range) ** 2
        
        # 检查障碍物影响
        if hasattr(self, 'pathfinder') and self.pathfinder.explored_map is not None:
            # 使用Bresenham算法检查视线
            x1, y1 = self.pos.astype(int)
            x2, y2 = other_agent.pos.astype(int)
            line_points = self._get_line_points(x1, y1, x2, y2)
            
            # 计算障碍物影响
            obstacle_count = 0
            for x, y in line_points:
                if 0 <= x < self.pathfinder.explored_map.shape[0] and \
                   0 <= y < self.pathfinder.explored_map.shape[1] and \
                   self.pathfinder.explored_map[x, y] == 1:
                    obstacle_count += 1
            
            # 障碍物越多，通信质量越差
            quality_factor *= max(0, 1 - 0.2 * obstacle_count)
        
        # 通信质量阈值
        return quality_factor >= 0.3
        
    def _get_line_points(self, x1, y1, x2, y2):
        """使用Bresenham算法获取两点之间的所有点"""
        points = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        x, y = x1, y1
        sx = 1 if x2 > x1 else -1
        sy = 1 if y2 > y1 else -1
        
        if dx > dy:
            err = dx / 2.0
            while x != x2:
                points.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y2:
                points.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
                
        points.append((x, y))
        return points
        
    def send_message(self, recipient_id, message_type, content):
        """发送消息给另一个智能体
        
        用于无标签化协作中的信息交换
        
        Args:
            recipient_id: 接收者ID
            message_type: 消息类型 ('task_info', 'battery_status', 'position', etc.)
            content: 消息内容
        """
        message = {
            'sender_id': self.id,
            'recipient_id': recipient_id,
            'type': message_type,
            'content': content,
            'timestamp': self.state.get('current_step', 0)
        }
        self.message_buffer.append(message)
        
    def receive_messages(self, messages):
        """接收其他智能体发送的消息
        
        Args:
            messages: 消息列表
        """
        for message in messages:
            if message['recipient_id'] == self.id or message['recipient_id'] == -1:  # -1表示广播
                self.process_message(message)
                
    def process_message(self, message):
        """处理接收到的消息
        
        根据消息类型执行不同操作
        
        Args:
            message: 消息
        """
        if message['type'] == 'task_info':
            # 处理任务信息消息
            pass
        elif message['type'] == 'battery_status':
            # 处理电量状态消息
            pass
        elif message['type'] == 'position':
            # 处理位置信息消息
            pass
            
    def evaluate_task(self, task) -> float:
        """评估任务对自身的适合程度
        
        考虑任务紧急程度、路径复杂度和负载均衡
        
        Args:
            task: 要评估的任务
            
        Returns:
            float: 评分，分数越高表示越适合
        """
        # 如果电量不足，直接返回负分
        if self.current_battery < task.estimated_energy:
            return -1000.0
            
        # 计算到任务起点的距离
        distance = np.linalg.norm(self.pos - task.start_pos)
        
        # 计算电量因子 (电量越多越好)
        battery_factor = self.current_battery / self.max_battery
        
        # 计算经验因子 (经验越丰富越好)
        experience_factor = min(1.0, self.state['experience'] / 10.0)
        
        # 计算空闲时间因子 (空闲时间越长越应该分配任务)
        idle_factor = min(1.0, self.state['idle_time'] / 20.0)
        
        # 计算任务紧急程度因子
        urgency_factor = 1.0
        if hasattr(task, 'deadline') and task.deadline is not None:
            time_left = task.deadline - task.appear_time
            urgency_factor = 2.0 if time_left < 30 else (1.5 if time_left < 60 else 1.0)
        
        # 计算路径复杂度因子
        path_complexity = 1.0
        if hasattr(self, 'pathfinder'):
            # 使用A*算法估算路径复杂度
            path = self.pathfinder.get_path(tuple(self.pos), tuple(task.start_pos))
            if path:
                # 路径转折次数作为复杂度指标
                turns = 0
                for i in range(1, len(path)-1):
                    if path[i-1][0] != path[i+1][0] and path[i-1][1] != path[i+1][1]:
                        turns += 1
                path_complexity = 1.0 / (1.0 + 0.1 * turns)  # 转折越多，复杂度越高
        
        # 计算负载均衡因子
        workload_factor = 1.0
        if hasattr(self, 'state') and 'total_tasks_completed' in self.state:
            # 完成任务数越多，接受新任务的倾向越低
            workload_factor = 1.0 / (1.0 + 0.1 * self.state['total_tasks_completed'])
        
        # 计算综合得分
        score = (
            (1.0 - distance / 20.0) * 3.0 +    # 距离因子 (距离越近越好)
            battery_factor * 2.0 +             # 电量因子
            experience_factor * 1.0 +          # 经验因子
            idle_factor * 1.5 +                # 空闲时间因子
            urgency_factor * 2.0 +             # 紧急程度因子
            path_complexity * 1.5 +            # 路径复杂度因子
            workload_factor * 1.0              # 负载均衡因子
        )
        
        return score
            
    def decide_next_task(self, available_tasks):
        """决定下一个要执行的任务
        
        无标签化方法中，智能体自主决策
        
        Args:
            available_tasks: 可用的任务列表
            
        Returns:
            最佳任务的索引，如果没有适合的任务则返回None
        """
        if not available_tasks or not self.state['available']:
            return None
            
        # 计算每个任务的得分
        task_scores = []
        for task in available_tasks:
            if task.status != 'pending':
                continue
                
            score = self.evaluate_task(task)
            task_scores.append((task, score))
            
        # 按得分降序排序
        task_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 如果有任务分数大于阈值，返回最高分的任务
        if task_scores and task_scores[0][1] > -100:
            return task_scores[0][0]
            
        return None
    
    def update(self, current_step):
        """更新智能体状态
        
        Args:
            current_step: 当前时间步
        """
        # 更新当前步骤
        self.state['current_step'] = current_step
        
        # 静态电量消耗 (即使不移动也会有少量消耗)
        self.update_battery(-0.01)
        
        # 如果有任务，更新任务进度
        if self.current_task and self.current_task.status == 'assigned':
            # 如果已经到达任务起点，向目标移动
            if np.array_equal(self.pos, self.current_task.start_pos):
                # 更新任务开始状态
                self.current_task.status = 'in_progress'
                
            # 如果任务正在进行中
            if self.current_task.status == 'in_progress':
                # 如果到达目标点，完成任务
                if np.array_equal(self.pos, self.current_task.goal_pos):
                    progress_increment = 100 - self.state['task_completion_rate']
                    is_completed = self.update_task_progress(progress_increment)
                    if is_completed:
                        self.complete_task()
                else:
                    # 正在前往目标点，更新进度
                    # 进度基于到目标的距离
                    total_distance = np.linalg.norm(
                        self.current_task.goal_pos - self.current_task.start_pos)
                    current_distance = np.linalg.norm(
                        self.current_task.goal_pos - self.pos)
                    
                    # 计算完成百分比
                    if total_distance > 0:
                        completion = (total_distance - current_distance) / total_distance * 100
                        self.state['task_completion_rate'] = completion

    def _get_action_from_path(self, path):
        """从路径中获取下一步动作
        
        Args:
            path: A*算法计算的路径
            
        Returns:
            int: 动作ID
        """
        # 确保路径有效
        if not path or len(path) < 2:
            return 0  # 如果路径无效，返回停止动作
        
        # 获取下一个位置
        next_pos = path[1]  # 当前位置是path[0]，下一步是path[1]
        
        # 安全检查：确保下一个位置不是障碍物
        if hasattr(self.pathfinder, 'obstacle_map') and self.pathfinder.obstacle_map is not None:
            if self.pathfinder.obstacle_map[next_pos[0], next_pos[1]] == 1:
                print(f"Agent {self.id}: A*路径下一步是障碍物，放弃该路径")
                
                # 尝试修复：寻找次优路径
                if len(path) > 2:
                    # 尝试看路径中的后续位置是否可达
                    for i in range(2, min(5, len(path))):
                        alt_next = path[i]
                        if self.pathfinder.obstacle_map[alt_next[0], alt_next[1]] == 0:
                            # 尝试直接前往这个位置
                            alt_path = self.pathfinder.find_path(self.pos, alt_next)
                            if alt_path and len(alt_path) > 1:
                                next_pos = alt_path[1]
                                print(f"Agent {self.id}: 找到替代路径，新的下一步是 {next_pos}")
                                break
                
                # 如果无法找到替代路径，随机选择一个非障碍方向
                if self.pathfinder.obstacle_map[next_pos[0], next_pos[1]] == 1:
                    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 上、右、下、左
                    random.shuffle(directions)  # 随机排序
                    
                    for dx, dy in directions:
                        new_x, new_y = self.pos[0] + dx, self.pos[1] + dy
                        if (0 <= new_x < self.map_size and 
                            0 <= new_y < self.map_size and 
                            self.pathfinder.obstacle_map[new_x, new_y] == 0):
                            next_pos = (new_x, new_y)
                            print(f"Agent {self.id}: 随机选择非障碍方向，新的下一步是 {next_pos}")
                            break
                    
                    # 如果所有方向都有障碍，返回停止
                    if self.pathfinder.obstacle_map[next_pos[0], next_pos[1]] == 1:
                        return 0
        
        # 计算方向向量
        direction = np.array(next_pos) - self.pos
        
        # 检查是否接近目标 - 如果靠近目标，尝试更精准的移动
        is_near_goal = False
        if self.current_task and self.current_task.status != TaskStatus.PENDING:
            # 确定目标位置（起点或终点）
            target_pos = self.current_task.goal_pos if self.current_task.reached_start else self.current_task.start_pos
            distance_to_target = np.linalg.norm(np.array(target_pos) - self.pos)
            
            # 如果非常接近目标（距离小于3），启用更精确的导航
            if distance_to_target < 3.0:
                is_near_goal = True
                
                # 检查到目标之间是否有障碍物
                has_obstacle = False
                if hasattr(self.pathfinder, 'obstacle_map') and self.pathfinder.obstacle_map is not None:
                    # 使用Bresenham算法检查视线
                    x1, y1 = self.pos.astype(int)
                    x2, y2 = target_pos
                    line_points = self._get_line_points(x1, y1, x2, y2)
                    
                    # 检查路径上是否有障碍物
                    for x, y in line_points:
                        if (0 <= x < self.pathfinder.obstacle_map.shape[0] and 
                            0 <= y < self.pathfinder.obstacle_map.shape[1] and 
                            self.pathfinder.obstacle_map[x, y] == 1):
                            has_obstacle = True
                            break
                
                if not has_obstacle:
                    # 如果没有障碍物，使用精确导航
                    if distance_to_target < 1.5:
                        # 直接计算到目标的精确方向
                        goal_direction = np.array(target_pos) - self.pos
                        direction = goal_direction
                        print(f"Agent {self.id}: 非常接近目标且无障碍物，使用精确导航")
                    else:
                        # 结合A*方向和目标方向
                        a_star_dir = direction / (np.linalg.norm(direction) or 1)
                        goal_dir = (np.array(target_pos) - self.pos) / (np.linalg.norm(np.array(target_pos) - self.pos) or 1)
                        
                        # 混合两个方向，接近目标时更倾向于直接朝目标
                        blend_factor = 0.8  # 80%目标方向，20%A*方向，增强直接朝目标移动的倾向
                        combined_dir = blend_factor * goal_dir + (1 - blend_factor) * a_star_dir
                        
                        # 重新计算方向
                        if np.linalg.norm(combined_dir) > 0:
                            direction = combined_dir
                            print(f"Agent {self.id}: 接近目标且无障碍物，混合导航策略")
                else:
                    # 如果有障碍物，继续使用A*路径
                    print(f"Agent {self.id}: 接近目标但有障碍物，继续使用A*路径")
        
        # 增加检测卡住和震荡的逻辑
        if hasattr(self, 'position_history') and len(self.position_history) >= 4:
            # 检查最近的位置
            recent_positions = self.position_history[-4:]
            
            # 检查是否已经在相同位置上下移动
            vertical_oscillation = self._check_vertical_oscillation(recent_positions)
            if vertical_oscillation:
                print(f"Agent {self.id}: 检测到路径上的垂直震荡，尝试水平移动")
                # 优先选择水平方向
                horizontal_directions = [(1, 0), (-1, 0)]  # 右、左
                for dx, dy in horizontal_directions:
                    new_x, new_y = self.pos[0] + dx, self.pos[1] + dy
                    if (0 <= new_x < self.map_size and 
                        0 <= new_y < self.map_size and 
                        (hasattr(self.pathfinder, 'obstacle_map') and
                         self.pathfinder.obstacle_map is not None and
                         self.pathfinder.obstacle_map[new_x, new_y] == 0)):
                        direction = np.array([dx, dy])
                        print(f"Agent {self.id}: 为避免垂直震荡，选择水平移动：{direction}")
                        break
            
            # 检查是否已经在相同位置左右移动
            horizontal_oscillation = self._check_horizontal_oscillation(recent_positions)
            if horizontal_oscillation:
                print(f"Agent {self.id}: 检测到路径上的水平震荡，尝试垂直移动")
                # 优先选择垂直方向
                vertical_directions = [(0, 1), (0, -1)]  # 上、下
                for dx, dy in vertical_directions:
                    new_x, new_y = self.pos[0] + dx, self.pos[1] + dy
                    if (0 <= new_x < self.map_size and 
                        0 <= new_y < self.map_size and 
                        (hasattr(self.pathfinder, 'obstacle_map') and
                         self.pathfinder.obstacle_map is not None and
                         self.pathfinder.obstacle_map[new_x, new_y] == 0)):
                        direction = np.array([dx, dy])
                        print(f"Agent {self.id}: 为避免水平震荡，选择垂直移动：{direction}")
                        break
        
        # 转换为动作ID
        action = self._direction_to_action(direction)
        
        # 如果接近目标但动作会导致远离目标，考虑不动
        if is_near_goal and self.current_task:
            target_pos = self.current_task.goal_pos if self.current_task.reached_start else self.current_task.start_pos
            current_dist = np.linalg.norm(self.pos - np.array(target_pos))
            new_pos = self.pos + self.ACTIONS[action]
            new_dist = np.linalg.norm(new_pos - np.array(target_pos))
            
            # 检查新位置是否有障碍物
            has_obstacle = False
            if hasattr(self.pathfinder, 'obstacle_map') and self.pathfinder.obstacle_map is not None:
                if (0 <= new_pos[0] < self.pathfinder.obstacle_map.shape[0] and 
                    0 <= new_pos[1] < self.pathfinder.obstacle_map.shape[1] and 
                    self.pathfinder.obstacle_map[new_pos[0], new_pos[1]] == 1):
                    has_obstacle = True
            
            # 如果新位置比当前位置离目标更远，且当前已经很接近，且新位置没有障碍物
            if new_dist > current_dist and current_dist < 1.2 and not has_obstacle:
                print(f"Agent {self.id}: 非常接近目标但动作会导致远离，选择停止")
                return 0  # 停止不动
        
        return action
        
    def _check_vertical_oscillation(self, positions):
        """检查是否存在垂直震荡（上下来回移动）
        
        Args:
            positions: 位置历史
            
        Returns:
            bool: 是否存在垂直震荡
        """
        if len(positions) < 4:
            return False
            
        # 提取纵坐标变化
        y_changes = []
        for i in range(1, len(positions)):
            y_change = positions[i][1] - positions[i-1][1]
            if y_change != 0:  # 只关注垂直方向的变化
                y_changes.append(y_change)
                
        # 至少需要2个垂直方向的变化才能判断
        if len(y_changes) < 2:
            return False
            
        # 检查最近的垂直变化是否呈现震荡（正负交替）
        for i in range(1, len(y_changes)):
            if y_changes[i] * y_changes[i-1] < 0:  # 符号相反，说明方向反转
                return True
                
        return False
        
    def _check_horizontal_oscillation(self, positions):
        """检查是否存在水平震荡（左右来回移动）
        
        Args:
            positions: 位置历史
            
        Returns:
            bool: 是否存在水平震荡
        """
        if len(positions) < 4:
            return False
            
        # 提取横坐标变化
        x_changes = []
        for i in range(1, len(positions)):
            x_change = positions[i][0] - positions[i-1][0]
            if x_change != 0:  # 只关注水平方向的变化
                x_changes.append(x_change)
                
        # 至少需要2个水平方向的变化才能判断
        if len(x_changes) < 2:
            return False
            
        # 检查最近的水平变化是否呈现震荡（正负交替）
        for i in range(1, len(x_changes)):
            if x_changes[i] * x_changes[i-1] < 0:  # 符号相反，说明方向反转
                return True
                
        return False

    def get_observation(self, obstacle_map=None, agents=None):
        """获取智能体的观察信息
        
        Args:
            obstacle_map: 障碍物地图
            agents: 其他智能体列表
            
        Returns:
            ndarray: 观察信息 (8, 9, 9)
        """
        # 观察窗口大小为9x9，观察半径为4
        obs_radius = 4
        obs = np.zeros((8, 9, 9))
        
        # 如果有障碍物地图，提取观察窗口
        if obstacle_map is not None:
            # 计算观察窗口范围
            x_min = max(0, self.pos[0] - obs_radius)
            x_max = min(self.map_size, self.pos[0] + obs_radius + 1)
            y_min = max(0, self.pos[1] - obs_radius)
            y_max = min(self.map_size, self.pos[1] + obs_radius + 1)
            
            # 计算填充范围
            pad_x_min = obs_radius - (self.pos[0] - x_min)
            pad_x_max = obs_radius + 1 + (x_max - self.pos[0] - 1)
            pad_y_min = obs_radius - (self.pos[1] - y_min)
            pad_y_max = obs_radius + 1 + (y_max - self.pos[1] - 1)
            
            # 1. 障碍物层
            obs[0, pad_x_min:pad_x_max, pad_y_min:pad_y_max] = obstacle_map[x_min:x_max, y_min:y_max]
        
        # 2. 其他智能体位置层
        if agents is not None:
            agent_map = np.zeros((self.map_size, self.map_size))
            for agent in agents:
                if agent.id != self.id:  # 不包括自己
                    agent_x, agent_y = agent.pos
                    if 0 <= agent_x < self.map_size and 0 <= agent_y < self.map_size:
                        agent_map[agent_x, agent_y] = 1
                        
            # 提取观察窗口
            if obstacle_map is not None:  # 确保已计算窗口范围
                obs[1, pad_x_min:pad_x_max, pad_y_min:pad_y_max] = agent_map[x_min:x_max, y_min:y_max]
        
        # 3-4. 任务起点和终点层
        task_start_map = np.zeros((self.map_size, self.map_size))
        task_goal_map = np.zeros((self.map_size, self.map_size))
        
        if self.current_task:
            start_x, start_y = self.current_task.start_pos
            goal_x, goal_y = self.current_task.goal_pos
            
            if 0 <= start_x < self.map_size and 0 <= start_y < self.map_size:
                task_start_map[start_x, start_y] = 1
            if 0 <= goal_x < self.map_size and 0 <= goal_y < self.map_size:
                task_goal_map[goal_x, goal_y] = 1
                
        # 提取观察窗口
        if obstacle_map is not None:  # 确保已计算窗口范围
            obs[2, pad_x_min:pad_x_max, pad_y_min:pad_y_max] = task_start_map[x_min:x_max, y_min:y_max]
            obs[3, pad_x_min:pad_x_max, pad_y_min:pad_y_max] = task_goal_map[x_min:x_max, y_min:y_max]
        
        # 5-6. 任务方向层 - 改进：使用A*算法计算最优路径
        # 初始化方向层
        path_direction_x = np.zeros((self.map_size, self.map_size))
        path_direction_y = np.zeros((self.map_size, self.map_size))
        
        if self.current_task and obstacle_map is not None:
            # 确定目标位置
            target_pos = self.current_task.start_pos if not self.current_task.started else self.current_task.goal_pos
            
            # 使用A*算法计算最优路径
            path = self.pathfinder.find_path(tuple(self.pos), tuple(target_pos))
            
            if path and len(path) > 1:
                # 路径存在且至少有一步可走
                # 为观察窗口内的每个位置计算方向场
                for dx in range(-obs_radius, obs_radius + 1):
                    for dy in range(-obs_radius, obs_radius + 1):
                        nx, ny = self.pos[0] + dx, self.pos[1] + dy
                        if 0 <= nx < self.map_size and 0 <= ny < self.map_size:
                            # 只处理非障碍物位置
                            if obstacle_map[nx, ny] == 0:
                                # 计算从该位置到目标的路径
                                cell_path = self.pathfinder.find_path((nx, ny), tuple(target_pos))
                                
                                if cell_path and len(cell_path) > 1:
                                    # 获取下一步方向
                                    next_x, next_y = cell_path[1]
                                    direction = np.array([next_x - nx, next_y - ny])
                                    
                                    # 归一化方向向量
                                    norm = np.linalg.norm(direction)
                                    if norm > 0:
                                        direction = direction / norm
                                        
                                    # 填充方向信息
                                    path_direction_x[nx, ny] = direction[0]
                                    path_direction_y[nx, ny] = direction[1]
            else:
                # 如果A*算法找不到路径，回退到简单的方向指引
                for dx in range(-obs_radius, obs_radius + 1):
                    for dy in range(-obs_radius, obs_radius + 1):
                        nx, ny = self.pos[0] + dx, self.pos[1] + dy
                        if 0 <= nx < self.map_size and 0 <= ny < self.map_size:
                            # 只处理非障碍物位置
                            if obstacle_map[nx, ny] == 0:
                                # 计算朝向目标的简单方向
                                direction = np.array(target_pos) - np.array([nx, ny])
                                
                                # 归一化方向向量
                                norm = np.linalg.norm(direction)
                                if norm > 0:
                                    direction = direction / norm
                                    
                                # 填充方向信息
                                path_direction_x[nx, ny] = direction[0]
                                path_direction_y[nx, ny] = direction[1]
        
        # 提取观察窗口
        if obstacle_map is not None:
            obs[4, pad_x_min:pad_x_max, pad_y_min:pad_y_max] = path_direction_x[x_min:x_max, y_min:y_max]
            obs[5, pad_x_min:pad_x_max, pad_y_min:pad_y_max] = path_direction_y[x_min:x_max, y_min:y_max]
        
        # 7. 充电站层
        station_map = np.zeros((self.map_size, self.map_size))
        # 填充充电站位置
        if hasattr(self, 'charging_stations') and self.charging_stations:
            for station in self.charging_stations:
                station_x, station_y = station
                if 0 <= station_x < self.map_size and 0 <= station_y < self.map_size:
                    station_map[station_x, station_y] = 1
                    
        # 提取观察窗口
        if obstacle_map is not None:
            obs[6, pad_x_min:pad_x_max, pad_y_min:pad_y_max] = station_map[x_min:x_max, y_min:y_max]
        
        # 8. 电池层
        battery_map = np.zeros((self.map_size, self.map_size))
        # 在当前位置填充电池电量
        battery_map[self.pos[0], self.pos[1]] = self.current_battery / self.max_battery
        
        # 提取观察窗口
        if obstacle_map is not None:
            obs[7, pad_x_min:pad_x_max, pad_y_min:pad_y_max] = battery_map[x_min:x_max, y_min:y_max]
        
        return obs