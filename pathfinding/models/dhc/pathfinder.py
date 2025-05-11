import numpy as np
from typing import List, Tuple, Optional, Dict
from queue import PriorityQueue

class PathFinder:
    """路径规划器类
    
    实现A*寻路算法和探索策略，帮助智能体在部分可观察环境中导航
    """
    def __init__(self, map_size: int):
        """初始化路径规划器
        
        Args:
            map_size: 地图大小
        """
        self.map_size = map_size
        self.explored_map = np.full((map_size, map_size), -1)  # -1表示未知，0表示空地，1表示障碍
        self.obstacle_map = None  # 障碍物地图
        self.obstacle_density_map = np.zeros((map_size, map_size))
        self.current_path = []
        self.explored_area = np.zeros((map_size, map_size), dtype=bool)  # 已探索区域
        # 添加路径引导层
        self.path_guidance_layer = np.zeros((map_size, map_size, 2))  # 引导层，包含x和y方向的引导
        
    def set_obstacle_map(self, obstacle_map: np.ndarray):
        """设置障碍物地图
        
        Args:
            obstacle_map: 障碍物地图，0表示空地，1表示障碍物
        """
        if obstacle_map.shape != (self.map_size, self.map_size):
            raise ValueError(f"障碍物地图大小 {obstacle_map.shape} 与初始化地图大小 {self.map_size} 不匹配")
        
        # 更新障碍物地图
        self.obstacle_map = obstacle_map
        
        # 更新explored_map
        for i in range(self.map_size):
            for j in range(self.map_size):
                if obstacle_map[i, j] == 1:
                    self.explored_map[i, j] = 1  # 标记为障碍物
                else:
                    self.explored_map[i, j] = 0  # 标记为空地
                    
        # 更新障碍物密度图
        for i in range(self.map_size):
            for j in range(self.map_size):
                self.obstacle_density_map[i, j] = self._calculate_obstacle_density((i, j))
        
        # 初始化所有区域为已探索
        self.explored_area.fill(True)
        
    def _calculate_obstacle_density(self, pos, radius=3):
        """计算指定位置周围的障碍物密度"""
        x, y = pos
        density = 0
        count = 0
        
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.map_size and 0 <= ny < self.map_size:
                    if self.explored_map[nx, ny] == 1:  # 障碍物
                        density += 1
                    count += 1
                    
        return density / count if count > 0 else 0
        
    def _is_affected_by_obstacle(self, node, obstacle):
        """检查节点是否受到新障碍物的影响"""
        # 如果节点就是障碍物位置
        if node == obstacle:
            return True
            
        # 如果节点在障碍物周围3格范围内
        if abs(node[0] - obstacle[0]) <= 3 and abs(node[1] - obstacle[1]) <= 3:
            return True
            
        return False
        
    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """改进的A*算法，考虑已知/未知区域和动态规划"""
        if not self._is_valid_position(start) or not self._is_valid_position(goal):
            return None
            
        # 初始化
        open_set = {start}
        closed_set = set()
        came_from = {}
        
        # 初始化g_score和f_score
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, goal)}
        
        while open_set:
            # 获取f_score最小的节点
            current = min(open_set, key=lambda x: f_score.get(x, float('inf')))
            
            if current == goal:
                return self._reconstruct_path(came_from, current)
                
            open_set.remove(current)
            closed_set.add(current)
            
            # 检查是否需要重新规划
            if self._should_replan(current, came_from):
                return self._replan_path(current, goal, closed_set)
                
            # 遍历邻居节点
            for neighbor in self._get_neighbors(current):
                if neighbor in closed_set:
                    continue
                    
                # 计算新的g_score
                tentative_g_score = g_score[current] + self._get_cost(current, neighbor)
                
                if neighbor not in open_set:
                    open_set.add(neighbor)
                elif tentative_g_score >= g_score.get(neighbor, float('inf')):
                    continue
                    
                # 更新路径
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, goal)
                
        return None
        
    def _heuristic(self, node, goal):
        """改进的启发式函数"""
        # 基础曼哈顿距离
        base_cost = abs(node[0] - goal[0]) + abs(node[1] - goal[1])
        
        # 未知区域惩罚（减小权重，更愿意探索）
        if self.explored_map[node] == -1:
            base_cost *= 1.2  # 原来是1.5，降低对未知区域的惩罚
            
        # 障碍物密度惩罚（增强权重，更好地避开障碍物密集区）
        density = self._calculate_obstacle_density(node)
        base_cost *= (1 + 0.8 * density)  # 原来是1 + density，增加对障碍物密集区的避让
        
        # 添加目标吸引力，使路径更倾向于朝目标方向前进
        # 计算当前节点到目标的单位向量方向
        if node != goal:
            dx = goal[0] - node[0]
            dy = goal[1] - node[1]
            # 单位化
            length = max(abs(dx), abs(dy))
            if length > 0:
                dx /= length
                dy /= length
                
            # 检查下一步是否可以沿着目标方向移动
            next_x = int(node[0] + dx)
            next_y = int(node[1] + dy)
            
            # 如果下一步位置有效且无障碍，给予额外奖励（减少代价）
            if (0 <= next_x < self.map_size and 0 <= next_y < self.map_size and 
                self.obstacle_map is not None and self.obstacle_map[next_x, next_y] == 0):
                # 给予方向性奖励 - 减轻直接朝向目标方向移动的代价
                base_cost *= 0.9  # 奖励朝目标方向的移动
                
            # 检查是否存在从目标到当前节点的直接视线
            if (self.obstacle_map is not None and 
                self._has_line_of_sight(node, goal)):
                # 有直接视线时大幅减轻成本
                base_cost *= 0.7
        
        # 检测和惩罚潜在的震荡路径
        if hasattr(self, 'current_path') and len(self.current_path) >= 3:
            # 检查当前节点是否会导致震荡
            if self._would_cause_oscillation(node):
                # 对可能导致震荡的路径增加成本
                base_cost *= 1.5
                # print(f"检测到可能的震荡路径，增加成本: {node}")
        
        return base_cost
        
    def _would_cause_oscillation(self, node):
        """检查添加该节点是否会导致路径震荡
        
        Args:
            node: 要检查的节点
            
        Returns:
            bool: 是否可能导致震荡
        """
        if not hasattr(self, 'current_path') or len(self.current_path) < 3:
            return False
            
        # 获取最近的几个路径点
        recent_path = self.current_path[-3:]
        
        # 检查是否是水平方向的震荡
        if node[0] == recent_path[0][0] and abs(node[1] - recent_path[0][1]) <= 1:
            # 检查最近的移动是否也是水平方向
            if recent_path[1][0] == recent_path[0][0] and recent_path[2][0] == recent_path[1][0]:
                # 检查是否是来回移动
                if ((node[1] - recent_path[0][1]) * (recent_path[1][1] - recent_path[0][1]) < 0 or
                    (recent_path[2][1] - recent_path[1][1]) * (recent_path[1][1] - recent_path[0][1]) < 0):
                    return True
                    
        # 检查是否是垂直方向的震荡
        if node[1] == recent_path[0][1] and abs(node[0] - recent_path[0][0]) <= 1:
            # 检查最近的移动是否也是垂直方向
            if recent_path[1][1] == recent_path[0][1] and recent_path[2][1] == recent_path[1][1]:
                # 检查是否是来回移动
                if ((node[0] - recent_path[0][0]) * (recent_path[1][0] - recent_path[0][0]) < 0 or
                    (recent_path[2][0] - recent_path[1][0]) * (recent_path[1][0] - recent_path[0][0]) < 0):
                    return True
                    
        return False
        
    def _has_line_of_sight(self, start, goal):
        """检查两点之间是否有直接视线（无障碍物）
        
        Args:
            start: 起点坐标
            goal: 终点坐标
            
        Returns:
            bool: 是否有直接视线
        """
        x0, y0 = start
        x1, y1 = goal
        
        # 使用Bresenham算法检查直线路径
        steep = abs(y1 - y0) > abs(x1 - x0)
        if steep:
            x0, y0 = y0, x0
            x1, y1 = y1, x1
            
        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0
            
        dx = x1 - x0
        dy = abs(y1 - y0)
        error = dx // 2
        y = y0
        
        if y0 < y1:
            y_step = 1
        else:
            y_step = -1
            
        for x in range(x0, x1 + 1):
            coord = (y, x) if steep else (x, y)
            # 检查该坐标是否在地图范围内且是否为障碍物
            if (0 <= coord[0] < self.map_size and 
                0 <= coord[1] < self.map_size and 
                self.obstacle_map[coord[0], coord[1]] == 1):
                return False
                
            error -= dy
            if error < 0:
                y += y_step
                error += dx
                
        return True
        
    def _get_cost(self, current, neighbor):
        """计算移动代价"""
        # 基础移动代价
        cost = 1.0
        
        # 未知区域额外代价（小幅减轻）
        if self.explored_map[neighbor] == -1:
            cost *= 1.1  # 原来是1.2，减轻对未知区域的惩罚
            
        # 障碍物密度影响（增强）
        density = self._calculate_obstacle_density(neighbor)
        cost *= (1 + 0.8 * density)  # 原来是1 + 0.5 * density，增加对障碍物密集区的避让
        
        # 添加转向惩罚（鼓励直线行走）
        if hasattr(self, 'current_path') and len(self.current_path) > 1:
            if len(self.current_path) >= 2 and current in self.current_path:
                # 找到当前位置在路径中的索引
                idx = self.current_path.index(current)
                if idx > 0:
                    # 计算当前移动方向
                    prev = self.current_path[idx-1]
                    curr_direction = (current[0] - prev[0], current[1] - prev[1])
                    
                    # 计算下一步移动方向
                    next_direction = (neighbor[0] - current[0], neighbor[1] - current[1])
                    
                    # 如果方向发生变化，增加惩罚
                    if curr_direction != next_direction:
                        cost *= 1.1  # 轻微惩罚转向，鼓励直线行走
        
        return cost
        
    def _should_replan(self, current, came_from):
        """检查是否需要重新规划路径"""
        if not came_from:
            return False
            
        # 检查当前节点周围是否有新发现的障碍物
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx, ny = current[0] + dx, current[1] + dy
                if 0 <= nx < self.map_size and 0 <= ny < self.map_size:
                    if self.explored_map[nx, ny] == 1:  # 新发现的障碍物
                        # 检查是否影响当前路径
                        path = self._reconstruct_path(came_from, current)
                        for node in path:
                            if self._is_affected_by_obstacle(node, (nx, ny)):
                                return True
                                
        return False
        
    def _replan_path(self, current, goal, closed_set):
        """重新规划路径"""
        # 清除部分closed_set，保留关键节点
        new_closed_set = {node for node in closed_set 
                         if self._calculate_obstacle_density(node) < 0.3}
        
        # 使用改进的A*重新规划
        return self.find_path(current, goal)
        
    def _get_neighbors(self, pos):
        """获取有效邻居节点"""
        x, y = pos
        neighbors = []
        
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.map_size and 0 <= ny < self.map_size:
                # 允许通过未知区域，但代价更高
                if self.explored_map[nx, ny] != 1:  # 不是障碍物
                    neighbors.append((nx, ny))
                    
        return neighbors
        
    def _is_valid_position(self, pos):
        """检查位置是否有效"""
        x, y = pos
        return (0 <= x < self.map_size and 
                0 <= y < self.map_size and 
                self.explored_map[x, y] != 1)
                
    def _reconstruct_path(self, came_from, current):
        """重建路径"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
        
    def update_explored_area(self, center_pos: Tuple[int, int], observation: np.ndarray, radius: int = 4):
        """更新已探索区域
        
        Args:
            center_pos: 观察中心位置
            observation: 观察数据
            radius: 观察半径
        """
        x, y = center_pos
        obs_size = 2 * radius + 1
        
        # 计算观察窗口的边界
        x_min = max(0, x - radius)
        x_max = min(self.map_size, x + radius + 1)
        y_min = max(0, y - radius)
        y_max = min(self.map_size, y + radius + 1)
        
        # 更新探索地图
        self.explored_map[x_min:x_max, y_min:y_max] = observation
        
        # 更新障碍物密度图
        for i in range(x_min, x_max):
            for j in range(y_min, y_max):
                if observation[i - x_min, j - y_min] == 1:  # 障碍物
                    self.obstacle_density_map[i, j] = self._calculate_obstacle_density((i, j))
        
    def get_exploration_target(self, current_pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """获取探索目标
        
        Args:
            current_pos: 当前位置
            
        Returns:
            Optional[Tuple[int, int]]: 探索目标位置
        """
        # 找到所有未探索区域的边界
        unexplored_borders = []
        for i in range(self.map_size):
            for j in range(self.map_size):
                if self.explored_map[i, j] == -1:  # 未探索
                    # 检查是否是边界（与已探索区域相邻）
                    for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        ni, nj = i + di, j + dj
                        if (0 <= ni < self.map_size and 0 <= nj < self.map_size and
                            self.explored_map[ni, nj] != -1):
                            unexplored_borders.append((i, j))
                            break
        
        if not unexplored_borders:
            return None
            
        # 选择最近的边界点
        current_x, current_y = current_pos
        distances = [abs(x - current_x) + abs(y - current_y) 
                    for x, y in unexplored_borders]
        min_idx = np.argmin(distances)
        return unexplored_borders[min_idx]
    
    def get_next_move(self, current_pos: Tuple[int, int], target_pos: Tuple[int, int]) -> Tuple[int, int]:
        """获取下一步移动位置
        
        Args:
            current_pos: 当前位置
            target_pos: 目标位置
            
        Returns:
            Tuple[int, int]: 下一步位置
        """
        # 如果当前路径为空或者目标改变，重新规划路径
        if not self.current_path or self.current_path[-1] != target_pos:
            self.current_path = self.find_path(current_pos, target_pos)
        
        # 如果找不到路径，尝试探索
        if not self.current_path:
            exploration_target = self.get_exploration_target(current_pos)
            if exploration_target:
                self.current_path = self.find_path(current_pos, exploration_target)
            
        # 返回下一步位置
        if len(self.current_path) > 1:
            return self.current_path[1]
        return current_pos  # 如果没有可行路径，保持原位
    
    def get_exploration_progress(self) -> float:
        """获取探索进度
        
        Returns:
            float: 探索完成百分比
        """
        explored_count = np.sum(self.explored_map != -1)
        total_cells = self.map_size * self.map_size
        return explored_count / total_cells * 100

    def generate_path_guidance_layer(self, start: Tuple[int, int], goal: Tuple[int, int]) -> np.ndarray:
        """生成从起点到目标的路径引导层
        
        Args:
            start: 起点位置
            goal: 目标位置
            
        Returns:
            np.ndarray: 路径引导层，形状为 (map_size, map_size, 2)，第一个通道是x方向引导，第二个通道是y方向引导
        """
        # 初始化引导层
        guidance_layer = np.zeros((self.map_size, self.map_size, 2))
        
        # 计算路径
        path = self.find_path(start, goal)
        
        if path is None or len(path) <= 1:
            return guidance_layer
            
        # 对于路径上的每个点，计算引导向量
        for i in range(len(path) - 1):
            current = path[i]
            next_pos = path[i + 1]
            
            # 计算方向向量
            direction = (next_pos[0] - current[0], next_pos[1] - current[1])
            
            # 归一化向量
            norm = max(abs(direction[0]), abs(direction[1]))
            if norm > 0:
                direction = (direction[0] / norm, direction[1] / norm)
                
            # 在当前位置记录引导向量
            guidance_layer[current[0], current[1], 0] = direction[0]  # x方向
            guidance_layer[current[0], current[1], 1] = direction[1]  # y方向
            
        # 对于目标点，指向自身
        guidance_layer[goal[0], goal[1], 0] = 0
        guidance_layer[goal[0], goal[1], 1] = 0
        
        # 扩散引导信息到周围区域
        temp_layer = np.copy(guidance_layer)
        for i in range(self.map_size):
            for j in range(self.map_size):
                if (i, j) not in path and self.explored_map[i, j] != 1:  # 不是路径点且不是障碍物
                    # 寻找最近的路径点
                    min_dist = float('inf')
                    closest_path_point = None
                    
                    for p in path:
                        dist = abs(p[0] - i) + abs(p[1] - j)
                        if dist < min_dist:
                            min_dist = dist
                            closest_path_point = p
                            
                    if closest_path_point and min_dist < 5:  # 只影响5格范围内的点
                        # 复制最近路径点的引导向量
                        temp_layer[i, j, 0] = guidance_layer[closest_path_point[0], closest_path_point[1], 0]
                        temp_layer[i, j, 1] = guidance_layer[closest_path_point[0], closest_path_point[1], 1]
                        
                        # 根据距离衰减引导强度
                        decay = max(0, 1 - min_dist / 5)
                        temp_layer[i, j, 0] *= decay
                        temp_layer[i, j, 1] *= decay
                        
        # 更新路径引导层
        self.path_guidance_layer = temp_layer
        return temp_layer
        
    def generate_global_guidance_layer(self, goal: Tuple[int, int]) -> np.ndarray:
        """生成全局引导层，从任意点到目标的最优方向
        
        Args:
            goal: 目标位置
            
        Returns:
            np.ndarray: 全局引导层，形状为 (map_size, map_size, 2)
        """
        # 初始化引导层
        guidance_layer = np.zeros((self.map_size, self.map_size, 2))
        
        # 为每个空地生成指向目标的最优路径
        for i in range(self.map_size):
            for j in range(self.map_size):
                if self.explored_map[i, j] != 1:  # 不是障碍物
                    path = self.find_path((i, j), goal)
                    
                    if path and len(path) > 1:
                        # 获取下一步位置
                        next_pos = path[1]  # path[0]是当前位置(i,j)
                        
                        # 计算方向向量
                        direction = (next_pos[0] - i, next_pos[1] - j)
                        
                        # 归一化向量
                        norm = max(abs(direction[0]), abs(direction[1]))
                        if norm > 0:
                            direction = (direction[0] / norm, direction[1] / norm)
                            
                        # 在当前位置记录引导向量
                        guidance_layer[i, j, 0] = direction[0]  # x方向
                        guidance_layer[i, j, 1] = direction[1]  # y方向
                        
        # 对于目标点，指向自身
        guidance_layer[goal[0], goal[1], 0] = 0
        guidance_layer[goal[0], goal[1], 1] = 0
        
        return guidance_layer
        
    def get_guidance_vector(self, pos: Tuple[int, int]) -> Tuple[float, float]:
        """获取指定位置的引导向量
        
        Args:
            pos: 位置
            
        Returns:
            Tuple[float, float]: 引导向量(x, y)
        """
        if 0 <= pos[0] < self.map_size and 0 <= pos[1] < self.map_size:
            return (self.path_guidance_layer[pos[0], pos[1], 0], 
                    self.path_guidance_layer[pos[0], pos[1], 1])
        return (0, 0)
        
    def find_path_to_task_goal(self, start: Tuple[int, int], task) -> List[Tuple[int, int]]:
        """为任务找到从起点到终点的完整路径
        
        Args:
            start: 起点位置
            task: 任务对象，包含start_pos和goal_pos
            
        Returns:
            List[Tuple[int, int]]: 完整路径
        """
        # 如果任务已经启动，直接寻找到终点的路径
        if hasattr(task, 'reached_start') and task.reached_start:
            # 增强到终点路径的计算
            # 生成新的A*引导层，增强对目标的导向
            self.generate_path_guidance_layer(start, task.goal_pos)
            return self.find_path(start, task.goal_pos)
        
        # 如果任务未启动，先找到任务起点
        path_to_start = self.find_path(start, task.start_pos)
        
        # 找不到到任务起点的路径
        if not path_to_start:
            return None
            
        # 然后找到从任务起点到终点的路径
        path_to_goal = self.find_path(task.start_pos, task.goal_pos)
        
        # 找不到到任务终点的路径
        if not path_to_goal:
            return path_to_start
            
        # 合并路径 (去掉重复的任务起点)
        complete_path = path_to_start + path_to_goal[1:]
        
        return complete_path