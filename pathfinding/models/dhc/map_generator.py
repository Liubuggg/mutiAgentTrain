import numpy as np
import random
from typing import List, Tuple, Optional

class MapGenerator:
    """地图生成器类
    
    负责生成结构化的地图，包括障碍物、房间和走廊
    """
    def __init__(self, map_size: int = 40, seed: Optional[int] = None):
        """初始化地图生成器
        
        Args:
            map_size: 地图大小
            seed: 随机种子
        """
        self.map_size = map_size
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
    def generate_structured_map(self, room_density: float = 0.3, corridor_width: int = 2) -> np.ndarray:
        """生成结构化地图
        
        Args:
            room_density: 房间密度
            corridor_width: 走廊宽度
            
        Returns:
            np.ndarray: 生成的地图，0表示可通行区域，1表示障碍物
        """
        # 初始化地图
        map_array = np.zeros((self.map_size, self.map_size), dtype=np.int32)
        
        # 生成房间
        room_size_range = (3, 6)  # 房间大小范围
        room_positions = self._generate_room_positions(room_density)
        
        for pos in room_positions:
            room_size = random.randint(*room_size_range)
            self._place_room(map_array, pos, room_size)
            
        # 生成走廊连接房间
        self._generate_corridors(map_array, room_positions, corridor_width)
        
        # 确保地图的可达性
        self._ensure_connectivity(map_array)
        
        return map_array
    
    def _generate_room_positions(self, density: float) -> List[Tuple[int, int]]:
        """生成房间位置
        
        Args:
            density: 房间密度
            
        Returns:
            List[Tuple[int, int]]: 房间位置列表
        """
        positions = []
        grid_size = 10  # 网格大小
        
        for i in range(0, self.map_size - grid_size, grid_size):
            for j in range(0, self.map_size - grid_size, grid_size):
                if random.random() < density:
                    # 在网格内随机选择位置
                    x = i + random.randint(2, grid_size-2)
                    y = j + random.randint(2, grid_size-2)
                    positions.append((x, y))
                    
        return positions
    
    def _place_room(self, map_array: np.ndarray, pos: Tuple[int, int], size: int):
        """在地图上放置房间
        
        Args:
            map_array: 地图数组
            pos: 房间位置
            size: 房间大小
        """
        x, y = pos
        x_start = max(0, x - size//2)
        x_end = min(self.map_size, x + size//2)
        y_start = max(0, y - size//2)
        y_end = min(self.map_size, y + size//2)
        
        # 放置房间墙壁
        map_array[x_start:x_end, y_start:y_end] = 1
        # 清空房间内部
        map_array[x_start+1:x_end-1, y_start+1:y_end-1] = 0
    
    def _generate_corridors(self, map_array: np.ndarray, room_positions: List[Tuple[int, int]], 
                          corridor_width: int):
        """生成走廊连接房间
        
        Args:
            map_array: 地图数组
            room_positions: 房间位置列表
            corridor_width: 走廊宽度
        """
        if len(room_positions) < 2:
            return
            
        # 使用最小生成树算法连接房间
        connected = {room_positions[0]}
        unconnected = set(room_positions[1:])
        
        while unconnected:
            min_dist = float('inf')
            best_pair = None
            
            for c_room in connected:
                for u_room in unconnected:
                    dist = abs(c_room[0] - u_room[0]) + abs(c_room[1] - u_room[1])
                    if dist < min_dist:
                        min_dist = dist
                        best_pair = (c_room, u_room)
            
            if best_pair:
                room1, room2 = best_pair
                self._create_corridor(map_array, room1, room2, corridor_width)
                connected.add(room2)
                unconnected.remove(room2)
    
    def _create_corridor(self, map_array: np.ndarray, start: Tuple[int, int], 
                        end: Tuple[int, int], width: int):
        """创建走廊
        
        Args:
            map_array: 地图数组
            start: 起点
            end: 终点
            width: 走廊宽度
        """
        x1, y1 = start
        x2, y2 = end
        
        # 先水平后垂直
        w = width // 2
        for x in range(min(x1, x2) - w, max(x1, x2) + w + 1):
            if 0 <= x < self.map_size:
                for dy in range(-w, w + 1):
                    y = y1 + dy
                    if 0 <= y < self.map_size:
                        map_array[x, y] = 0
                        
        for y in range(min(y1, y2) - w, max(y1, y2) + w + 1):
            if 0 <= y < self.map_size:
                for dx in range(-w, w + 1):
                    x = x2 + dx
                    if 0 <= x < self.map_size:
                        map_array[x, y] = 0
    
    def _ensure_connectivity(self, map_array: np.ndarray):
        """确保地图的可达性
        
        Args:
            map_array: 地图数组
        """
        # 使用广度优先搜索检查连通性
        def bfs(start):
            visited = np.zeros_like(map_array, dtype=bool)
            queue = [start]
            visited[start] = True
            while queue:
                x, y = queue.pop(0)
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < self.map_size and 0 <= ny < self.map_size and
                        not visited[nx, ny] and map_array[nx, ny] == 0):
                        queue.append((nx, ny))
                        visited[nx, ny] = True
            return visited
        
        # 找到第一个空地作为起点
        start = None
        for i in range(self.map_size):
            for j in range(self.map_size):
                if map_array[i, j] == 0:
                    start = (i, j)
                    break
            if start:
                break
                
        if start is None:
            return
            
        # 检查所有空地的可达性
        visited = bfs(start)
        for i in range(self.map_size):
            for j in range(self.map_size):
                if map_array[i, j] == 0 and not visited[i, j]:
                    # 如果发现不可达的空地，创建一条通道
                    self._create_path_to_unreachable(map_array, start, (i, j))
    
    def _create_path_to_unreachable(self, map_array: np.ndarray, start: Tuple[int, int], 
                                  end: Tuple[int, int]):
        """创建通向不可达区域的路径
        
        Args:
            map_array: 地图数组
            start: 起点
            end: 终点
        """
        x1, y1 = start
        x2, y2 = end
        
        # 创建一条直接的路径
        for x in range(min(x1, x2), max(x1, x2) + 1):
            map_array[x, y1] = 0
        for y in range(min(y1, y2), max(y1, y2) + 1):
            map_array[x2, y] = 0

    def is_valid_position(self, pos: Tuple[int, int], map_array: np.ndarray) -> bool:
        """检查位置是否有效（不在障碍物上且在地图范围内）
        
        Args:
            pos: 要检查的位置
            map_array: 地图数组
            
        Returns:
            bool: 位置是否有效
        """
        x, y = pos
        if not (0 <= x < self.map_size and 0 <= y < self.map_size):
            return False
        return map_array[x, y] == 0

    def find_valid_position(self, map_array: np.ndarray, min_distance: int = 2) -> Tuple[int, int]:
        """在地图上寻找一个有效位置
        
        Args:
            map_array: 地图数组
            min_distance: 与障碍物的最小距离
            
        Returns:
            Tuple[int, int]: 有效位置
        """
        valid_positions = []
        for x in range(self.map_size):
            for y in range(self.map_size):
                if map_array[x, y] == 0:
                    # 检查周围是否有障碍物
                    is_valid = True
                    for dx in range(-min_distance, min_distance + 1):
                        for dy in range(-min_distance, min_distance + 1):
                            nx, ny = x + dx, y + dy
                            if (0 <= nx < self.map_size and 0 <= ny < self.map_size and
                                map_array[nx, ny] == 1):
                                is_valid = False
                                break
                        if not is_valid:
                            break
                    if is_valid:
                        valid_positions.append((x, y))
        
        if valid_positions:
            return random.choice(valid_positions)
        return None

    def generate_charging_stations(self, map_array: np.ndarray, num_stations: int = 3) -> List[np.ndarray]:
        """生成充电站位置
        
        Args:
            map_array: 地图数组
            num_stations: 充电站数量
            
        Returns:
            List[np.ndarray]: 充电站位置列表
        """
        stations = []
        for _ in range(num_stations):
            pos = self.find_valid_position(map_array)
            if pos:
                stations.append(np.array(pos))
        return stations

    def generate_agent_positions(self, map_array: np.ndarray, num_agents: int) -> List[np.ndarray]:
        """生成智能体初始位置
        
        Args:
            map_array: 地图数组
            num_agents: 智能体数量
            
        Returns:
            List[np.ndarray]: 智能体位置列表
        """
        positions = []
        for _ in range(num_agents):
            pos = self.find_valid_position(map_array)
            if pos:
                positions.append(np.array(pos))
        return positions

    def generate_task_positions(self, map_array: np.ndarray, num_tasks: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """生成任务起点和终点位置
        
        Args:
            map_array: 地图数组
            num_tasks: 任务数量
            
        Returns:
            List[Tuple[np.ndarray, np.ndarray]]: 任务位置列表，每个元素为(起点, 终点)
        """
        task_positions = []
        for _ in range(num_tasks):
            start_pos = self.find_valid_position(map_array)
            if start_pos:
                # 确保终点与起点有足够距离
                end_pos = None
                attempts = 0
                while attempts < 10:
                    end_pos = self.find_valid_position(map_array)
                    if end_pos and np.linalg.norm(np.array(end_pos) - np.array(start_pos)) >= 5:
                        break
                    attempts += 1
                if end_pos:
                    task_positions.append((np.array(start_pos), np.array(end_pos)))
        return task_positions 