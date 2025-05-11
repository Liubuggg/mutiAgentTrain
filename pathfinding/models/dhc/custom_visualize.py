import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
import random
import time
import math
from matplotlib.patches import Rectangle, Circle
from typing import Dict, List, Tuple, Any, Optional
from collections import deque

# 设置样式
plt.style.use('ggplot')
COLORS = {
    'agent': 'blue',
    'charging_station': 'purple',
    'task_pending': 'red',
    'task_in_progress': 'orange',
    'task_completed': 'green',
    'task_failed': 'gray',
    'low_battery': 'darkred',
    'background': '#f8f8f8',
    'grid': '#dddddd',
    'agent_0': '#1f77b4',  # 蓝色
    'agent_1': '#ff7f0e',  # 橙色
    'agent_2': '#2ca02c',  # 绿色
    'agent_3': '#d62728',  # 红色
}

# 简化版任务状态枚举
class TaskStatus:
    PENDING = 'pending'
    IN_PROGRESS = 'in_progress'
    COMPLETED = 'completed'
    FAILED = 'failed'

class TaskPriority:
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4

# 简化版Agent类
class Agent:
    def __init__(self, id: int, pos: Tuple[int, int], map_size: int = 40, max_battery: float = 100.0):
        self.id = id
        self.pos = np.array(pos)
        self.map_size = map_size
        self.max_battery = max_battery
        self.current_battery = max_battery
        self.current_task = None
        self.state = {'charging': False, 'idle_time': 0}
        self.pathfinder = None

    def assign_task(self, task):
        """分配任务给智能体"""
        self.current_task = task
        task.status = TaskStatus.IN_PROGRESS
        task.assigned_agent_id = self.id
        return True

# 简化版Task类
class Task:
    def __init__(self, id: int, start_pos: Tuple[int, int], goal_pos: Tuple[int, int], 
                 priority=TaskPriority.MEDIUM, appear_time: int = 0):
        self.id = id
        self.start_pos = np.array(start_pos)
        self.goal_pos = np.array(goal_pos)
        self.priority = priority
        self.appear_time = appear_time
        self.status = TaskStatus.PENDING
        self.assigned_agent_id = None
        self.reached_start = False

# 简化版TaskAllocator类
class TaskAllocator:
    def __init__(self, num_agents: int, map_size: int):
        self.num_agents = num_agents
        self.map_size = map_size
        self.agents = {}
        self.tasks = {}
        self.charging_stations = []
        self.obstacle_map = None

    def set_obstacle_map(self, obstacle_map):
        """设置障碍物地图"""
        self.obstacle_map = obstacle_map

    def add_agent(self, agent):
        """添加智能体"""
        self.agents[agent.id] = agent

    def add_task(self, task):
        """添加任务"""
        self.tasks[task.id] = task

    def set_charging_stations(self, stations):
        """设置充电站位置"""
        self.charging_stations = stations

    def update(self):
        """更新状态"""
        pass

# 简化版PathFinder类
class PathFinder:
    def __init__(self, map_size: int):
        self.map_size = map_size
        self.obstacle_map = None
        self.explored_area = np.ones((map_size, map_size), dtype=bool)

    def set_obstacle_map(self, obstacle_map):
        """设置障碍物地图"""
        self.obstacle_map = obstacle_map

def setup_environment(seed: int = 42):
    """设置环境和随机种子
    
    Args:
        seed: 随机种子
    """
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    
    # 创建输出目录
    os.makedirs("visualizations", exist_ok=True)

def generate_obstacle_map(map_size: int, obstacle_density: float = 0.2, seed: int = 42):
    """生成障碍物地图，确保有足够的通道
    
    Args:
        map_size: 地图大小
        obstacle_density: 障碍物密度
        seed: 随机种子
        
    Returns:
        np.ndarray: 障碍物地图
    """
    np.random.seed(seed)
    
    # 初始化空地图
    obstacle_map = np.zeros((map_size, map_size), dtype=np.int8)
    
    # 计算障碍物总数
    num_obstacles = int(map_size * map_size * obstacle_density)
    
    # 记录充电站位置，确保周围没有障碍物
    charging_stations = [(35, 30), (5, 15)]
    station_radius = 2
    
    # 确保地图中心区域没有障碍物，方便智能体移动
    center_x, center_y = map_size // 2, map_size // 2
    center_radius = map_size // 10
    
    # 方法1：随机放置单个障碍物（30%）
    individual_obstacles = int(num_obstacles * 0.3)
    placed_obstacles = 0
    
    while placed_obstacles < individual_obstacles:
        x = np.random.randint(0, map_size)
        y = np.random.randint(0, map_size)
        
        # 检查是否在中心区域
        dist_to_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        if dist_to_center <= center_radius:
            continue
            
        # 检查是否靠近充电站
        too_close_to_station = False
        for station in charging_stations:
            if np.sqrt((x - station[0])**2 + (y - station[1])**2) <= station_radius:
                too_close_to_station = True
                break
                
        if too_close_to_station:
            continue
            
        # 放置障碍物并确保周围至少有3个方向可通行
        if obstacle_map[x, y] == 0:
            free_directions = 4  # 默认四个方向都是自由的
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < map_size and 0 <= ny < map_size and obstacle_map[nx, ny] == 1:
                    free_directions -= 1
            
            if free_directions >= 3:  # 至少保留3个方向通行
                obstacle_map[x, y] = 1
                placed_obstacles += 1
    
    # 方法2：放置小型障碍物集群（70%）
    cluster_obstacles = num_obstacles - placed_obstacles
    num_clusters = cluster_obstacles // 3  # 每个集群平均3个障碍物
    
    for _ in range(num_clusters):
        # 随机选择集群中心
        center_x = np.random.randint(1, map_size-1)
        center_y = np.random.randint(1, map_size-1)
        
        # 检查是否靠近充电站
        too_close_to_station = False
        for station in charging_stations:
            if np.sqrt((center_x - station[0])**2 + (center_y - station[1])**2) <= station_radius + 1:
                too_close_to_station = True
                break
                
        if too_close_to_station:
            continue
        
        # 在中心周围随机放置2-4个障碍物
        cluster_size = np.random.randint(2, 5)
        for _ in range(cluster_size):
            # 在中心周围2x2区域内随机选择位置
            dx = np.random.randint(-2, 3)
            dy = np.random.randint(-2, 3)
            x = np.clip(center_x + dx, 0, map_size-1)
            y = np.clip(center_y + dy, 0, map_size-1)
            
            # 检查是否已经是障碍物
            if obstacle_map[x, y] == 1:
                continue
                
            # 检查是否靠近充电站
            too_close_to_station = False
            for station in charging_stations:
                if np.sqrt((x - station[0])**2 + (y - station[1])**2) <= station_radius:
                    too_close_to_station = True
                    break
                    
            if too_close_to_station:
                continue
            
            # 检查周围通行性
            free_directions = 4
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < map_size and 0 <= ny < map_size and obstacle_map[nx, ny] == 1:
                    free_directions -= 1
            
            # 确保至少有2个方向可通行
            if free_directions >= 2:
                obstacle_map[x, y] = 1
                placed_obstacles += 1
                
            # 如果已经放置足够多的障碍物，就停止
            if placed_obstacles >= num_obstacles:
                break
                
        # 如果已经放置足够多的障碍物，就停止
        if placed_obstacles >= num_obstacles:
            break
    
    # 确保地图边缘有一圈空地，避免封闭区域
    obstacle_map[0, :] = 0
    obstacle_map[:, 0] = 0
    obstacle_map[-1, :] = 0
    obstacle_map[:, -1] = 0
    
    # 确保充电站周围没有障碍物
    for station in charging_stations:
        x, y = station
        for dx in range(-station_radius, station_radius + 1):
            for dy in range(-station_radius, station_radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < map_size and 0 <= ny < map_size:
                    obstacle_map[nx, ny] = 0
    
    return obstacle_map

def get_valid_positions(obstacle_map, num_positions):
    """获取有效位置
    
    Args:
        obstacle_map: 障碍物地图
        num_positions: 需要的位置数量
        
    Returns:
        List: 有效位置列表
    """
    map_size = obstacle_map.shape[0]
    valid_positions = []
    
    for x in range(map_size):
        for y in range(map_size):
            if obstacle_map[x, y] == 0:
                valid_positions.append((x, y))
    
    # 随机打乱有效位置
    random.shuffle(valid_positions)
    
    return valid_positions[:num_positions]

def get_safe_positions(obstacle_map, num_positions, charging_stations=None, min_distance_to_obstacle=1):
    """获取安全的位置，确保位置远离障碍物
    
    Args:
        obstacle_map: 障碍物地图
        num_positions: 需要的位置数量
        charging_stations: 充电站位置列表
        min_distance_to_obstacle: 与障碍物的最小距离
        
    Returns:
        List: 安全位置列表
    """
    if charging_stations is None:
        charging_stations = []
        
    map_size = obstacle_map.shape[0]
    
    # 收集所有非障碍物位置
    valid_positions = []
    for x in range(map_size):
        for y in range(map_size):
            # 检查位置是否安全（不是障碍物）
            if obstacle_map[x, y] == 0:
                # 检查是否与障碍物保持最小距离
                is_safe = True
                if min_distance_to_obstacle > 0:
                    for dx in range(-min_distance_to_obstacle, min_distance_to_obstacle + 1):
                        for dy in range(-min_distance_to_obstacle, min_distance_to_obstacle + 1):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < map_size and 0 <= ny < map_size and obstacle_map[nx, ny] == 1:
                                is_safe = False
                                break
                        if not is_safe:
                            break
                
                # 检查是否与充电站太近
                too_close_to_station = False
                for station in charging_stations:
                    if np.linalg.norm(np.array([x, y]) - station) < 2:
                        too_close_to_station = True
                        break
                
                if is_safe and not too_close_to_station:
                    valid_positions.append((x, y))
    
    # 如果找不到足够的位置，放宽条件
    if len(valid_positions) < num_positions and min_distance_to_obstacle > 0:
        return get_safe_positions(obstacle_map, num_positions, charging_stations, 0)
    
    # 如果还是找不到足够的位置，就使用所有非障碍物位置
    if len(valid_positions) < num_positions:
        valid_positions = []
        for x in range(map_size):
            for y in range(map_size):
                if obstacle_map[x, y] == 0:
                    valid_positions.append((x, y))
    
    # 随机打乱位置
    random.shuffle(valid_positions)
    
    # 尝试分散选择位置，确保位置之间有一定距离
    selected_positions = []
    
    # 先选择四个角落附近的位置作为初始位置，确保分布均匀
    corners = [(5, 5), (5, map_size-6), (map_size-6, 5), (map_size-6, map_size-6)]
    for corner in corners:
        if len(selected_positions) >= num_positions:
            break
            
        # 找到离角落最近的有效位置
        closest_pos = None
        closest_dist = float('inf')
        for pos in valid_positions:
            dist = np.sqrt((pos[0] - corner[0])**2 + (pos[1] - corner[1])**2)
            if dist < closest_dist:
                closest_dist = dist
                closest_pos = pos
        
        if closest_pos and closest_pos not in selected_positions:
            selected_positions.append(closest_pos)
            valid_positions.remove(closest_pos)
    
    # 然后添加剩余的位置，确保与已选位置有最小距离
    min_distance_between_positions = 5  # 位置之间的最小距离
    
    while len(selected_positions) < num_positions and valid_positions:
        best_pos = None
        max_min_dist = -1
        
        for pos in valid_positions:
            if not selected_positions:
                best_pos = pos
                break
                
            # 计算与所有已选位置的最小距离
            min_dist = float('inf')
            for sel_pos in selected_positions:
                dist = np.sqrt((pos[0] - sel_pos[0])**2 + (pos[1] - sel_pos[1])**2)
                min_dist = min(min_dist, dist)
            
            # 选择与已有位置距离最大的位置
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                best_pos = pos
        
        if best_pos:
            selected_positions.append(best_pos)
            valid_positions.remove(best_pos)
        else:
            break
        
        # 如果无法找到满足最小距离的位置，则减小最小距离要求
        if max_min_dist < min_distance_between_positions and min_distance_between_positions > 1:
            min_distance_between_positions -= 1
    
    # 如果还是不够，就使用剩余的有效位置
    while len(selected_positions) < num_positions and valid_positions:
        selected_positions.append(valid_positions.pop())
    
    return selected_positions

def has_path(start, end, obstacle_map):
    """检查两点之间是否有路径
    
    Args:
        start: 起点
        end: 终点
        obstacle_map: 障碍物地图
        
    Returns:
        bool: 是否有路径
    """
    map_size = obstacle_map.shape[0]
    queue = deque([start])
    visited = {start}
    
    while queue:
        current = queue.popleft()
        
        if current == end:
            return True
        
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = current[0] + dx, current[1] + dy
            next_pos = (nx, ny)
            
            if (0 <= nx < map_size and 0 <= ny < map_size and 
                obstacle_map[nx, ny] == 0 and next_pos not in visited):
                
                queue.append(next_pos)
                visited.add(next_pos)
    
    return False

def create_scenario(
    num_agents: int = 4,
    map_size: int = 40,
    num_tasks: int = 4,
    obstacle_density: float = 0.15,
    seed: int = 42
):
    """创建场景
    
    Args:
        num_agents: 智能体数量
        map_size: 地图大小
        num_tasks: 任务数量
        obstacle_density: 障碍物密度
        seed: 随机种子
        
    Returns:
        Tuple: (任务分配器, 智能体列表, 任务列表)
    """
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    
    # 生成障碍物地图
    obstacle_map = generate_obstacle_map(map_size, obstacle_density, seed)
    
    # 创建任务分配器
    task_allocator = TaskAllocator(num_agents=num_agents, map_size=map_size)
    task_allocator.set_obstacle_map(obstacle_map)
    
    # 创建充电站
    charging_stations = []
    
    # 第一个充电站 - 放在坐标(35, 30)
    station1_pos = (35, 30)
    # 确保充电站不在障碍物上
    if obstacle_map[station1_pos[0], station1_pos[1]] == 1:
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                new_x, new_y = station1_pos[0] + dx, station1_pos[1] + dy
                if (0 <= new_x < map_size and 0 <= new_y < map_size and 
                    obstacle_map[new_x, new_y] == 0):
                    station1_pos = (new_x, new_y)
                    break
            else:
                continue
            break
    
    # 第二个充电站 - 放在坐标(5, 15)
    station2_pos = (5, 15)
    # 确保充电站不在障碍物上
    if obstacle_map[station2_pos[0], station2_pos[1]] == 1:
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                new_x, new_y = station2_pos[0] + dx, station2_pos[1] + dy
                if (0 <= new_x < map_size and 0 <= new_y < map_size and 
                    obstacle_map[new_x, new_y] == 0):
                    station2_pos = (new_x, new_y)
                    break
            else:
                continue
            break
    
    charging_stations = [np.array(station1_pos), np.array(station2_pos)]
    task_allocator.set_charging_stations(charging_stations)
    
    # 获取安全位置
    num_positions_needed = num_agents + num_tasks * 2
    safe_positions = get_safe_positions(obstacle_map, num_positions_needed, charging_stations)
    
    if len(safe_positions) < num_positions_needed:
        print(f"警告：无法找到足够的安全位置，只找到了 {len(safe_positions)} 个，需要 {num_positions_needed} 个")
    
    # 随机打乱位置
    random.shuffle(safe_positions)
    
    # 创建智能体
    agents = []
    for i in range(num_agents):
        if i < len(safe_positions):
            pos = safe_positions.pop()
        else:
            # 找一个非障碍物位置
            while True:
                x = random.randint(0, map_size - 1)
                y = random.randint(0, map_size - 1)
                if obstacle_map[x, y] == 0:
                    pos = (x, y)
                    break
        
        agent = Agent(id=i, pos=pos, map_size=map_size)
        agent.pathfinder = PathFinder(map_size)
        agent.pathfinder.set_obstacle_map(obstacle_map)
        task_allocator.add_agent(agent)
        agents.append(agent)
    
    # 创建任务
    tasks = []
    for i in range(num_tasks):
        if len(safe_positions) >= 2:
            start_pos = safe_positions.pop()
            goal_pos = safe_positions.pop()
            
            # 确保起点和终点之间有路径
            if not has_path(start_pos, goal_pos, obstacle_map):
                print(f"警告：任务 {i} 的起点和终点之间没有路径，尝试重新选择位置")
                safe_positions.append(start_pos)  # 把不可用的位置放回去
                safe_positions.append(goal_pos)
                
                # 尝试找到有路径的起点和终点
                found_path = False
                for s_idx, start_candidate in enumerate(safe_positions):
                    for g_idx, goal_candidate in enumerate(safe_positions):
                        if s_idx != g_idx and has_path(start_candidate, goal_candidate, obstacle_map):
                            start_pos = start_candidate
                            goal_pos = goal_candidate
                            
                            # 从列表中移除这两个位置
                            if s_idx > g_idx:
                                safe_positions.pop(s_idx)
                                safe_positions.pop(g_idx)
                            else:
                                safe_positions.pop(g_idx)
                                safe_positions.pop(s_idx)
                                
                            found_path = True
                            break
                    if found_path:
                        break
                
                if not found_path:
                    # 如果找不到有路径的起点和终点，随机选择
                    print(f"无法找到有路径的起点和终点，使用随机位置")
                    while True:
                        x1 = random.randint(0, map_size - 1)
                        y1 = random.randint(0, map_size - 1)
                        if obstacle_map[x1, y1] == 0:
                            start_pos = (x1, y1)
                            break
                    
                    while True:
                        x2 = random.randint(0, map_size - 1)
                        y2 = random.randint(0, map_size - 1)
                        if obstacle_map[x2, y2] == 0 and (x2, y2) != start_pos:
                            goal_pos = (x2, y2)
                            if has_path(start_pos, goal_pos, obstacle_map):
                                break
        else:
            # 找两个非障碍物位置，确保它们之间有路径
            attempts = 0
            while attempts < 100:  # 最多尝试100次
                x1 = random.randint(0, map_size - 1)
                y1 = random.randint(0, map_size - 1)
                if obstacle_map[x1, y1] == 0:
                    start_pos = (x1, y1)
                    break
                attempts += 1
            
            attempts = 0
            while attempts < 100:  # 最多尝试100次
                x2 = random.randint(0, map_size - 1)
                y2 = random.randint(0, map_size - 1)
                if obstacle_map[x2, y2] == 0 and (x2, y2) != start_pos:
                    goal_pos = (x2, y2)
                    if has_path(start_pos, goal_pos, obstacle_map):
                        break
                attempts += 1
                
                # 如果100次都找不到，放弃路径检查
                if attempts >= 100:
                    print(f"警告：无法找到有路径的位置对，使用随机位置")
                    while True:
                        x2 = random.randint(0, map_size - 1)
                        y2 = random.randint(0, map_size - 1)
                        if obstacle_map[x2, y2] == 0 and (x2, y2) != start_pos:
                            goal_pos = (x2, y2)
                            break
                    break
        
        task = Task(
            id=i,
            start_pos=start_pos,
            goal_pos=goal_pos,
            priority=TaskPriority.MEDIUM,
            appear_time=0
        )
        task_allocator.add_task(task)
        tasks.append(task)
    
    return task_allocator, agents, tasks

def render_frame(
    task_allocator: TaskAllocator,
    agents: List[Agent],
    tasks: List[Task],
    step: int,
    ax: plt.Axes,
    title: Optional[str] = None,
    show_paths: bool = True
) -> List:
    """渲染单帧场景
    
    Args:
        task_allocator: 任务分配器
        agents: 智能体列表
        tasks: 任务列表
        step: 当前步数
        ax: matplotlib轴
        title: 标题
        show_paths: 是否显示路径
        
    Returns:
        List: 艺术家对象列表
    """
    artists = []
    
    # 清除轴
    ax.clear()
    
    # 设置边界和浅灰色背景
    ax.set_xlim(-1, task_allocator.map_size + 1)
    ax.set_ylim(-1, task_allocator.map_size + 1)
    ax.set_facecolor('#E5E5E5')  # 设置背景为浅灰色
    
    # 绘制障碍物地图 - 障碍物使用深灰色方块填充单元格
    if hasattr(task_allocator, 'obstacle_map') and task_allocator.obstacle_map is not None:
        # 为每个障碍物创建一个矩形
        for x in range(task_allocator.map_size):
            for y in range(task_allocator.map_size):
                if task_allocator.obstacle_map[x, y] == 1:
                    # 创建一个填充整个单元格的矩形
                    rect = plt.Rectangle(
                        (x - 0.5, y - 0.5),  # 左下角坐标
                        1.0,  # 宽度为1个单位
                        1.0,  # 高度为1个单位
                        facecolor='#404040',  # 深灰色填充
                        edgecolor='#404040',  # 深灰色边框
                        alpha=1.0,  # 完全不透明
                        zorder=1  # 确保在网格下方
                    )
                    ax.add_patch(rect)
                    artists.append(rect)
    
    # 添加网格
    ax.grid(True, color='#CCCCCC', linestyle='-', linewidth=0.5)
    
    # 设置标题
    if title:
        title_obj = ax.set_title(title, fontsize=14)
    else:
        completed_tasks = sum(1 for task in tasks if task.status == TaskStatus.COMPLETED)
        title_obj = ax.set_title(f'Step: {step} | Completed Tasks: {completed_tasks}/{len(tasks)}', fontsize=14)
    artists.append(title_obj)
    
    # 绘制充电站
    for station in task_allocator.charging_stations:
        x, y = station
        station_marker = ax.scatter(x, y, c=COLORS['charging_station'], marker='^', s=200, zorder=3)
        artists.append(station_marker)
        
        # 添加充电站标签
        station_text = ax.text(x, y - 1.5, 'Charging Station', ha='center', fontsize=8, color=COLORS['charging_station'])
        artists.append(station_text)
    
    # 绘制任务
    for task in task_allocator.tasks.values():
        start_x, start_y = task.start_pos
        goal_x, goal_y = task.goal_pos
        
        # 根据任务状态选择颜色和标签
        if task.status == TaskStatus.COMPLETED:
            color = COLORS['task_completed']
            status_text = 'Completed'
        elif task.status == TaskStatus.IN_PROGRESS:
            color = COLORS['task_in_progress']
            status_text = 'In Progress'
        elif task.status == TaskStatus.FAILED:
            color = COLORS['task_failed']
            status_text = 'Failed'
        else:
            color = COLORS['task_pending']
            status_text = 'Pending'
            
        # 绘制起点和终点
        start_marker = ax.scatter(start_x, start_y, c=color, marker='s', s=120, zorder=2)
        goal_marker = ax.scatter(goal_x, goal_y, c=color, marker='*', s=180, zorder=2)
        
        # 绘制任务路径
        path = ax.plot([start_x, goal_x], [start_y, goal_y], c=color, linestyle='--', linewidth=2, alpha=0.5, zorder=1)[0]
        
        # 添加任务标签
        start_text = ax.text(start_x, start_y - 1.5, f'Task{task.id} Start', ha='center', fontsize=8, color=color)
        goal_text = ax.text(goal_x, goal_y + 1.5, f'Task{task.id} Goal\n{status_text}', ha='center', fontsize=8, color=color)
        
        artists.extend([start_marker, goal_marker, path, start_text, goal_text])
    
    # 绘制智能体
    for agent in agents:
        x, y = agent.pos
        
        # 根据电量选择颜色和状态
        battery_percentage = agent.current_battery / agent.max_battery
        agent_color = COLORS[f'agent_{agent.id}']
        
        if battery_percentage < 0.2:
            battery_status = 'Low Battery'
            # 闪烁效果 - 半透明
            agent_alpha = 0.5 + 0.5 * (np.sin(step * 0.5) * 0.5 + 0.5)
        else:
            battery_status = 'Normal'
            agent_alpha = 1.0
        
        # 绘制智能体
        agent_marker = ax.scatter(x, y, c=agent_color, s=150, zorder=4, alpha=agent_alpha)
        
        # 添加智能体状态信息
        status_text = f'Agent{agent.id}\n{battery_status}\n{int(battery_percentage*100)}%'
        if agent.current_task:
            status_text += f'\nTask{agent.current_task.id}'
        
        agent_text = ax.text(
            x, y + 0.5,
            status_text,
            ha='center', va='bottom', fontsize=8,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor=agent_color)
        )
        artists.extend([agent_marker, agent_text])
        
        # 如果智能体正在执行任务且显示路径，显示任务路径
        if show_paths and agent.current_task:
            task = agent.current_task
            if not hasattr(task, 'reached_start') or not task.reached_start:
                # 显示到任务起点的路径
                start_line = ax.plot(
                    [x, task.start_pos[0]], [y, task.start_pos[1]],
                    c=agent_color, linestyle='-', linewidth=2, alpha=0.5, zorder=1
                )[0]
                artists.append(start_line)
            else:
                # 显示到任务终点的路径
                goal_line = ax.plot(
                    [x, task.goal_pos[0]], [y, task.goal_pos[1]],
                    c=agent_color, linestyle='-', linewidth=2, alpha=0.5, zorder=1
                )[0]
                artists.append(goal_line)
    
    return artists

def create_animation(
    task_allocator: TaskAllocator,
    agents: List[Agent],
    tasks: List[Task],
    movement_plans: List[List[Tuple[int, int]]],
    max_steps: int = 100,
    output_path: str = 'visualizations/custom_simulation.gif',
    fps: int = 5
):
    """创建动画
    
    Args:
        task_allocator: 任务分配器
        agents: 智能体列表
        tasks: 任务列表
        movement_plans: 智能体移动计划
        max_steps: 最大步数
        output_path: 输出路径
        fps: 帧率
    """
    # 创建图形和轴
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # 场景状态 - 用于在帧函数中更新
    state = {
        'task_allocator': task_allocator,
        'agents': agents,
        'tasks': tasks,
        'step': 0,
        'movement_plans': movement_plans,
        'completed_tasks': 0,
        'animation_finished': False
    }
    
    def update_frame(frame, *fargs):
        """更新帧"""
        if state['animation_finished']:
            return []
            
        # 更新智能体位置
        update_scene(state)
        
        # 渲染场景
        state['step'] = frame
        
        # 检查是否所有任务都已完成
        completed_tasks = sum(1 for task in state['task_allocator'].tasks.values() 
                         if task.status == TaskStatus.COMPLETED)
        
        # 更新标题，显示完成进度
        title = f'Step: {frame} | Completed Tasks: {completed_tasks}/{len(tasks)}'
        
        # 如果所有任务完成，结束动画
        if completed_tasks == len(tasks) and frame > 10:
            title += " | All tasks completed!"
            # 允许多运行几帧后结束
            if frame > max_steps - 10:
                state['animation_finished'] = True
        
        return render_frame(
            state['task_allocator'],
            state['agents'],
            state['tasks'],
            frame,
            ax,
            title=title
        )
    
    # 创建动画
    ani = animation.FuncAnimation(
        fig, update_frame, frames=max_steps,
        interval=1000/fps, blit=True
    )
    
    # 保存动画
    print(f"Saving animation to {output_path}...")
    if output_path.endswith('.gif'):
        ani.save(output_path, writer='pillow', fps=fps)
    else:
        ani.save(output_path, writer='ffmpeg', fps=fps)
    
    print(f"Animation saved to {output_path}")
    plt.close(fig)
    
def update_scene(state):
    """更新场景状态，确保智能体永远不会移动到障碍物上
    
    Args:
        state: 场景状态
    """
    step = state['step']
    agents = state['agents']
    tasks = state['tasks']
    task_allocator = state['task_allocator']
    movement_plans = state['movement_plans']
    obstacle_map = task_allocator.obstacle_map
    
    # 调试信息 - 打印障碍物位置
    if step == 0:
        obstacle_positions = []
        for x in range(obstacle_map.shape[0]):
            for y in range(obstacle_map.shape[1]):
                if obstacle_map[x, y] == 1:
                    obstacle_positions.append((x, y))
        print(f"障碍物位置(第一个10个): {obstacle_positions[:10]}...")
    
    # 检查智能体初始位置，确保它们不在障碍物上
    for i, agent in enumerate(agents):
        agent_pos_int = tuple(map(int, agent.pos))
        if (0 <= agent_pos_int[0] < obstacle_map.shape[0] and 
            0 <= agent_pos_int[1] < obstacle_map.shape[1] and 
            obstacle_map[agent_pos_int[0], agent_pos_int[1]] == 1):
            print(f"严重错误：智能体{i}位于障碍物上 {agent_pos_int}，寻找安全位置")
            
            # 寻找安全位置
            safe_pos = None
            for dist in range(1, 10):
                for dx in range(-dist, dist+1):
                    for dy in range(-dist, dist+1):
                        nx, ny = agent_pos_int[0] + dx, agent_pos_int[1] + dy
                        if (0 <= nx < obstacle_map.shape[0] and 
                            0 <= ny < obstacle_map.shape[1] and 
                            obstacle_map[nx, ny] == 0):
                            safe_pos = (nx, ny)
                            break
                    if safe_pos:
                        break
                if safe_pos:
                    break
            
            if safe_pos:
                print(f"  将智能体{i}从障碍物位置 {agent_pos_int} 移动到安全位置 {safe_pos}")
                agent.pos = np.array(safe_pos)
            else:
                # 随机找一个非障碍物位置
                valid_positions = []
                for x in range(obstacle_map.shape[0]):
                    for y in range(obstacle_map.shape[1]):
                        if obstacle_map[x, y] == 0:
                            valid_positions.append((x, y))
                
                if valid_positions:
                    random_pos = random.choice(valid_positions)
                    print(f"  将智能体{i}从障碍物位置 {agent_pos_int} 随机移动到安全位置 {random_pos}")
                    agent.pos = np.array(random_pos)
    
    # 更新智能体位置
    for i, agent in enumerate(agents):
        if i < len(movement_plans) and step < len(movement_plans[i]):
            # 获取计划中的下一个位置
            next_pos = movement_plans[i][step]
            
            # 确保位置格式正确
            next_pos = tuple(map(int, next_pos))
            
            # 安全检查 - 确保目标位置不是障碍物
            is_safe = True
            
            # 检查边界
            if not (0 <= next_pos[0] < obstacle_map.shape[0] and 0 <= next_pos[1] < obstacle_map.shape[1]):
                print(f"边界错误: 智能体{i}计划移动到边界外 {next_pos}")
                is_safe = False
            
            # 检查障碍物 - 这是最关键的检查
            elif obstacle_map[next_pos[0], next_pos[1]] == 1:
                print(f"障碍物错误: 智能体{i}计划移动到障碍物位置 {next_pos}")
                is_safe = False
            
            # 检查智能体碰撞
            else:
                for j, other_agent in enumerate(agents):
                    if i != j and tuple(map(int, other_agent.pos)) == next_pos:
                        print(f"碰撞错误: 智能体{i}计划移动到智能体{j}的位置 {next_pos}")
                        is_safe = False
                        break
            
            # 移动距离检查
            current_pos = tuple(map(int, agent.pos))
            distance = math.sqrt((next_pos[0] - current_pos[0])**2 + (next_pos[1] - current_pos[1])**2)
            
            if distance > 1.5:  # 最大距离应该是sqrt(2)≈1.414
                print(f"距离错误: 智能体{i}计划移动距离过大 ({distance:.2f}) 从 {current_pos} 到 {next_pos}")
                is_safe = False
            
            # 仅当安全时移动
            if is_safe:
                # 最后再次确认目标位置不是障碍物
                if obstacle_map[next_pos[0], next_pos[1]] == 1:
                    print(f"最终障碍物检查失败: 智能体{i}不应移动到障碍物位置 {next_pos}")
                    is_safe = False
            
            if is_safe:
                # 更新电量
                energy_cost = distance * 0.1
                agent.current_battery = max(0, agent.current_battery - energy_cost)
                
                # 移动智能体
                agent.pos = np.array(next_pos)
            else:
                # 如果不安全，保持原位置
                print(f"安全检查失败: 智能体{i}保持在原位置 {current_pos}")
                
                # 确保原位置也是安全的
                if obstacle_map[current_pos[0], current_pos[1]] == 1:
                    print(f"原位置不安全: 智能体{i}当前位于障碍物上 {current_pos}")
                    
                    # 寻找附近安全位置
                    for dist in range(1, 5):
                        found = False
                        for dx in range(-dist, dist+1):
                            for dy in range(-dist, dist+1):
                                nx, ny = current_pos[0] + dx, current_pos[1] + dy
                                if (0 <= nx < obstacle_map.shape[0] and 
                                    0 <= ny < obstacle_map.shape[1] and 
                                    obstacle_map[nx, ny] == 0):
                                    agent.pos = np.array([nx, ny])
                                    print(f"  紧急修正: 智能体{i}移动到安全位置 {(nx, ny)}")
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            break
        
        # 最终检查 - 确保智能体不在障碍物上
        final_pos = tuple(map(int, agent.pos))
        if (0 <= final_pos[0] < obstacle_map.shape[0] and 
            0 <= final_pos[1] < obstacle_map.shape[1] and 
            obstacle_map[final_pos[0], final_pos[1]] == 1):
            print(f"严重错误: 步骤{step}后，智能体{i}仍在障碍物上 {final_pos}")
            
            # 紧急修正 - 移动到附近安全位置
            for dist in range(1, 10):
                found = False
                for dx in range(-dist, dist+1):
                    for dy in range(-dist, dist+1):
                        nx, ny = final_pos[0] + dx, final_pos[1] + dy
                        if (0 <= nx < obstacle_map.shape[0] and 
                            0 <= ny < obstacle_map.shape[1] and 
                            obstacle_map[nx, ny] == 0):
                            print(f"  最终修正: 将智能体{i}从障碍物位置 {final_pos} 移动到安全位置 {(nx, ny)}")
                            agent.pos = np.array([nx, ny])
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
        
        # 更新充电状态
        for station in task_allocator.charging_stations:
            station_pos = tuple(map(int, station))
            agent_pos = tuple(map(int, agent.pos))
            distance_to_station = np.linalg.norm(np.array(station_pos) - np.array(agent_pos))
            if distance_to_station <= 1.5 and agent.current_battery < agent.max_battery * 0.8:
                agent.current_battery = min(agent.max_battery, agent.current_battery + 2.0)
                agent.state['charging'] = True
                break
        else:
            agent.state['charging'] = False
            
        # 任务状态检查和更新
        if agent.current_task:
            task = agent.current_task
            
            # 检查是否到达起点
            start_pos = tuple(map(int, task.start_pos))
            agent_pos = tuple(map(int, agent.pos))
            distance_to_start = np.linalg.norm(np.array(start_pos) - np.array(agent_pos))
            if distance_to_start <= 1.5:
                task.reached_start = True
                
            # 检查是否到达终点（且已到达起点）
            if hasattr(task, 'reached_start') and task.reached_start:
                goal_pos = tuple(map(int, task.goal_pos))
                agent_pos = tuple(map(int, agent.pos))
                distance_to_goal = np.linalg.norm(np.array(goal_pos) - np.array(agent_pos))
                if distance_to_goal <= 1.5:
                    task.status = TaskStatus.COMPLETED
                    state['completed_tasks'] = state.get('completed_tasks', 0) + 1
                    agent.current_task = None
    
    # 任务分配
    for agent in agents:
        if agent.current_task is None and not agent.state.get('charging', False):
            available_tasks = [task for task in task_allocator.tasks.values() 
                           if task.status == TaskStatus.PENDING]
            
            if available_tasks:
                nearest_task = min(available_tasks, 
                               key=lambda t: np.linalg.norm(agent.pos - t.start_pos))
                agent.assign_task(nearest_task)

def is_adjacent_to_obstacle(pos, obstacle_map):
    """检查位置是否与障碍物相邻
    
    Args:
        pos: 要检查的位置
        obstacle_map: 障碍物地图
    
    Returns:
        bool: 是否与障碍物相邻
    """
    map_size = obstacle_map.shape[0]
    x, y = pos
    
    # 检查八个方向
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue  # 跳过当前位置
                
            nx, ny = x + dx, y + dy
            if 0 <= nx < map_size and 0 <= ny < map_size:
                if obstacle_map[nx, ny] == 1:
                    return True
    
    return False

def plan_agent_movements(
    task_allocator: TaskAllocator,
    agents: List[Agent],
    tasks: List[Task],
    max_steps: int = 100,
    seed: int = 42
) -> List[List[Tuple[int, int]]]:
    """规划智能体移动路径
    
    规划内容包括：
    1. 智能体移动到任务起点
    2. 智能体完成任务
    3. 智能体探索地图
    4. 智能体寻找充电站充电
    5. 智能体避免碰撞
    
    Args:
        task_allocator: 任务分配器
        agents: 智能体列表
        tasks: 任务列表
        max_steps: 最大步数
        seed: 随机种子
        
    Returns:
        List[List[Tuple[int, int]]]: 智能体移动计划列表，每个智能体一个列表
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # 初始化移动计划，每个智能体一个列表
    movement_plans = [[] for _ in range(len(agents))]
    
    # 为每个智能体分配初始任务
    for i, agent in enumerate(agents):
        if i < len(tasks):
            agent.assign_task(tasks[i])
    
    # 每个智能体计算从当前位置到任务起点的路径
    for i, agent in enumerate(agents):
        # 当前位置
        current_pos = tuple(map(int, agent.pos))
        movement_plans[i].append(current_pos)
        
        # 分配的任务路径
        task_path = []
        
        if agent.current_task:
            # 规划到任务起点的路径
            path_to_start = plan_path(
                current_pos, 
                tuple(map(int, agent.current_task.start_pos)),
                task_allocator.obstacle_map
            )
            
            # 确保路径有效 - 起点与终点相同时跳过
            if len(path_to_start) > 1:
                # 添加到移动计划中
                task_path.extend(path_to_start[1:])  # 跳过起点（已经在当前位置）
            
            # 规划从起点到终点的路径
            start_pos = path_to_start[-1] if path_to_start else current_pos
            path_to_goal = plan_path(
                start_pos,
                tuple(map(int, agent.current_task.goal_pos)),
                task_allocator.obstacle_map
            )
            
            # 确保路径有效
            if len(path_to_goal) > 1:
                # 添加到移动计划中，避免重复位置
                if task_path and task_path[-1] == path_to_goal[0]:
                    task_path.extend(path_to_goal[1:])
                else:
                    task_path.extend(path_to_goal)
        
        # 添加探索行为 - 根据智能体编号选择不同的探索区域
        # 将地图划分为4个象限，每个智能体负责一个
        quadrants = [
            (task_allocator.map_size - 10, task_allocator.map_size - 10),  # 右上
            (10, task_allocator.map_size - 10),  # 左上
            (task_allocator.map_size - 10, 10),  # 右下
            (10, 10)  # 左下
        ]
        
        quadrant_index = i % len(quadrants)
        explore_target = quadrants[quadrant_index]
            
        # 规划探索路径
        if task_path:
            # 如果任务已完成，规划到探索目标的路径
            explore_path = plan_path(
                task_path[-1],
                explore_target,
                task_allocator.obstacle_map
            )
        else:
            # 如果没有任务，直接从当前位置规划到探索目标
            explore_path = plan_path(
                current_pos,
                explore_target,
                task_allocator.obstacle_map
            )
        
        # 选择最近的充电站
        nearest_station_pos = None
        nearest_station_dist = float('inf')
        
        for station in task_allocator.charging_stations:
            station_pos = tuple(map(int, station))
            if explore_path:
                dist = manhattan_distance(explore_path[-1], station_pos)
            else:
                dist = manhattan_distance(current_pos, station_pos)
                
            if dist < nearest_station_dist:
                nearest_station_dist = dist
                nearest_station_pos = station_pos
        
        # 规划返回充电站的路径
        if explore_path:
            # 从探索结束位置规划到充电站的路径
            path_to_station = plan_path(
                explore_path[-1],
                nearest_station_pos,
                task_allocator.obstacle_map
            )
        else:
            # 从当前位置规划到充电站的路径
            path_to_station = plan_path(
                current_pos,
                nearest_station_pos,
                task_allocator.obstacle_map
            )
        
        # 组合所有路径段：任务 -> 探索 -> 充电站
        movement_plan = []
        
        # 添加任务路径（如果有）
        if task_path:
            movement_plan.extend(task_path)
        
        # 添加探索路径（如果有）
        if explore_path and len(explore_path) > 1:
            # 避免重复添加位置
            if movement_plan and movement_plan[-1] == explore_path[0]:
                movement_plan.extend(explore_path[1:])
            else:
                movement_plan.extend(explore_path)
        
        # 添加充电站路径（如果有）
        if path_to_station and len(path_to_station) > 1:
            # 避免重复添加位置
            if movement_plan and movement_plan[-1] == path_to_station[0]:
                movement_plan.extend(path_to_station[1:])
            else:
                movement_plan.extend(path_to_station)
        
        # 确保路径中没有障碍物或边界问题
        safe_movement_plan = []
        last_safe_pos = current_pos
        obstacle_map = task_allocator.obstacle_map
        map_size = obstacle_map.shape[0]
        
        for pos in movement_plan:
            # 检查位置是否在地图范围内且不是障碍物
            if (0 <= pos[0] < map_size and 0 <= pos[1] < map_size and 
                obstacle_map[pos[0], pos[1]] == 0):
                safe_movement_plan.append(pos)
                last_safe_pos = pos
            else:
                print(f"警告：智能体{i}的路径中发现不安全位置 {pos}，使用上一个安全位置 {last_safe_pos}")
                safe_movement_plan.append(last_safe_pos)
        
        # 确保移动计划不超过最大步数
        if len(safe_movement_plan) >= max_steps:
            safe_movement_plan = safe_movement_plan[:max_steps]
        else:
            # 如果路径较短，在最后安全位置停留
            while len(safe_movement_plan) < max_steps:
                safe_movement_plan.append(safe_movement_plan[-1] if safe_movement_plan else current_pos)
        
        # 添加到智能体的移动计划
        movement_plans[i] = safe_movement_plan
    
    # 处理碰撞 - 使用时间窗口方法避免碰撞
    collision_free_plans = handle_collisions(movement_plans, max_steps, task_allocator.obstacle_map)
    
    # 再次验证生成的路径是否安全
    for i, plan in enumerate(collision_free_plans):
        for j, pos in enumerate(plan):
            if not (0 <= pos[0] < task_allocator.map_size and 0 <= pos[1] < task_allocator.map_size):
                print(f"错误：智能体{i}的路径在步骤{j}中包含边界外位置 {pos}，替换为最后安全位置")
                if j > 0:
                    plan[j] = plan[j-1]  # 使用前一个位置
                else:
                    plan[j] = tuple(map(int, agents[i].pos))  # 使用初始位置
            elif task_allocator.obstacle_map[pos[0], pos[1]] == 1:
                print(f"错误：智能体{i}的路径在步骤{j}中包含障碍物位置 {pos}，替换为最后安全位置")
                if j > 0:
                    plan[j] = plan[j-1]  # 使用前一个位置
                else:
                    plan[j] = tuple(map(int, agents[i].pos))  # 使用初始位置
    
    return collision_free_plans

def handle_collisions(movement_plans, max_steps, obstacle_map):
    """处理智能体之间的碰撞
    
    使用时间窗口方法（Time-Window Based Collision Resolution）：
    1. 检测每个时间步的碰撞
    2. 当发现碰撞时，让其中一个智能体停留在当前位置
    3. 重复检查直到没有碰撞
    
    Args:
        movement_plans: 原始移动计划
        max_steps: 最大步数
        obstacle_map: 障碍物地图
        
    Returns:
        List[List[Tuple[int, int]]]: 无碰撞的移动计划
    """
    num_agents = len(movement_plans)
    map_size = obstacle_map.shape[0]
    
    # 复制原始计划，避免修改原始数据
    collision_free_plans = [plan.copy() for plan in movement_plans]
    
    # 迭代多次，确保解决所有碰撞
    max_iterations = 5
    for iteration in range(max_iterations):
        # 检查是否还有碰撞
        has_collision = False
        
        # 检查节点碰撞（两个智能体同时占用同一个位置）
        for t in range(min(max_steps, max(len(plan) for plan in collision_free_plans))):
            # 记录每个位置被占用的智能体列表
            occupied_positions = {}
            
            # 检查每个智能体在此时间步的位置
            for i, plan in enumerate(collision_free_plans):
                if t < len(plan):
                    pos = plan[t]
                    
                    # 检查是否有其他智能体在同一位置
                    if pos in occupied_positions:
                        has_collision = True
                        
                        # 随机决定哪个智能体停留在原位置
                        conflicting_agent = random.choice([i, occupied_positions[pos]])
                        
                        # 让智能体停留在前一个位置
                        if t > 0 and t-1 < len(collision_free_plans[conflicting_agent]):
                            # 获取前一个安全位置
                            prev_pos = collision_free_plans[conflicting_agent][t-1]
                            
                            # 确保前一个位置是安全的
                            if (0 <= prev_pos[0] < map_size and 0 <= prev_pos[1] < map_size and 
                                obstacle_map[prev_pos[0], prev_pos[1]] == 0):
                                # 增加等待时间 - 在原地等待1-2步
                                wait_steps = random.randint(1, 2)
                                
                                # 插入等待位置
                                new_plan = collision_free_plans[conflicting_agent][:t]
                                for _ in range(wait_steps):
                                    new_plan.append(prev_pos)
                                new_plan.extend(collision_free_plans[conflicting_agent][t:])
                                
                                # 更新计划，确保不超过最大步数
                                collision_free_plans[conflicting_agent] = new_plan[:max_steps]
                            else:
                                # 如果前一个位置不安全，找一个安全的位置停留
                                new_plan = collision_free_plans[conflicting_agent][:t]
                                
                                # 回溯找到上一个安全位置
                                safe_pos = None
                                for p in range(t-2, -1, -1):
                                    if p < len(collision_free_plans[conflicting_agent]):
                                        test_pos = collision_free_plans[conflicting_agent][p]
                                        if (0 <= test_pos[0] < map_size and 0 <= test_pos[1] < map_size and 
                                            obstacle_map[test_pos[0], test_pos[1]] == 0):
                                            safe_pos = test_pos
                                            break
                                
                                if safe_pos is None:
                                    # 如果找不到安全位置，使用原始位置（应该在初始位置已经是安全的）
                                    safe_pos = collision_free_plans[conflicting_agent][0]
                                
                                # 添加等待
                                for _ in range(3):  # 等待3步
                                    new_plan.append(safe_pos)
                                
                                # 添加剩余计划
                                new_plan.extend(collision_free_plans[conflicting_agent][t:])
                                
                                # 更新计划
                                collision_free_plans[conflicting_agent] = new_plan[:max_steps]
                    else:
                        occupied_positions[pos] = i
        
        # 检查边缘碰撞（两个智能体交换位置）
        for t in range(min(max_steps-1, max(len(plan)-1 for plan in collision_free_plans))):
            for i in range(num_agents):
                for j in range(i+1, num_agents):
                    # 确保两个计划都足够长
                    if (t < len(collision_free_plans[i]) and t+1 < len(collision_free_plans[i]) and
                        t < len(collision_free_plans[j]) and t+1 < len(collision_free_plans[j])):
                        
                        # 获取两个时间步的位置
                        pos_i_t = collision_free_plans[i][t]
                        pos_i_t1 = collision_free_plans[i][t+1]
                        pos_j_t = collision_free_plans[j][t]
                        pos_j_t1 = collision_free_plans[j][t+1]
                        
                        # 检查是否交换位置
                        if pos_i_t == pos_j_t1 and pos_i_t1 == pos_j_t:
                            has_collision = True
                            
                            # 随机决定哪个智能体停留在原位置
                            conflicting_agent = random.choice([i, j])
                            
                            # 让智能体停留在当前位置一步
                            curr_pos = collision_free_plans[conflicting_agent][t]
                            
                            # 确保当前位置是安全的
                            if (0 <= curr_pos[0] < map_size and 0 <= curr_pos[1] < map_size and 
                                obstacle_map[curr_pos[0], curr_pos[1]] == 0):
                                new_plan = collision_free_plans[conflicting_agent][:t+1]
                                new_plan.append(curr_pos)  # 在原地等待一步
                                new_plan.extend(collision_free_plans[conflicting_agent][t+1:])
                                
                                # 更新计划，确保不超过最大步数
                                collision_free_plans[conflicting_agent] = new_plan[:max_steps]
                            else:
                                # 如果当前位置不安全，使用最后一个安全位置
                                safe_pos = None
                                for p in range(t, -1, -1):
                                    if p < len(collision_free_plans[conflicting_agent]):
                                        test_pos = collision_free_plans[conflicting_agent][p]
                                        if (0 <= test_pos[0] < map_size and 0 <= test_pos[1] < map_size and 
                                            obstacle_map[test_pos[0], test_pos[1]] == 0):
                                            safe_pos = test_pos
                                            break
                                
                                if safe_pos is None:
                                    # 如果找不到安全位置，使用原始位置
                                    safe_pos = collision_free_plans[conflicting_agent][0]
                                
                                # 添加等待
                                new_plan = collision_free_plans[conflicting_agent][:t+1]
                                new_plan.append(safe_pos)  # 在安全位置等待一步
                                new_plan.extend(collision_free_plans[conflicting_agent][t+1:])
                                
                                # 更新计划
                                collision_free_plans[conflicting_agent] = new_plan[:max_steps]
        
        # 如果没有检测到碰撞，退出循环
        if not has_collision:
            break
    
    # 最终检查 - 确保所有计划的长度相等（等于max_steps）并且所有位置都是安全的
    for i in range(num_agents):
        safe_plan = []
        last_safe_pos = None
        
        # 确保每个位置都是安全的
        for pos in collision_free_plans[i]:
            if (0 <= pos[0] < map_size and 0 <= pos[1] < map_size and 
                obstacle_map[pos[0], pos[1]] == 0):
                safe_plan.append(pos)
                last_safe_pos = pos
            elif last_safe_pos is not None:
                # 如果位置不安全，使用上一个安全位置
                safe_plan.append(last_safe_pos)
            else:
                # 如果没有安全位置，尝试找一个
                for x in range(map_size):
                    for y in range(map_size):
                        if obstacle_map[x, y] == 0:
                            safe_plan.append((x, y))
                            last_safe_pos = (x, y)
                            break
                    if last_safe_pos is not None:
                        break
                
                # 如果还是找不到安全位置（不太可能发生），使用原位置
                if last_safe_pos is None:
                    safe_plan.append(pos)  # 使用原始位置，尽管它可能不安全
        
        # 如果计划太短，在末尾重复最后一个位置
        if len(safe_plan) < max_steps:
            last_pos = safe_plan[-1] if safe_plan else (0, 0)  # 确保有默认值
            while len(safe_plan) < max_steps:
                safe_plan.append(last_pos)
        elif len(safe_plan) > max_steps:
            # 如果计划太长，截断
            safe_plan = safe_plan[:max_steps]
        
        collision_free_plans[i] = safe_plan
    
    return collision_free_plans

def plan_path(start_pos, goal_pos, obstacle_map):
    """使用A*算法规划安全路径，严格确保不会穿过障碍物
    
    Args:
        start_pos: 起始位置
        goal_pos: 目标位置
        obstacle_map: 障碍物地图
        
    Returns:
        List[Tuple[int, int]]: 路径点列表
    """
    # 确保位置是元组形式
    start_pos = tuple(map(int, start_pos))
    goal_pos = tuple(map(int, goal_pos))
    
    # 确保起点和终点不在障碍物上
    map_size = obstacle_map.shape[0]
    if 0 <= start_pos[0] < map_size and 0 <= start_pos[1] < map_size and obstacle_map[start_pos] == 1:
        # 起点在障碍物上，找最近的非障碍物位置
        found_safe_start = False
        for dist in range(1, 8):  # 在更大半径内寻找（增大搜索范围）
            for dx in range(-dist, dist+1):
                for dy in range(-dist, dist+1):
                    nx, ny = start_pos[0] + dx, start_pos[1] + dy
                    if 0 <= nx < map_size and 0 <= ny < map_size and obstacle_map[nx, ny] == 0:
                        # 确保找到的点周围至少有2个方向可通行
                        free_directions = 0
                        for direction in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            cx, cy = nx + direction[0], ny + direction[1]
                            if 0 <= cx < map_size and 0 <= cy < map_size and obstacle_map[cx, cy] == 0:
                                free_directions += 1
                        
                        if free_directions >= 2:  # 至少2个方向可通行，避免死角
                            start_pos = (nx, ny)
                            found_safe_start = True
                            break
                if found_safe_start:
                    break
            if found_safe_start:
                break
        
        # 如果找不到合适位置，使用任意一个非障碍物位置
        if not found_safe_start:
            for x in range(map_size):
                for y in range(map_size):
                    if obstacle_map[x, y] == 0:
                        start_pos = (x, y)
                        found_safe_start = True
                        break
                if found_safe_start:
                    break
    
    if 0 <= goal_pos[0] < map_size and 0 <= goal_pos[1] < map_size and obstacle_map[goal_pos] == 1:
        # 终点在障碍物上，找最近的非障碍物位置
        found_safe_goal = False
        for dist in range(1, 8):  # 在更大半径内寻找（增大搜索范围）
            for dx in range(-dist, dist+1):
                for dy in range(-dist, dist+1):
                    nx, ny = goal_pos[0] + dx, goal_pos[1] + dy
                    if 0 <= nx < map_size and 0 <= ny < map_size and obstacle_map[nx, ny] == 0:
                        # 确保找到的点周围至少有2个方向可通行
                        free_directions = 0
                        for direction in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            cx, cy = nx + direction[0], ny + direction[1]
                            if 0 <= cx < map_size and 0 <= cy < map_size and obstacle_map[cx, cy] == 0:
                                free_directions += 1
                        
                        if free_directions >= 2:  # 至少2个方向可通行，避免死角
                            goal_pos = (nx, ny)
                            found_safe_goal = True
                            break
                if found_safe_goal:
                    break
            if found_safe_goal:
                break
        
        # 如果找不到合适位置，使用任意一个非障碍物位置
        if not found_safe_goal:
            for x in range(map_size):
                for y in range(map_size):
                    if obstacle_map[x, y] == 0:
                        goal_pos = (x, y)
                        found_safe_goal = True
                        break
                if found_safe_goal:
                    break
    
    # 如果起点和终点是同一个位置，直接返回
    if start_pos == goal_pos:
        return [start_pos]
    
    # 使用A*算法寻找最短路径
    open_set = []  # 优先队列
    from heapq import heappush, heappop
    
    # 启发式函数：曼哈顿距离
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    # 初始化起点
    g_score = {start_pos: 0}  # 从起点到当前点的实际代价
    f_score = {start_pos: heuristic(start_pos, goal_pos)}  # 估计总代价
    
    # 使用优先队列，按照f_score排序
    heappush(open_set, (f_score[start_pos], start_pos))
    
    came_from = {}  # 记录每个点的前一个点
    closed_set = set()  # 已经探索过的点
    
    # 四个方向：上、右、下、左
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    while open_set:
        _, current = heappop(open_set)
        
        if current == goal_pos:
            # 构建路径
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            
            path.reverse()  # 反转路径，从起点到终点
            
            # 额外安全检查：确保路径上没有障碍物
            safe_path = []
            for pos in path:
                if 0 <= pos[0] < map_size and 0 <= pos[1] < map_size and obstacle_map[pos] == 0:
                    safe_path.append(pos)
                else:
                    print(f"警告：路径中发现障碍物位置 {pos}，跳过")
            
            # 如果路径为空，返回起点
            if not safe_path:
                return [start_pos]
            
            # 如果起点不在路径中，添加起点
            if safe_path[0] != start_pos:
                safe_path.insert(0, start_pos)
            
            # 如果路径非空但不包含终点，添加终点（如果终点是安全的）
            if safe_path[-1] != goal_pos and obstacle_map[goal_pos] == 0:
                safe_path.append(goal_pos)
            
            return safe_path
        
        closed_set.add(current)
        
        # 检查四个方向
        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)
            
            # 检查是否有效（在地图范围内、不是障碍物、未访问过）
            if (not (0 <= neighbor[0] < map_size and 0 <= neighbor[1] < map_size) or
                obstacle_map[neighbor] == 1 or
                neighbor in closed_set):
                continue
            
            # 计算从起点到邻居的代价
            tentative_g_score = g_score.get(current, float('inf')) + 1
            
            # 如果这个邻居已经在open_set中，且新路径不比旧路径更好，则跳过
            if tentative_g_score >= g_score.get(neighbor, float('inf')):
                continue
            
            # 这是目前找到的最佳路径，记录它
            came_from[neighbor] = current
            g_score[neighbor] = tentative_g_score
            f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal_pos)
            
            # 添加到优先队列中
            heappush(open_set, (f_score[neighbor], neighbor))
    
    # 如果无法找到路径，尝试对角线移动
    print(f"无法找到从 {start_pos} 到 {goal_pos} 的路径，尝试使用带对角线的搜索")
    
    # 重置A*算法，这次包括对角线移动
    open_set = []
    g_score = {start_pos: 0}
    f_score = {start_pos: heuristic(start_pos, goal_pos)}
    heappush(open_set, (f_score[start_pos], start_pos))
    came_from = {}
    closed_set = set()
    
    # 八个方向：上、右上、右、右下、下、左下、左、左上
    directions_with_diagonals = [
        (0, 1), (1, 1), (1, 0), (1, -1), 
        (0, -1), (-1, -1), (-1, 0), (-1, 1)
    ]
    
    while open_set:
        _, current = heappop(open_set)
        
        if current == goal_pos:
            # 构建路径
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            
            path.reverse()
            
            # 确保路径上没有障碍物
            safe_path = []
            for pos in path:
                if 0 <= pos[0] < map_size and 0 <= pos[1] < map_size and obstacle_map[pos] == 0:
                    safe_path.append(pos)
                else:
                    print(f"警告：路径中发现障碍物位置 {pos}，跳过")
            
            # 如果路径为空，返回起点
            if not safe_path:
                return [start_pos]
            
            # 如果起点不在路径中，添加起点
            if safe_path[0] != start_pos:
                safe_path.insert(0, start_pos)
            
            # 如果路径非空但不包含终点，添加终点（如果终点是安全的）
            if safe_path[-1] != goal_pos and obstacle_map[goal_pos] == 0:
                safe_path.append(goal_pos)
            
            return safe_path
        
        closed_set.add(current)
        
        for dx, dy in directions_with_diagonals:
            neighbor = (current[0] + dx, current[1] + dy)
            
            # 检查是否有效
            if (not (0 <= neighbor[0] < map_size and 0 <= neighbor[1] < map_size) or
                obstacle_map[neighbor] == 1 or
                neighbor in closed_set):
                continue
            
            # 对角线移动时，确保两个相邻格子也不是障碍物
            if abs(dx) + abs(dy) == 2:  # 对角线移动
                if (obstacle_map.get((current[0] + dx, current[1]), 1) == 1 or 
                    obstacle_map.get((current[0], current[1] + dy), 1) == 1):
                    continue
            
            # 计算代价（对角线移动的代价为1.4）
            movement_cost = 1.4 if abs(dx) + abs(dy) == 2 else 1.0
            tentative_g_score = g_score.get(current, float('inf')) + movement_cost
            
            if tentative_g_score >= g_score.get(neighbor, float('inf')):
                continue
            
            came_from[neighbor] = current
            g_score[neighbor] = tentative_g_score
            f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal_pos)
            
            heappush(open_set, (f_score[neighbor], neighbor))
    
    # 如果最终无法找到路径，返回一条直接的路径，但保证每一步都不在障碍物上
    print(f"无法找到从 {start_pos} 到 {goal_pos} 的路径，构建安全直线路径")
    
    # 使用直线插值创建一条直接朝向目标的路径，但跳过障碍物
    direct_path = [start_pos]
    current = start_pos
    
    # 使用Bresenham算法创建直线路径
    steps = max(abs(goal_pos[0] - start_pos[0]), abs(goal_pos[1] - start_pos[1]))
    if steps == 0:
        return [start_pos]
    
    dx = (goal_pos[0] - start_pos[0]) / steps
    dy = (goal_pos[1] - start_pos[1]) / steps
    
    for i in range(1, steps + 1):
        x = int(start_pos[0] + dx * i)
        y = int(start_pos[1] + dy * i)
        
        # 确保位置在地图范围内
        if not (0 <= x < map_size and 0 <= y < map_size):
            continue
        
        # 如果点在障碍物上，尝试找附近的非障碍物点
        if obstacle_map[x, y] == 1:
            found_safe = False
            for search_dist in range(1, 4):  # 在附近搜索安全点
                for off_x in range(-search_dist, search_dist + 1):
                    for off_y in range(-search_dist, search_dist + 1):
                        nx, ny = x + off_x, y + off_y
                        if (0 <= nx < map_size and 0 <= ny < map_size and 
                            obstacle_map[nx, ny] == 0):
                            # 确保与前一个点是连通的
                            if abs(nx - current[0]) <= 1 and abs(ny - current[1]) <= 1:
                                x, y = nx, ny
                                found_safe = True
                                break
                    if found_safe:
                        break
                if found_safe:
                    break
            
            if not found_safe:
                # 如果找不到连通的安全点，就使用上一个安全点
                x, y = current
        
        if (x, y) != current:  # 避免重复添加相同位置
            current = (x, y)
            direct_path.append(current)
    
    # 如果路径末尾不是目标点，并且目标点是安全的，添加目标点
    if direct_path[-1] != goal_pos and obstacle_map[goal_pos] == 0:
        direct_path.append(goal_pos)
    
    # 最终安全检查
    safe_direct_path = []
    last_safe = start_pos
    for pos in direct_path:
        if 0 <= pos[0] < map_size and 0 <= pos[1] < map_size and obstacle_map[pos] == 0:
            safe_direct_path.append(pos)
            last_safe = pos
        else:
            safe_direct_path.append(last_safe)
    
    return safe_direct_path

def manhattan_distance(a, b):
    """计算曼哈顿距离
    
    Args:
        a: 位置1
        b: 位置2
        
    Returns:
        int: 曼哈顿距离
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def run_custom_simulation(
    num_agents: int = 4,
    map_size: int = 40,
    num_tasks: int = 6,  # 将默认任务数量从4修改为6
    max_steps: int = 100,
    obstacle_density: float = 0.15,
    seed: int = 42,
    output_path: str = 'visualizations/custom_simulation.gif',
    fps: int = 5,
    avoid_obstacles: bool = True,
    avoid_collisions: bool = True,
    safety_margin: int = 1
):
    """运行自定义模拟
    
    Args:
        num_agents: 智能体数量
        map_size: 地图大小
        num_tasks: 任务数量，默认为6个
        max_steps: 最大步数
        obstacle_density: 障碍物密度
        seed: 随机种子
        output_path: 输出路径
        fps: 帧率
        avoid_obstacles: 是否避开障碍物
        avoid_collisions: 是否避开智能体碰撞
        safety_margin: 安全距离
    """
    # 设置环境
    setup_environment(seed)
    
    # 创建场景
    task_allocator, agents, tasks = create_scenario(
        num_agents=num_agents,
        map_size=map_size,
        num_tasks=num_tasks,
        obstacle_density=obstacle_density,
        seed=seed
    )
    
    # 调试: 打印障碍物数量
    obstacle_count = np.sum(task_allocator.obstacle_map)
    total_cells = task_allocator.obstacle_map.size
    print(f"调试: 障碍物数量 = {obstacle_count}, 总格子数 = {total_cells}, 实际密度 = {obstacle_count/total_cells:.4f}")
    
    # 严格检查: 确保所有智能体初始位置不在障碍物上
    for i, agent in enumerate(agents):
        pos = tuple(map(int, agent.pos))
        if task_allocator.obstacle_map[pos[0], pos[1]] == 1:
            print(f"严重错误: 智能体{i}初始位置 {pos} 在障碍物上!")
            
            # 寻找安全位置
            for dist in range(1, 10):
                found = False
                for dx in range(-dist, dist+1):
                    for dy in range(-dist, dist+1):
                        nx, ny = pos[0] + dx, pos[1] + dy
                        if (0 <= nx < map_size and 0 <= ny < map_size and 
                            task_allocator.obstacle_map[nx, ny] == 0):
                            agent.pos = np.array([nx, ny])
                            print(f"  修正: 智能体{i}移动到安全位置 {(nx, ny)}")
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
    
    # 严格检查: 确保所有任务起点和终点不在障碍物上
    for i, task in enumerate(tasks):
        start_pos = tuple(map(int, task.start_pos))
        goal_pos = tuple(map(int, task.goal_pos))
        
        # 检查起点
        if task_allocator.obstacle_map[start_pos[0], start_pos[1]] == 1:
            print(f"严重错误: 任务{i}起点 {start_pos} 在障碍物上!")
            
            # 寻找安全位置
            for dist in range(1, 10):
                found = False
                for dx in range(-dist, dist+1):
                    for dy in range(-dist, dist+1):
                        nx, ny = start_pos[0] + dx, start_pos[1] + dy
                        if (0 <= nx < map_size and 0 <= ny < map_size and 
                            task_allocator.obstacle_map[nx, ny] == 0):
                            task.start_pos = np.array([nx, ny])
                            print(f"  修正: 任务{i}起点移动到安全位置 {(nx, ny)}")
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
        
        # 检查终点
        if task_allocator.obstacle_map[goal_pos[0], goal_pos[1]] == 1:
            print(f"严重错误: 任务{i}终点 {goal_pos} 在障碍物上!")
            
            # 寻找安全位置
            for dist in range(1, 10):
                found = False
                for dx in range(-dist, dist+1):
                    for dy in range(-dist, dist+1):
                        nx, ny = goal_pos[0] + dx, goal_pos[1] + dy
                        if (0 <= nx < map_size and 0 <= ny < map_size and 
                            task_allocator.obstacle_map[nx, ny] == 0):
                            task.goal_pos = np.array([nx, ny])
                            print(f"  修正: 任务{i}终点移动到安全位置 {(nx, ny)}")
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
    
    # 规划智能体移动
    movement_plans = plan_agent_movements(
        task_allocator=task_allocator,
        agents=agents,
        tasks=tasks,
        max_steps=max_steps,
        seed=seed
    )
    
    # 安全检查: 确保路径中不包含障碍物
    for i, plan in enumerate(movement_plans):
        for t, pos in enumerate(plan):
            if 0 <= pos[0] < map_size and 0 <= pos[1] < map_size:
                if task_allocator.obstacle_map[pos[0], pos[1]] == 1:
                    print(f"路径错误: 智能体{i}在步骤{t}计划移动到障碍物位置 {pos}，替换为安全位置")
                    
                    # 寻找最近安全位置
                    if t > 0:
                        # 使用上一个位置
                        movement_plans[i][t] = movement_plans[i][t-1]
                    else:
                        # 使用智能体初始位置
                        movement_plans[i][t] = tuple(map(int, agents[i].pos))
    
    # 输出规划统计信息
    for i, plan in enumerate(movement_plans):
        unique_positions = len(set(plan))
        total_positions = len(plan)
        agent_start = tuple(map(int, agents[i].pos))
        agent_end = plan[-1] if plan else agent_start
        
        print(f"智能体{i}: 总步数={total_positions}, 唯一位置数={unique_positions}, " +
              f"起点={agent_start}, 终点={agent_end}")
        
        if i < len(tasks):
            task = tasks[i]
            task_start = tuple(map(int, task.start_pos))
            task_goal = tuple(map(int, task.goal_pos))
            print(f"  任务{i}: 起点={task_start}, 终点={task_goal}")
    
    # 创建并保存动画
    create_animation(
        task_allocator=task_allocator,
        agents=agents,
        tasks=tasks,
        movement_plans=movement_plans,
        max_steps=max_steps,
        output_path=output_path,
        fps=fps
    )

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Custom Multi-agent Pathfinding Visualization Tool')
    
    parser.add_argument('--num_agents', type=int, default=4, help='Number of agents')
    parser.add_argument('--map_size', type=int, default=40, help='Map size')
    parser.add_argument('--num_tasks', type=int, default=6, help='Number of tasks')
    parser.add_argument('--max_steps', type=int, default=100, help='Maximum steps')
    parser.add_argument('--obstacle_density', type=float, default=0.15, help='Obstacle density')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='visualizations/custom_simulation.gif', help='Output path')
    parser.add_argument('--fps', type=int, default=5, help='Frames per second')
    parser.add_argument('--avoid_obstacles', action='store_true', help='Enable strict obstacle avoidance')
    parser.add_argument('--avoid_collisions', action='store_true', help='Enable strict collision avoidance')
    parser.add_argument('--safety_margin', type=int, default=1, help='Safety margin around obstacles')
    
    args = parser.parse_args()
    
    print("="*50)
    print("Custom Multi-agent Pathfinding Visualization Configuration:")
    print(f"Number of agents: {args.num_agents}")
    print(f"Map size: {args.map_size}")
    print(f"Number of tasks: {args.num_tasks}")
    print(f"Maximum steps: {args.max_steps}")
    print(f"Obstacle density: {args.obstacle_density}")
    print(f"Random seed: {args.seed}")
    print(f"Output path: {args.output}")
    print(f"FPS: {args.fps}")
    print(f"Avoid obstacles: {args.avoid_obstacles}")
    print(f"Avoid collisions: {args.avoid_collisions}")
    print(f"Safety margin: {args.safety_margin}")
    print("="*50)
    
    run_custom_simulation(
        num_agents=args.num_agents,
        map_size=args.map_size,
        num_tasks=args.num_tasks,
        max_steps=args.max_steps,
        obstacle_density=args.obstacle_density,
        seed=args.seed,
        output_path=args.output,
        fps=args.fps,
        avoid_obstacles=args.avoid_obstacles,
        avoid_collisions=args.avoid_collisions,
        safety_margin=args.safety_margin
    )

if __name__ == "__main__":
    main() 