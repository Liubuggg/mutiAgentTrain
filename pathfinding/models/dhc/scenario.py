import numpy as np
import random
from typing import List, Tuple, Optional

from pathfinding.models.dhc.agent import Agent
from pathfinding.models.dhc.task import Task, TaskPriority
from pathfinding.models.dhc.task_allocator import TaskAllocator
from pathfinding.models.dhc.task_generator import TaskGenerator

def is_valid_position(pos: Tuple[int, int], obstacle_map: np.ndarray) -> bool:
    """检查位置是否有效（不在障碍物上）
    
    Args:
        pos: 位置坐标 (x, y)
        obstacle_map: 障碍物地图
        
    Returns:
        bool: 位置是否有效
    """
    x, y = pos
    map_size = obstacle_map.shape[0]
    if x < 0 or x >= map_size or y < 0 or y >= map_size:
        return False
    return obstacle_map[x, y] == 0

def get_valid_position(obstacle_map: np.ndarray, excluded_positions: set) -> Tuple[int, int]:
    """获取一个有效的位置（不在障碍物上且不在已占用位置中）
    
    Args:
        obstacle_map: 障碍物地图
        excluded_positions: 已被占用的位置集合
        
    Returns:
        Tuple[int, int]: 有效位置坐标
    """
    map_size = obstacle_map.shape[0]
    available_positions = [(x, y) for x in range(map_size) for y in range(map_size)
                         if obstacle_map[x, y] == 0 and (x, y) not in excluded_positions]
    
    if not available_positions:
        raise ValueError("没有可用的有效位置")
    
    return random.choice(available_positions)

def create_scenario(
    num_agents: int,
    map_size: int,
    num_tasks: int,
    num_charging_stations: int = 2,
    obstacle_density: float = 0.3
) -> Tuple[TaskAllocator, List[Agent], List[Task]]:
    """创建场景，确保所有位置都不在障碍物上
    
    Args:
        num_agents: 智能体数量
        map_size: 地图大小
        num_tasks: 任务数量
        num_charging_stations: 充电站数量
        obstacle_density: 障碍物密度
        
    Returns:
        Tuple: (task_allocator, agents, tasks)
    """
    # 限制最大障碍物密度
    max_obstacle_density = 0.3
    if obstacle_density > max_obstacle_density:
        print(f"警告：障碍物密度 {obstacle_density} 过高，调整为 {max_obstacle_density}")
        obstacle_density = max_obstacle_density

    # 创建任务分配器
    task_allocator = TaskAllocator(num_agents, map_size)
    
    # 创建障碍物地图
    obstacle_map = np.zeros((map_size, map_size), dtype=np.int32)
    
    # 预留位置用于智能体和充电站
    reserved_positions = set()
    
    # 为智能体预留位置
    for _ in range(num_agents):
        while True:
            x = np.random.randint(0, map_size)
            y = np.random.randint(0, map_size)
            pos = (x, y)
            if pos not in reserved_positions:
                reserved_positions.add(pos)
                break
    
    # 为充电站预留位置
    for _ in range(num_charging_stations):
        while True:
            x = np.random.randint(0, map_size)
            y = np.random.randint(0, map_size)
            pos = (x, y)
            if pos not in reserved_positions:
                reserved_positions.add(pos)
                break
    
    # 生成障碍物，避开预留位置
    total_positions = map_size * map_size
    max_obstacles = int(total_positions * obstacle_density)
    available_obstacle_positions = [(x, y) for x in range(map_size) for y in range(map_size) 
                                  if (x, y) not in reserved_positions]
    num_obstacles = min(max_obstacles, len(available_obstacle_positions))
    obstacle_positions = random.sample(available_obstacle_positions, num_obstacles)
    for x, y in obstacle_positions:
        obstacle_map[x, y] = 1
    
    # 验证预留位置是否为非障碍物
    for x, y in reserved_positions:
        if obstacle_map[x, y] == 1:
            raise RuntimeError(f"预留位置 ({x}, {y}) 被错误标记为障碍物")
    
    task_allocator.set_obstacle_map(obstacle_map)
    
    # 找到可用位置（非障碍物且不在预留位置）
    available_positions = [(x, y) for x in range(map_size) for y in range(map_size) 
                         if obstacle_map[x, y] == 0 and (x, y) not in reserved_positions]
    
    # 确保有足够位置用于任务
    required_positions = num_agents + num_charging_stations + 2 * num_tasks
    if len(available_positions) + len(reserved_positions) < required_positions:
        raise ValueError(f"不足以容纳所有实体：需要 {required_positions} 个位置，仅有 {len(available_positions) + len(reserved_positions)} 个")
    
    # 创建智能体
    agents = []
    agent_positions = list(reserved_positions)[:num_agents]
    for i, pos in enumerate(agent_positions):
        battery = random.uniform(60, 100)
        agent = Agent(
            id=i,
            pos=np.array(pos),  # 确保为np.array
            map_size=map_size,
            max_battery=100.0,
            communication_range=10.0
        )
        agent.current_battery = battery
        agent.state['experience'] = random.uniform(0, 5)
        agent.pathfinder.set_obstacle_map(obstacle_map)
        agent.pathfinder.explored_area = np.ones((map_size, map_size), dtype=bool)
        agents.append(agent)
        task_allocator.add_agent(agent)
    
    # 创建充电站
    charging_stations = []
    station_positions = list(reserved_positions)[num_agents:num_agents+num_charging_stations]
    for pos in station_positions:
        charging_stations.append(np.array(pos))  # 确保为np.array
    task_allocator.set_charging_stations(charging_stations)
    
    # 生成任务
    tasks = []
    occupied_positions = set(reserved_positions)  # 跟踪所有占用位置
    for i in range(num_tasks):
        if len(available_positions) < 2:
            print(f"警告：仅生成 {i} 个任务，剩余位置不足")
            break
        
        # 选择起点
        start_pos = random.choice(available_positions)
        available_positions.remove(start_pos)
        occupied_positions.add(start_pos)
        
        # 选择终点
        goal_pos = random.choice(available_positions)
        available_positions.remove(goal_pos)
        occupied_positions.add(goal_pos)
        
        # 创建任务
        task = Task(
            id=i,
            start_pos=np.array(start_pos),  # 确保为np.array
            goal_pos=np.array(goal_pos),
            priority=random.choice(list(TaskPriority))
        )
        task.appear_time = 0
        tasks.append(task)
        task_allocator.add_task(task)
    
    # 验证所有实体位置
    def validate_positions(obstacle_map, agents, tasks, charging_stations):
        errors = []
        for agent in agents:
            x, y = agent.pos
            if obstacle_map[x, y] == 1:
                errors.append(f"Agent {agent.id} at obstacle position {agent.pos}")
        for task in tasks:
            start_x, start_y = task.start_pos
            goal_x, goal_y = task.goal_pos
            if obstacle_map[start_x, start_y] == 1:
                errors.append(f"Task {task.id} start at obstacle position {task.start_pos}")
            if obstacle_map[goal_x, goal_y] == 1:
                errors.append(f"Task {task.id} goal at obstacle position {task.goal_pos}")
        for i, station in enumerate(charging_stations):
            x, y = station
            if obstacle_map[x, y] == 1:
                errors.append(f"Charging station {i} at obstacle position {station}")
        if errors:
            raise RuntimeError("Validation failed:\n" + "\n".join(errors))
    
    validate_positions(obstacle_map, agents, tasks, charging_stations)
    
    return task_allocator, agents, tasks

def check_collision(pos: Tuple[int, int], obstacle_map: np.ndarray) -> bool:
    """检查给定位置是否会与障碍物发生碰撞
    
    Args:
        pos: 位置坐标 (x, y)
        obstacle_map: 障碍物地图
        
    Returns:
        bool: 是否发生碰撞
    """
    return not is_valid_position(pos, obstacle_map)

def get_valid_actions(pos: Tuple[int, int], obstacle_map: np.ndarray) -> List[Tuple[int, int]]:
    """获取当前位置的有效动作（不会导致碰撞）
    
    Args:
        pos: 当前位置
        obstacle_map: 障碍物地图
        
    Returns:
        List[Tuple[int, int]]: 有效的动作列表（相对移动）
    """
    actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 上右下左
    valid_actions = []
    
    for dx, dy in actions:
        new_pos = (pos[0] + dx, pos[1] + dy)
        if is_valid_position(new_pos, obstacle_map):
            valid_actions.append((dx, dy))
    
    return valid_actions