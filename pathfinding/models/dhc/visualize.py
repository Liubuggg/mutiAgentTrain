import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
import random
import torch
import time
from matplotlib.patches import Rectangle, Circle
from tqdm import tqdm
from typing import Dict, List, Tuple, Any, Optional

from pathfinding.models.dhc.model import Network, device
from pathfinding.models.dhc.agent import Agent
from pathfinding.models.dhc.task import Task, TaskStatus, TaskPriority
from pathfinding.models.dhc.task_generator import TaskGenerator
from pathfinding.models.dhc.task_allocator import TaskAllocator
from pathfinding.settings import yaml_data as settings

# Set style
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
    'grid': '#dddddd'
}

def setup_environment(seed: int = 42):
    """Set up environment and random seed
    
    Args:
        seed: Random seed
    """
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Create output directory
    os.makedirs("visualizations", exist_ok=True)

from pathfinding.models.dhc.scenario import create_scenario, check_collision, get_valid_actions

def render_frame(
    task_allocator: TaskAllocator,
    agents: List[Agent],
    tasks: List[Task],
    step: int,
    ax: plt.Axes,
    title: Optional[str] = None,
    show_communication: bool = False,
    show_task_info: bool = True
) -> List:
    """Render single frame scenario"""
    artists = []
    
    # Clear axis
    ax.clear()
    
    # Set boundaries and background
    ax.set_xlim(-1, task_allocator.map_size + 1)
    ax.set_ylim(-1, task_allocator.map_size + 1)
    
    # Draw obstacle map
    if hasattr(task_allocator, 'obstacle_map') and task_allocator.obstacle_map is not None:
        for x in range(task_allocator.map_size):
            for y in range(task_allocator.map_size):
                if task_allocator.obstacle_map[x, y] == 1:
                    rect = Rectangle((x - 0.5, y - 0.5), 1, 1, color='#444444', alpha=0.8, zorder=1)
                    ax.add_patch(rect)
                    artists.append(rect)
    
    # Add grid
    ax.grid(True, color=COLORS['grid'], linestyle='-', linewidth=0.5)
    
    # Set title
    if title:
        title_obj = ax.set_title(title, fontsize=14, fontfamily='DejaVu Sans')
    else:
        title_obj = ax.set_title(f'Step: {step}', fontsize=14, fontfamily='DejaVu Sans')
    artists.append(title_obj)
    
    # Draw charging stations
    for station in task_allocator.charging_stations:
        x, y = station
        station_marker = ax.scatter(x, y, c=COLORS['charging_station'], marker='^', s=200, zorder=3)
        artists.append(station_marker)
        
        # Add charging station label
        station_text = ax.text(x, y - 1.5, 'Station', ha='center', fontsize=8, color=COLORS['charging_station'], fontfamily='DejaVu Sans')
        artists.append(station_text)
    
    # Draw tasks
    for task in task_allocator.tasks.values():
        start_x, start_y = task.start_pos
        goal_x, goal_y = task.goal_pos
        
        # Choose color and label based on task status
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
            
        # Draw start and end points with labels
        start_marker = ax.scatter(start_x, start_y, c=color, marker='s', s=120, zorder=2)
        goal_marker = ax.scatter(goal_x, goal_y, c=color, marker='*', s=180, zorder=2)
        
        # Draw task path
        path = ax.plot([start_x, goal_x], [start_y, goal_y], c=color, linestyle='--', linewidth=2, alpha=0.5, zorder=1)[0]
        
        # Add task labels
        start_text = ax.text(start_x, start_y - 1.5, f'T{task.id} Start', ha='center', fontsize=8, color=color, fontfamily='DejaVu Sans')
        goal_text = ax.text(goal_x, goal_y + 1.5, f'T{task.id} Goal\n{status_text}', ha='center', fontsize=8, color=color, fontfamily='DejaVu Sans')
        
        artists.extend([start_marker, goal_marker, path, start_text, goal_text])
    
    # Draw agents
    shown_in_legend = False
    for agent in agents:
        x, y = agent.pos
        
        # Choose color and status based on battery percentage
        battery_percentage = agent.current_battery / agent.max_battery
        if battery_percentage < 0.2:
            agent_color = COLORS['low_battery']
            battery_status = 'Low Battery'
        else:
            agent_color = COLORS['agent']
            battery_status = 'Normal'
        
        # Draw agent
        if not shown_in_legend:
            agent_marker = ax.scatter(x, y, c=agent_color, s=150, zorder=4, label='Agent')
            shown_in_legend = True
        else:
            agent_marker = ax.scatter(x, y, c=agent_color, s=150, zorder=4)
        
        # Add agent status information
        status_text = f'Agent {agent.id}\n{battery_status}\n{agent.current_battery:.0f}%'
        if agent.current_task:
            status_text += f'\nTask {agent.current_task.id}'
        
        agent_text = ax.text(
            x, y + 0.5,
            status_text,
            ha='center', va='bottom', fontsize=8,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor=agent_color),
            fontfamily='DejaVu Sans'
        )
        artists.extend([agent_marker, agent_text])
        
        # If agent is executing a task, show task path
        if agent.current_task:
            task = agent.current_task
            if not task.started:
                # Show path to task start
                start_line = ax.plot(
                    [x, task.start_pos[0]], [y, task.start_pos[1]],
                    c=agent_color, linestyle='-', linewidth=2, alpha=0.5, zorder=1
                )[0]
                artists.append(start_line)
            else:
                # Show path to task end
                goal_line = ax.plot(
                    [x, task.goal_pos[0]], [y, task.goal_pos[1]],
                    c=agent_color, linestyle='-', linewidth=2, alpha=0.5, zorder=1
                )[0]
                artists.append(goal_line)
    
    # Add legend
    legend = ax.legend(loc='upper right', fontsize=10, prop={'family': 'DejaVu Sans'})
    artists.append(legend)
    
    return artists

def simulate_model(
    model_path: str,
    num_agents: int = 4,
    map_size: int = 40, 
    num_tasks: int = 8,
    max_steps: int = 100,
    seed: int = 42,
    output_path: str = None,
    fps: int = 5,
    save_static_frames: bool = False,
    static_frame_interval: int = 20,
    output_format: str = 'gif'
):
    """Simulate model and generate animation"""
    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA for visualization")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS for visualization")
    else:
        device = torch.device("cpu")
        print("Using CPU for visualization")
    
    # Load model
    model = None
    try:
        print(f"Loading model: {model_path}")
        # 使用Network类的load_model方法加载模型
        from pathfinding.models.dhc.model import Network
        model = Network.load_model(model_path)
        model.to(device)
        model.eval()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Create output path
    if output_path is None:
        if output_format.lower() == 'mp4':
            output_path = 'visualizations/simulation.mp4'
        else:
            output_path = 'visualizations/simulation.gif'
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create directory for static frames
    static_frames_dir = None
    if save_static_frames:
        static_frames_dir = os.path.join('visualizations', 'static_frames')
        os.makedirs(static_frames_dir, exist_ok=True)
        print(f"Will save static frames to: {static_frames_dir}")
    
    # Create scenario with lower obstacle density to ensure enough space
    task_allocator, agents, tasks = create_scenario(
        num_agents=num_agents, 
        map_size=map_size, 
        num_tasks=num_tasks,
        obstacle_density=0.2  # Lower obstacle density for visualization
    )
    
    # 确保所有实体都不在障碍物上
    if hasattr(task_allocator, 'obstacle_map') and task_allocator.obstacle_map is not None:
        obstacle_map = task_allocator.obstacle_map
        
        # 获取所有非障碍物位置
        valid_positions = []
        for x in range(map_size):
            for y in range(map_size):
                if obstacle_map[x, y] == 0:
                    valid_positions.append((x, y))
        
        if not valid_positions:
            print("错误：没有找到有效位置，障碍物地图可能全是障碍物")
            return
            
        # 已分配的位置
        occupied_positions = set()
            
        # 验证并修复智能体位置
        for agent in agents:
            x, y = agent.pos
            if obstacle_map[x, y] == 1:
                # 找一个未被占用的有效位置
                found_valid_pos = False
                for pos in valid_positions:
                    if pos not in occupied_positions:
                        print(f"修正：智能体 {agent.id} 从障碍物位置 {agent.pos} 移动到 {pos}")
                        agent.pos = np.array(pos)
                        occupied_positions.add(pos)
                        found_valid_pos = True
                        break
                        
                if not found_valid_pos:
                    print(f"警告：无法为智能体 {agent.id} 找到有效位置")
            else:
                occupied_positions.add((x, y))
        
        # 验证并修复任务位置
        for task in list(task_allocator.tasks.values()):
            start_x, start_y = task.start_pos
            goal_x, goal_y = task.goal_pos
            
            # 检查起点
            if obstacle_map[start_x, start_y] == 1:
                found_valid_pos = False
                for pos in valid_positions:
                    if pos not in occupied_positions:
                        print(f"修正：任务 {task.id} 起点从障碍物位置 {task.start_pos} 移动到 {pos}")
                        task.start_pos = np.array(pos)
                        occupied_positions.add(pos)
                        found_valid_pos = True
                        break
                        
                if not found_valid_pos:
                    print(f"警告：无法为任务 {task.id} 起点找到有效位置")
            else:
                occupied_positions.add((start_x, start_y))
                
            # 检查终点
            if obstacle_map[goal_x, goal_y] == 1:
                found_valid_pos = False
                for pos in valid_positions:
                    if pos not in occupied_positions:
                        print(f"修正：任务 {task.id} 终点从障碍物位置 {task.goal_pos} 移动到 {pos}")
                        task.goal_pos = np.array(pos)
                        occupied_positions.add(pos)
                        found_valid_pos = True
                        break
                        
                if not found_valid_pos:
                    print(f"警告：无法为任务 {task.id} 终点找到有效位置")
            else:
                occupied_positions.add((goal_x, goal_y))
        
        # 验证并修复充电站位置
        new_stations = []
        for i, station in enumerate(task_allocator.charging_stations):
            x, y = station
            if obstacle_map[x, y] == 1:
                found_valid_pos = False
                for pos in valid_positions:
                    if pos not in occupied_positions:
                        print(f"修正：充电站从障碍物位置 {station} 移动到 {pos}")
                        new_stations.append(np.array(pos))
                        occupied_positions.add(pos)
                        found_valid_pos = True
                        break
                        
                if not found_valid_pos:
                    print(f"警告：无法为充电站 {i} 找到有效位置")
            else:
                new_stations.append(station)
                occupied_positions.add((x, y))
                
        task_allocator.charging_stations = new_stations
    
    # Ensure PathFinder correctly initialized and all entities are not on obstacles
    for agent in agents:
        if agent.pathfinder.obstacle_map is None and hasattr(task_allocator, 'obstacle_map'):
            agent.pathfinder.set_obstacle_map(task_allocator.obstacle_map)
            agent.pathfinder.explored_area.fill(True)  # Set all areas as explored initially
    
    # Create figure and animation
    plt.ioff()  # Turn off interactive mode
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor(COLORS['background'])
    
    # Create output directory for frames
    frames_dir = os.path.join('visualizations', 'frames')
    os.makedirs(frames_dir, exist_ok=True)
    
    # Initialize model hidden state
    hidden = None
    
    print("Starting simulation...")
    progress_bar = tqdm(total=max_steps)
    
    # Main loop
    step = 0
    done = False
    frame_paths = []
    
    while not done and step < max_steps:
        # Clear current frame
        ax.clear()
        
        # Get agent observation and features
        agent_obs = np.zeros((1, num_agents, 8, 9, 9))  # [B, N, 8, 9, 9]
        agent_states = np.zeros((1, num_agents, 2))     # [B, N, 2] - battery and experience
        task_features = np.zeros((1, num_agents, 4))    # [B, N, 4] - Task features for each agent
        
        # Build observation for each agent
        for i, agent in enumerate(agents):
            # Get observation window around agent
            x, y = agent.pos
            obs_radius = 4  # Observation radius is 4, resulting in a 9x9 observation window
            
            # Extract observation window
            x_min = max(0, x - obs_radius)
            x_max = min(map_size, x + obs_radius + 1)
            y_min = max(0, y - obs_radius)
            y_max = min(map_size, y + obs_radius + 1)
            
            # Calculate padding range
            pad_x_min = obs_radius - (x - x_min)
            pad_x_max = obs_radius + 1 + (x_max - x - 1)
            pad_y_min = obs_radius - (y - y_min)
            pad_y_max = obs_radius + 1 + (y_max - y - 1)
            
            # 1. Obstacle layer
            if hasattr(task_allocator, 'obstacle_map') and task_allocator.obstacle_map is not None:
                agent_obs[0, i, 0, pad_x_min:pad_x_max, pad_y_min:pad_y_max] = \
                    task_allocator.obstacle_map[x_min:x_max, y_min:y_max]
            
            # 2. Other agent positions layer
            agent_map = np.zeros((map_size, map_size))
            for other_agent in agents:
                if other_agent.id != agent.id:
                    ox, oy = other_agent.pos
                    if 0 <= ox < map_size and 0 <= oy < map_size:
                        agent_map[ox, oy] = 1
            agent_obs[0, i, 1, pad_x_min:pad_x_max, pad_y_min:pad_y_max] = \
                agent_map[x_min:x_max, y_min:y_max]
            
            # 3. Task start layer
            task_start_map = np.zeros((map_size, map_size))
            # 4. Task end layer
            task_goal_map = np.zeros((map_size, map_size))
            
            if agent.current_task:
                start_x, start_y = agent.current_task.start_pos
                goal_x, goal_y = agent.current_task.goal_pos
                if 0 <= start_x < map_size and 0 <= start_y < map_size:
                    task_start_map[start_x, start_y] = 1
                if 0 <= goal_x < map_size and 0 <= goal_y < map_size:
                    task_goal_map[goal_x, goal_y] = 1
            agent_obs[0, i, 2, pad_x_min:pad_x_max, pad_y_min:pad_y_max] = \
                task_start_map[x_min:x_max, y_min:y_max]
            agent_obs[0, i, 3, pad_x_min:pad_x_max, pad_y_min:pad_y_max] = \
                task_goal_map[x_min:x_max, y_min:y_max]
            
            # 5-6. Task direction layers (new)
            task_direction_x = np.zeros((map_size, map_size))
            task_direction_y = np.zeros((map_size, map_size))
            
            if agent.current_task:
                # 计算方向指引
                if not agent.current_task.started:
                    # 如果任务尚未开始，指向起点
                    direction = np.array(agent.current_task.start_pos) - np.array(agent.pos)
                else:
                    # 否则指向终点
                    direction = np.array(agent.current_task.goal_pos) - np.array(agent.pos)
                
                # 归一化方向向量
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction = direction / norm
                    
                    # 在智能体周围区域添加方向指引
                    for dx in range(-2, 3):
                        for dy in range(-2, 3):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < map_size and 0 <= ny < map_size:
                                if task_allocator.obstacle_map is None or task_allocator.obstacle_map[nx, ny] == 0:
                                    intensity = 1.0 - (abs(dx) + abs(dy)) * 0.2
                                    if intensity > 0:
                                        task_direction_x[nx, ny] = direction[0] * intensity
                                        task_direction_y[nx, ny] = direction[1] * intensity
            
            agent_obs[0, i, 4, pad_x_min:pad_x_max, pad_y_min:pad_y_max] = \
                task_direction_x[x_min:x_max, y_min:y_max]
            agent_obs[0, i, 5, pad_x_min:pad_x_max, pad_y_min:pad_y_max] = \
                task_direction_y[x_min:x_max, y_min:y_max]
            
            # 7. Charging station layer (moved from channel 5)
            station_map = np.zeros((map_size, map_size))
            for station in task_allocator.charging_stations:
                sx, sy = station
                if 0 <= sx < map_size and 0 <= sy < map_size:
                    station_map[sx, sy] = 1
            agent_obs[0, i, 6, pad_x_min:pad_x_max, pad_y_min:pad_y_max] = \
                station_map[x_min:x_max, y_min:y_max]
            
            # 8. Battery layer (moved from channel 6)
            battery_map = np.zeros((map_size, map_size))
            battery_map[x, y] = agent.current_battery / agent.max_battery
            agent_obs[0, i, 7, pad_x_min:pad_x_max, pad_y_min:pad_y_max] = \
                battery_map[x_min:x_max, y_min:y_max]
            
            # Update agent state
            agent_states[0, i, 0] = agent.current_battery / agent.max_battery  # Normalize battery
            agent_states[0, i, 1] = agent.state.get('experience', 0) / 10.0  # Normalize experience value
            
            # Update task features
            if agent.current_task:
                task = agent.current_task
                # Calculate distance to task start and end
                start_distance = np.linalg.norm(np.array(agent.pos) - np.array(task.start_pos))
                goal_distance = np.linalg.norm(np.array(agent.pos) - np.array(task.goal_pos))
                # Task priority
                priority = task.priority.value / 4.0  # Normalize priority
                # Fill task features
                task_features[0, i, 0] = start_distance / map_size  # Normalize distance
                task_features[0, i, 1] = goal_distance / map_size
                task_features[0, i, 2] = priority
                task_features[0, i, 3] = 1.0  # Task flag
        
        # Generate position information
        positions = np.array([agent.pos for agent in agents]).reshape(1, num_agents, 2)
        
        # Create communication mask
        comm_mask = task_allocator.generate_communication_mask()
        
        # Convert to appropriate tensor shape [1, N, N]
        comm_mask_tensor = torch.tensor(comm_mask, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Convert to tensor
        obs_tensor = torch.tensor(agent_obs, dtype=torch.float32).to(device)
        pos_tensor = torch.tensor(positions, dtype=torch.float32).to(device)
        agent_states_tensor = torch.tensor(agent_states, dtype=torch.float32).to(device)
        task_features_tensor = torch.tensor(task_features, dtype=torch.float32).to(device)
        
        # Calculate action and task assignment
        with torch.no_grad():
            action_values, hidden, task_assignment = model.step(
                obs=obs_tensor,
                pos=pos_tensor,
                tasks=task_features_tensor,
                agent_states=agent_states_tensor, 
                hidden=hidden,
                comm_mask=comm_mask_tensor
            )
        
        # Convert to numpy array
        action_values_np = action_values.detach().cpu().numpy()
        task_assignment_np = task_assignment.detach().cpu().numpy()
        
        # Choose action and task
        actions = np.argmax(action_values_np, axis=-1)
        task_choices = np.argmax(task_assignment_np, axis=-1)
        
        # Update task allocator
        # Assign tasks based on model output
        for agent_idx, agent in enumerate(agents):
            if not agent.current_task:  # Only assign new task when agent has no task
                available_tasks = task_allocator.get_available_tasks(agent)
                if available_tasks:
                    # Get task scores
                    task_scores = task_assignment_np[0, agent_idx, :len(available_tasks)]
                    
                    # Consider battery constraint
                    for i, task in enumerate(available_tasks):
                        # Calculate estimated steps to complete task
                        start_dist = np.linalg.norm(np.array(agent.pos) - np.array(task.start_pos))
                        goal_dist = np.linalg.norm(np.array(task.start_pos) - np.array(task.goal_pos))
                        estimated_steps = (start_dist + goal_dist) * 1.5  # Add 1.5 safety factor
                        
                        # If estimated battery is insufficient, lower task score
                        if agent.current_battery < estimated_steps * 0.2:  # Each step consumes 0.1% battery
                            task_scores[i] *= 0.1
                    
                    if len(task_scores) > 0:
                        best_task_idx = np.argmax(task_scores)
                        task = available_tasks[best_task_idx]
                        
                        # Only assign task if score is high enough
                        if task_scores[best_task_idx] > 0.2:  # Set a threshold
                            task_allocator.assign_task(agent, task)
                            print(f"Step: {step}: 任务{task.id} (优先级: {task.priority.name}) 分配给智能体{agent.id}")

        # Execute action and task assignment
        for i, agent in enumerate(agents):
            # Get action
            agent_action = actions[0, i]  # Get action for this agent
            
            # 检查动作是否会导致碰撞或超出边界
            is_valid_action = True
            if agent_action > 0:  # 不是停止动作
                action_vector = agent.ACTIONS[agent_action]
                new_pos = agent.pos + action_vector
                
                # 检查是否超出地图边界或碰撞
                if not (0 <= new_pos[0] < map_size and 0 <= new_pos[1] < map_size) or \
                   (hasattr(task_allocator, 'obstacle_map') and task_allocator.obstacle_map is not None and 
                    task_allocator.obstacle_map[new_pos[0], new_pos[1]] == 1) or \
                   any(np.array_equal(other_agent.pos, new_pos) for other_agent in agents if other_agent.id != agent.id):
                    
                    target_pos = None
                    if agent.current_task:
                        if not agent.current_task.started:
                            target_pos = agent.current_task.start_pos
                        else:
                            target_pos = agent.current_task.goal_pos
                    elif agent.current_battery / agent.max_battery < 0.3:
                        # 如果电量低，尝试找最近的充电站
                        charging_stations = task_allocator.charging_stations
                        if charging_stations:
                            nearest_station = min(charging_stations, 
                                                key=lambda s: np.linalg.norm(np.array(agent.pos) - np.array(s)))
                            target_pos = nearest_station
                    
                    if target_pos is not None:
                        direction_to_target = np.array(target_pos) - np.array(agent.pos)
                        if abs(direction_to_target[0]) > abs(direction_to_target[1]):
                            if direction_to_target[0] > 0:
                                primary_actions = [3, 1, 2, 4]  # 右、上、下、左
                            else:
                                primary_actions = [4, 1, 2, 3]  # 左、上、下、右
                        else:
                            if direction_to_target[1] > 0:
                                primary_actions = [1, 3, 4, 2]  # 上、右、左、下
                            else:
                                primary_actions = [2, 3, 4, 1]  # 下、右、左、上
                        action_found = False
                        for alt_action in primary_actions:
                            alt_vector = agent.ACTIONS[alt_action]
                            alt_pos = agent.pos + alt_vector
                            if (0 <= alt_pos[0] < map_size and 0 <= alt_pos[1] < map_size and
                                (not hasattr(task_allocator, 'obstacle_map') or 
                                 task_allocator.obstacle_map[alt_pos[0], alt_pos[1]] == 0) and
                                not any(np.array_equal(other_agent.pos, alt_pos) 
                                       for other_agent in agents if other_agent.id != agent.id)):
                                agent_action = alt_action
                                action_found = True
                                break
                        
                        if not action_found:
                            # 如果没有有效动作，则停止
                            agent_action = 0
                            is_valid_action = False
                    else:
                        # 如果没有目标位置，尝试随机方向
                        possible_actions = []
                        for alt_action in range(1, 5):  # 1-4 对应上下左右
                            alt_vector = agent.ACTIONS[alt_action]
                            alt_pos = agent.pos + alt_vector
                            
                            # 检查新位置是否有效
                            if (0 <= alt_pos[0] < map_size and 0 <= alt_pos[1] < map_size and
                                (not hasattr(task_allocator, 'obstacle_map') or 
                                 task_allocator.obstacle_map[alt_pos[0], alt_pos[1]] == 0) and
                                not any(np.array_equal(other_agent.pos, alt_pos) 
                                       for other_agent in agents if other_agent.id != agent.id)):
                                possible_actions.append(alt_action)
                        
                        if possible_actions:
                            # 随机选择一个有效动作
                            agent_action = np.random.choice(possible_actions)
                        else:
                            # 如果没有有效动作，则停止
                            agent_action = 0
                            is_valid_action = False
            
            # 执行动作
            success = agent.execute_action(agent_action)
            
            # 添加电量消耗，即使动作执行失败也消耗一定电量
            if agent_action > 0:  # 如果不是停止动作
                # 基础消耗，即使尝试移动失败也消耗电量
                base_consumption = 0.1  # 与agent.py中的_calculate_movement_energy_cost保持一致
                agent.update_battery(-base_consumption)  # 每次尝试移动都消耗电量
            
            # 如果动作执行失败，执行停止动作
            if not success:
                print(f"Agent {agent.id} 的动作执行失败，执行停止动作")
                agent.execute_action(0)  # 执行停止动作
            
            # 验证智能体位置是否在障碍物上
            if hasattr(task_allocator, 'obstacle_map') and task_allocator.obstacle_map is not None:
                x, y = agent.pos
                if task_allocator.obstacle_map[x, y] == 1:
                    print(f"警告：智能体 {agent.id} 位于障碍物上，位置：{agent.pos}")
                    # 寻找最近的可用位置
                    for dx in range(-3, 4):
                        for dy in range(-3, 4):
                            new_x, new_y = x + dx, y + dy
                            if (0 <= new_x < map_size and 0 <= new_y < map_size and 
                                task_allocator.obstacle_map[new_x, new_y] == 0):
                                print(f"修正：将智能体 {agent.id} 移动到位置 ({new_x}, {new_y})")
                                agent.pos = np.array([new_x, new_y])
                                break
                        else:
                            continue
                        break
            
            # Check for task completion
            if agent.current_task:
                if not agent.current_task.started and np.array_equal(agent.pos, agent.current_task.start_pos):
                    # Agent reached start position - mark task as started
                    agent.current_task.started = True
                    task_reward = 10.0 * agent.current_task.priority.value
                    print(f"Step {step}: 智能体{agent.id}到达任务{agent.current_task.id}起点，额外奖励：{task_reward:.1f}")
                elif agent.current_task.started and np.array_equal(agent.pos, agent.current_task.goal_pos):
                    # Agent reached goal position - mark task as completed
                    agent.current_task.status = TaskStatus.COMPLETED
                    task_reward = 50.0 * agent.current_task.priority.value
                    print(f"Step {step}: --------智能体{agent.id}完成任务{agent.current_task.id}，额外奖励：{task_reward:.1f}--------")
                    agent.current_task = None

        # Update task allocator and get rewards
        rewards = task_allocator.update()
        reward = np.mean(rewards) if len(rewards) > 0 else 0.0
        
        # Render current frame
        render_frame(
            task_allocator, 
            agents, 
            task_allocator.tasks, 
            step, 
            ax, 
            title=f'Step: {step} | Reward: {reward:.2f}',
            show_communication=True,
            show_task_info=True
        )
        
        # Save frame as image file
        frame_path = os.path.join(frames_dir, f'frame_{step:04d}.png')
        plt.savefig(frame_path)
        frame_paths.append(frame_path)
        
        # Check if done
        done = task_allocator.is_done()
        step += 1
        progress_bar.update(1)
        
        # Add new task during simulation
        if step % 10 == 0 and len(task_allocator.tasks) < num_tasks:  # Check every 10 steps if new task is needed
            # Get valid positions (not obstacles)
            valid_positions = []
            
            # First verify obstacle map exists
            if not hasattr(task_allocator, 'obstacle_map') or task_allocator.obstacle_map is None:
                print("Warning: No obstacle map available for task creation")
                continue
                
            obstacle_map = task_allocator.obstacle_map
            
            # Find all valid positions (not obstacles)
            for i in range(map_size):
                for j in range(map_size):
                    if obstacle_map[i, j] == 0:  # Not an obstacle
                        # Also check that position is not occupied by an agent
                        is_occupied = False
                        for agent in agents:
                            if agent.pos[0] == i and agent.pos[1] == j:
                                is_occupied = True
                                break
                                
                        if not is_occupied:
                            valid_positions.append((i, j))
            
            # Ensure there are available positions
            if len(valid_positions) >= 2:
                # Randomly select start and end positions
                start_idx = np.random.randint(0, len(valid_positions))
                start_pos = valid_positions[start_idx]
                # Make a copy to avoid modifying the original position
                start_pos = (start_pos[0], start_pos[1])
                
                # Remove start position from valid positions
                valid_positions.pop(start_idx)
                
                goal_idx = np.random.randint(0, len(valid_positions))
                goal_pos = valid_positions[goal_idx]
                # Make a copy to avoid modifying the original position
                goal_pos = (goal_pos[0], goal_pos[1])
                
                # Verify positions are not obstacles (double-check)
                if obstacle_map[start_pos[0], start_pos[1]] == 0 and obstacle_map[goal_pos[0], goal_pos[1]] == 0:
                    # Create new task
                    new_task = Task(
                        id=len(task_allocator.tasks),
                        start_pos=start_pos,
                        goal_pos=goal_pos,
                        priority=random.choice(list(TaskPriority))
                    )
                    new_task.appear_time = step  # Set task appearance time
                    task_allocator.add_task(new_task)
                    print(f"Step: {step}: 添加新任务 {new_task.id} 从 {start_pos} 到 {goal_pos} (优先级: {new_task.priority.name})")
                else:
                    print(f"警告: 添加任务失败 - 位置检查失败")
    
    progress_bar.close()
    
    print(f"Generating animation with {len(frame_paths)} frames...")
    
    if output_format.lower() == 'mp4':
        try:
            import cv2
            # Create video writer
            video_path = output_path
            first_frame = cv2.imread(frame_paths[0])
            height, width = first_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            
            # Write each frame
            for frame_path in frame_paths:
                frame = cv2.imread(frame_path)
                out.write(frame)
            
            out.release()
            print(f"\nVideo saved to: {video_path}")
            
        except Exception as e:
            print(f"Failed to save video: {e}")
            # Save as GIF as fallback
            try:
                import imageio
                gif_path = os.path.splitext(output_path)[0] + '.gif'
                images = [imageio.imread(frame_path) for frame_path in frame_paths]
                imageio.mimsave(gif_path, images, fps=fps)
                print(f"Saved as GIF instead: {gif_path}")
            except Exception as e:
                print(f"Failed to save GIF: {e}")
    else:
        try:
            import imageio
            images = [imageio.imread(frame_path) for frame_path in frame_paths]
            imageio.mimsave(output_path, images, fps=fps)
            print(f"\nAnimation saved to: {output_path}")
        except Exception as e:
            print(f"Failed to save animation: {e}")
    
    # Clean up frame files
    import shutil
    shutil.rmtree(frames_dir)
    
    plt.close(fig)
    return task_allocator, agents, tasks

def plot_training_results(
    log_dir: str,
    output_dir: str = 'visualizations',
    dpi: int = 100
):
    """Plot training results
    
    Args:
        log_dir: Log directory
        output_dir: Output directory
        dpi: Resolution
    """
    try:
        from tensorboard.backend.event_processing import event_accumulator
        import glob
    except ImportError:
        print("Missing tensorboard library, please install: pip install tensorboard")
        return
    
    # Find all event files
    event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    if not event_files:
        print(f"No tensorboard event files found in directory {log_dir}")
        return
    
    # Load event data
    ea = event_accumulator.EventAccumulator(event_files[0])
    ea.Reload()
    
    # Get available tags
    tags = ea.Tags()['scalars']
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Draw each metric
    for tag in tags:
        print(f"Drawing {tag}...")
        events = ea.Scalars(tag)
        
        steps = [event.step for event in events]
        values = [event.value for event in events]
        
        fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)
        fig.patch.set_facecolor(COLORS['background'])
        
        ax.plot(steps, values, color=COLORS['agent'], linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(tag.split('/')[-1] if '/' in tag else tag)
        ax.set_title(f'Training Progress - {tag}')
        ax.grid(True)
        
        # Calculate moving average
        if len(steps) > 10:
            window_size = max(10, len(steps) // 50)
            moving_avg = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
            moving_avg_steps = steps[window_size-1:]
            
            ax.plot(moving_avg_steps, moving_avg, color='red', linewidth=1.5, 
                    linestyle='--', label=f'Moving Average (Window={window_size})')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{tag.replace('/', '_')}.png"))
        plt.close(fig)
    
    # Additionally, create a combined success rate plot for start and end points
    if 'Success/start_point' in tags and 'Success/end_point' in tags:
        start_events = ea.Scalars('Success/start_point')
        end_events = ea.Scalars('Success/end_point')
        
        start_steps = [event.step for event in start_events]
        start_values = [event.value for event in start_events]
        
        end_steps = [event.step for event in end_events]
        end_values = [event.value for event in end_events]
        
        fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)
        fig.patch.set_facecolor(COLORS['background'])
        
        ax.plot(start_steps, start_values, color='blue', linewidth=2, label='Start Point Success Rate')
        ax.plot(end_steps, end_values, color='green', linewidth=2, label='End Point Success Rate')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Success Rate')
        ax.set_title('Task Completion Success Rates')
        ax.grid(True)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "task_completion_rates.png"))
        plt.close(fig)
    
    print(f"All charts saved to {output_dir}")

def visualize_training_trajectory(log_dir: str, output_dir: str = 'visualizations'):
    """Visualize training trajectory
    
    Args:
        log_dir: Log directory
        output_dir: Output directory
    """
    try:
        from tensorboard.backend.event_processing import event_accumulator
        import matplotlib.pyplot as plt
        import os
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all event files
        event_files = []
        for root, dirs, files in os.walk(log_dir):
            for file in files:
                if file.startswith("events.out.tfevents"):
                    event_files.append(os.path.join(root, file))
        
        if not event_files:
            print(f"No tensorboard event files found in directory {log_dir}")
            return
        
        # Process each event file
        for event_file in event_files:
            # Load event
            ea = event_accumulator.EventAccumulator(event_file)
            ea.Reload()
            
            # Get all scalar tags
            tags = ea.Tags()['scalars']
            
            for tag in tags:
                print(f"Drawing {tag}...")
                
                # Get scalar events
                events = ea.Scalars(tag)
                steps = [event.step for event in events]
                values = [event.value for event in events]
                
                # Draw chart
                plt.figure(figsize=(10, 6))
                plt.plot(steps, values)
                plt.title(tag)
                plt.xlabel('Steps')
                plt.ylabel('Value')
                plt.grid(True)
                
                # Save chart
                clean_tag = tag.replace('/', '_')
                output_path = os.path.join(output_dir, f"{clean_tag}.png")
                plt.savefig(output_path)
                plt.close()
        
        print(f"All charts saved to {output_dir}")
    
    except ImportError:
        print("Missing tensorboard library, please install: pip install tensorboard")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='DHC Model Visualization Tool')
    subparsers = parser.add_subparsers(dest='command', help='commands')
    
    # Simulate model subcommand
    simulate_parser = subparsers.add_parser('simulate', help='Simulate and visualize model')
    simulate_parser.add_argument('--model_path', type=str, required=True, help='Model path')
    simulate_parser.add_argument('--num_agents', type=int, default=4, help='Number of agents')
    simulate_parser.add_argument('--map_size', type=int, default=40, help='Map size')
    simulate_parser.add_argument('--num_tasks', type=int, default=8, help='Number of tasks')
    simulate_parser.add_argument('--max_steps', type=int, default=100, help='Maximum steps')
    simulate_parser.add_argument('--seed', type=int, default=42, help='Random seed')
    simulate_parser.add_argument('--output', type=str, default='visualizations/simulation.gif', help='Output path')
    simulate_parser.add_argument('--fps', type=int, default=5, help='Frames per second')
    simulate_parser.add_argument('--hide_communication', action='store_true', help='Hide communication range')
    simulate_parser.add_argument('--hide_task_info', action='store_true', help='Hide task details')
    
    # Plot training results subcommand
    plot_parser = subparsers.add_parser('plot', help='Plot training results')
    plot_parser.add_argument('--log_dir', type=str, required=True, help='Log directory')
    plot_parser.add_argument('--output_dir', type=str, default='visualizations', help='Output directory')
    plot_parser.add_argument('--dpi', type=int, default=100, help='Resolution')
    
    args = parser.parse_args()
    
    if args.command == 'simulate':
        print("="*50)
        print("DHC Model Simulation Configuration:")
        print(f"Model path: {args.model_path}")
        print(f"Number of agents: {args.num_agents}")
        print(f"Map size: {args.map_size}")
        print(f"Number of tasks: {args.num_tasks}")
        print(f"Maximum steps: {args.max_steps}")
        print(f"Random seed: {args.seed}")
        print(f"Output path: {args.output}")
        print(f"FPS: {args.fps}")
        print(f"Show communication range: {not args.hide_communication}")
        print(f"Show task details: {not args.hide_task_info}")
        print("="*50)
        
        simulate_model(
            model_path=args.model_path,
            num_agents=args.num_agents,
            map_size=args.map_size,
            num_tasks=args.num_tasks,
            max_steps=args.max_steps,
            seed=args.seed,
            output_path=args.output,
            fps=args.fps,
            save_static_frames=True,
            static_frame_interval=20
        )
    elif args.command == 'plot':
        print("="*50)
        print("Training Results Plot Configuration:")
        print(f"Log directory: {args.log_dir}")
        print(f"Output directory: {args.output_dir}")
        print(f"Resolution: {args.dpi}")
        print("="*50)
        
        plot_training_results(
            log_dir=args.log_dir,
            output_dir=args.output_dir,
            dpi=args.dpi
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
