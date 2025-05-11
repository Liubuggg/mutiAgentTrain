import argparse
import numpy as np
import os
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Tuple
import yaml
from yaml.constructor import ConstructorError

# 尝试导入tensorboard
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    print("警告: tensorboard未安装，将使用假的SummaryWriter")
    print("可以通过运行以下命令安装tensorboard:")
    print("pip install tensorboard")
    
    class FakeSummaryWriter:
        def __init__(self, log_dir=None):
            self.log_dir = log_dir
            print(f"日志目录: {log_dir}")
            
        def add_scalar(self, *args, **kwargs):
            pass
            
        def close(self):
            pass
    SummaryWriter = FakeSummaryWriter

from pathfinding.models.dhc.model import Network, device
from pathfinding.models.dhc.agent import Agent
from pathfinding.models.dhc.task import Task, TaskPriority, TaskStatus
from pathfinding.models.dhc.task_generator import TaskGenerator
from pathfinding.models.dhc.task_allocator import TaskAllocator
from pathfinding.models.dhc.buffer import ExperienceBuffer
from pathfinding.settings import yaml_data as settings

# 配置信息
try:
    TRAIN_CONFIG = settings["dhc"]["train"]
except (KeyError, TypeError):
    print("警告：设置中不存在'dhc'或'dhc.train'键，使用默认值")
    TRAIN_CONFIG = {
        "epochs": 10000,
        "learning_rate": 1e-4,
        "batch_size": 64,
        "buffer_capacity": 10000,
        "update_target_interval": 100,
        "save_interval": 500,
        "eval_interval": 200
    }

try:
    LOG_CONFIG = settings["logging"]
except KeyError:
    print("警告：设置中不存在'logging'键，使用默认值")
    LOG_CONFIG = {
        "log_dir": "logs",
        "save_model_dir": "models"
    }

# 添加元组支持
def construct_python_tuple(self, node):
    return tuple(self.construct_sequence(node))

yaml.SafeLoader.add_constructor(u'tag:yaml.org,2002:python/tuple', construct_python_tuple)

def check_environment():
    """检查训练环境"""
    # 检查CUDA可用性
    if torch.cuda.is_available():
        print("CUDA可用，使用GPU训练")
    elif torch.backends.mps.is_available():
        print("MPS可用，使用Apple Silicon GPU训练")
    else:
        print("无可用GPU，使用CPU训练")
        
    # 检查必要的目录
    required_dirs = ['logs', 'models']
    for d in required_dirs:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"创建目录: {d}")
            
    # 检查必要的文件
    required_files = [
        'pathfinding/settings.py',
        'pathfinding/models/dhc/model.py',
        'pathfinding/models/dhc/buffer.py'
    ]
    for f in required_files:
        if not os.path.exists(f):
            print(f"错误: 缺少必要文件 {f}")
            return False
            
    return True

def setup_environment(seed: int = 42, log_dir: str = None):
    """设置环境和随机种子
    
    Args:
        seed: 随机种子
        log_dir: 日志根目录
    """
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # 设置日志目录
    if log_dir is None:
        log_dir = "logs"
    
    # 创建时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建日志和模型目录
    run_dir = os.path.join(log_dir, f"dhc_{timestamp}")
    model_dir = os.path.join("models", f"dhc_{timestamp}")
    
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    return run_dir, model_dir

from pathfinding.models.dhc.scenario import create_scenario, check_collision, get_valid_actions

def train(
    epochs: int = 1000,
    learning_rate: float = 3e-4,
    batch_size: int = 128,
    num_agents: int = 8,
    map_size: int = 40,
    num_tasks: int = 10,
    max_steps: int = 200,
    buffer_capacity: int = 100000,
    update_target_interval: int = 100,
    save_interval: int = 50,
    eval_interval: int = 50,
    seed: int = 42,
    log_dir: str = None
) -> dict:
    """训练Network模型
    Args:
        epochs: 训练轮数
        learning_rate: 学习率
        batch_size: 批次大小
        num_agents: 智能体数量
        map_size: 地图大小
        num_tasks: 任务数量
        max_steps: 最大步数
        buffer_capacity: 经验回放缓冲区容量
        update_target_interval: 更新目标网络的间隔
        save_interval: 保存模型的间隔
        eval_interval: 评估模型的间隔
        seed: 随机种子
        log_dir: 日志目录
        
    Returns:
        dict: 包含训练结果的字典，包括日志目录和训练历史数据
    """
    # 检查环境
    if not check_environment():
        print("环境检查失败，退出训练")
        return
    # 设置环境
    log_dir, model_dir = setup_environment(seed, log_dir)
    writer = SummaryWriter(log_dir=log_dir)
    # 创建模型
    model = Network()
    target_model = Network()
    target_model.load_state_dict(model.state_dict())
    
    model.to(device)
    target_model.to(device)
    
    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 创建学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.9)
    
    # 创建经验回放缓冲区
    buffer = ExperienceBuffer(capacity=buffer_capacity, batch_size=batch_size)
    
    # 训练循环
    total_steps = 0
    best_reward = float('-inf')
    episode_rewards = []
    episode_successes = []
    episode_lengths = []
    
    # 添加新的统计指标
    start_point_successes = []  # 记录到达任务起点的成功率
    end_point_successes = []    # 记录到达任务终点的成功率
    total_rewards_history = []  # 记录每个epoch的总奖励
    
    progress_bar = tqdm(range(epochs), desc="Training Progress")
    for epoch in progress_bar:
        # 创建任务分配器和场景
        task_allocator, agents, tasks = create_scenario(num_agents, map_size, num_tasks)
        
        # 初始化回合变量
        episode_reward = 0.0
        episode_steps = 0
        done = False
        hidden = None  # GRU 隐藏状态
        
        # 初始化任务完成状态统计
        tasks_started = 0      # 到达起点的任务数
        tasks_completed = 0    # 到达终点的任务数
        total_tasks = len(task_allocator.tasks)  # 总任务数
        
        # 收集经验
        while not done and episode_steps < max_steps:
            # 获取智能体和任务特征
            agent_obs = np.zeros((1, num_agents, 8, 9, 9))  # [B, N, C, H, W]
            agent_states = np.zeros((1, num_agents, 2))     # [B, N, 2] - battery和experience
            task_features = np.zeros((1, num_agents, 4))    # [B, N, 4] - 每个智能体的任务特征
            
            # 为每个智能体构建观察
            for i, agent in enumerate(agents):
                # 获取智能体周围的观察窗口
                x, y = agent.pos
                obs_radius = 4  # 观察半径
                
                # 提取观察窗口
                x_min = max(0, x - obs_radius)
                x_max = min(map_size, x + obs_radius + 1)
                y_min = max(0, y - obs_radius)
                y_max = min(map_size, y + obs_radius + 1)
                
                # 计算填充范围
                pad_x_min = obs_radius - (x - x_min)
                pad_x_max = obs_radius + 1 + (x_max - x - 1)
                pad_y_min = obs_radius - (y - y_min)
                pad_y_max = obs_radius + 1 + (y_max - y - 1)
                
                # 1. 障碍物层
                if hasattr(task_allocator, 'obstacle_map') and task_allocator.obstacle_map is not None:
                    agent_obs[0, i, 0, pad_x_min:pad_x_max, pad_y_min:pad_y_max] = \
                        task_allocator.obstacle_map[x_min:x_max, y_min:y_max]
                
                # 2. 智能体位置层
                agent_map = np.zeros((map_size, map_size))
                for other_agent in agents:
                    if other_agent.id != agent.id:
                        ox, oy = other_agent.pos
                        if x_min <= ox < x_max and y_min <= oy < y_max:
                            agent_map[ox, oy] = 1
                agent_obs[0, i, 1, pad_x_min:pad_x_max, pad_y_min:pad_y_max] = \
                    agent_map[x_min:x_max, y_min:y_max]
                
                # 3. 任务起点层
                task_start_map = np.zeros((map_size, map_size))
                # 4. 任务终点层
                task_goal_map = np.zeros((map_size, map_size))
                
                # 任务方向指示层（添加方向指引信息）
                task_direction_map = np.zeros((map_size, map_size, 2))
                
                if agent.current_task:
                    # 当有分配的任务时，标记任务起点和终点
                    start_x, start_y = agent.current_task.start_pos
                    goal_x, goal_y = agent.current_task.goal_pos
                    
                    # 增加起点和终点的可见性
                    # 如果起点/终点在观察窗口中，标记为1
                    if x_min <= start_x < x_max and y_min <= start_y < y_max:
                        task_start_map[start_x, start_y] = 1
                        
                        # 在起点周围添加高亮，帮助智能体更容易找到
                        for dx in range(-1, 2):
                            for dy in range(-1, 2):
                                sx, sy = start_x + dx, start_y + dy
                                if 0 <= sx < map_size and 0 <= sy < map_size and task_allocator.obstacle_map[sx, sy] == 0:
                                    if x_min <= sx < x_max and y_min <= sy < y_max:
                                        task_start_map[sx, sy] = 0.5  # 周围格子标记为0.5
                    
                    if x_min <= goal_x < x_max and y_min <= goal_y < y_max:
                        task_goal_map[goal_x, goal_y] = 1
                        
                        # 在终点周围添加高亮，帮助智能体更容易找到
                        for dx in range(-1, 2):
                            for dy in range(-1, 2):
                                gx, gy = goal_x + dx, goal_y + dy
                                if 0 <= gx < map_size and 0 <= gy < map_size and task_allocator.obstacle_map[gx, gy] == 0:
                                    if x_min <= gx < x_max and y_min <= gy < y_max:
                                        task_goal_map[gx, gy] = 0.5  # 周围格子标记为0.5
                    
                    # 即使任务不在观察范围内，也添加方向指引
                    # 计算从智能体到任务起点/终点的方向向量
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
                        
                    # 在智能体附近添加方向指引
                    for dx in range(-2, 3):
                        for dy in range(-2, 3):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < map_size and 0 <= ny < map_size and task_allocator.obstacle_map[nx, ny] == 0:
                                if x_min <= nx < x_max and y_min <= ny < y_max:
                                    intensity = 1.0 - (abs(dx) + abs(dy)) * 0.2  # 距离中心越远，指引越弱
                                    if intensity > 0:
                                        task_direction_map[nx, ny] = direction * intensity
                else:
                    # 如果没有分配任务，显示最近的可用任务
                    available_tasks = task_allocator.get_available_tasks(agent)
                    if available_tasks:
                        # 找到最近的任务
                        closest_task = None
                        min_distance = float('inf')
                        for task in available_tasks:
                            distance = np.linalg.norm(np.array(agent.pos) - np.array(task.start_pos))
                            if distance < min_distance:
                                min_distance = distance
                                closest_task = task
                        
                        if closest_task:
                            start_x, start_y = closest_task.start_pos
                            
                            # 如果最近任务的起点在观察窗口中，标记为0.7（比已分配任务弱）
                            if x_min <= start_x < x_max and y_min <= start_y < y_max:
                                task_start_map[start_x, start_y] = 0.7
                            
                            # 添加方向指引到最近的任务
                            direction = np.array(closest_task.start_pos) - np.array(agent.pos)
                            norm = np.linalg.norm(direction)
                            if norm > 0:
                                direction = direction / norm
                                
                            # 在智能体附近添加方向指引，强度较弱
                            for dx in range(-1, 2):
                                for dy in range(-1, 2):
                                    nx, ny = x + dx, y + dy
                                    if 0 <= nx < map_size and 0 <= ny < map_size and task_allocator.obstacle_map[nx, ny] == 0:
                                        if x_min <= nx < x_max and y_min <= ny < y_max:
                                            intensity = 0.5 - (abs(dx) + abs(dy)) * 0.1  # 强度较弱
                                            if intensity > 0:
                                                task_direction_map[nx, ny] = direction * intensity
                
                # 将这些层添加到观察空间
                agent_obs[0, i, 2, pad_x_min:pad_x_max, pad_y_min:pad_y_max] = task_start_map[x_min:x_max, y_min:y_max]
                agent_obs[0, i, 3, pad_x_min:pad_x_max, pad_y_min:pad_y_max] = task_goal_map[x_min:x_max, y_min:y_max]
                
                # 添加两个通道来表示方向向量的x和y分量
                agent_obs[0, i, 4, pad_x_min:pad_x_max, pad_y_min:pad_y_max] = task_direction_map[x_min:x_max, y_min:y_max, 0]  # x方向
                agent_obs[0, i, 5, pad_x_min:pad_x_max, pad_y_min:pad_y_max] = task_direction_map[x_min:x_max, y_min:y_max, 1]  # y方向
                
                # 智能体状态
                agent_states[0, i, 0] = agent.current_battery / agent.max_battery  # 归一化电量
                agent_states[0, i, 1] = agent.state['experience'] / 5.0  # 归一化经验值
                
                # 任务特征
                if agent.current_task:
                    task = agent.current_task
                    # 计算与任务起点和终点的距离
                    start_distance = np.linalg.norm(np.array(agent.pos) - np.array(task.start_pos))
                    goal_distance = np.linalg.norm(np.array(agent.pos) - np.array(task.goal_pos))
                    # 任务优先级
                    priority = task.priority.value / 4.0  # 归一化优先级
                    # 填充任务特征
                    task_features[0, i, 0] = start_distance / map_size  # 归一化距离
                    task_features[0, i, 1] = goal_distance / map_size
                    task_features[0, i, 2] = priority
                    task_features[0, i, 3] = 1.0  # 有任务标志
            
            # 生成位置信息
            positions = np.array([agent.pos for agent in agents]).reshape(1, num_agents, 2)
            
            # 创建通信掩码
            comm_mask = task_allocator.generate_communication_mask()
            comm_mask_tensor = torch.tensor(comm_mask, dtype=torch.float32).to(device)
            
            # 转换为张量
            obs_tensor = torch.tensor(agent_obs, dtype=torch.float32).to(device)
            pos_tensor = torch.tensor(positions, dtype=torch.float32).to(device)
            agent_states_tensor = torch.tensor(agent_states, dtype=torch.float32).to(device)
            task_features_tensor = torch.tensor(task_features, dtype=torch.float32).to(device)
            
            # 计算动作和任务分配
            with torch.no_grad():
                action_values, hidden, task_assignment = model.step(
                    obs=obs_tensor,
                    pos=pos_tensor,
                    tasks=task_features_tensor,
                    agent_states=agent_states_tensor, 
                    hidden=hidden,
                    comm_mask=comm_mask_tensor
                )
            
            # 转换为numpy数组
            action_values_np = action_values.detach().cpu().numpy()
            task_assignment_np = task_assignment.detach().cpu().numpy()
            
            # 根据任务评分分配任务
            actions = np.argmax(action_values_np, axis=-1)
            task_choices = np.argmax(task_assignment_np, axis=-1)
            
            # 初始化rewards列表
            rewards = [0.0] * len(agents)
            
            # 执行动作和任务分配
            for i, agent in enumerate(agents):
                # 执行动作
                agent_action = actions[0, i]  # 获取该智能体的动作
                
                # 计算新位置
                action = agent.ACTIONS[agent_action]
                new_pos = agent.pos + action
                
                # 检查是否会与其他智能体相撞
                collision = False
                for j, other_agent in enumerate(agents):
                    if i != j:
                        if np.array_equal(new_pos, other_agent.pos):
                            # 发现碰撞
                            collision = True
                            # 对相撞的智能体应用惩罚
                            rewards[i] -= 1.0  # 给当前智能体添加碰撞惩罚
                            break
                            
                if not collision:
                    success = agent.execute_action(agent_action)
                    
                    # 如果是尝试进入障碍物导致失败，确保标志已设置
                    # 这个标志将在task_allocator.update()中被利用来计算惩罚
                    if not success and hasattr(agent, 'attempted_invalid_move') and agent.attempted_invalid_move:
                        # 已经在agent.execute_action中设置了标志，奖励惩罚将在task_allocator.update()中计算
                        pass
                    
                    if not success:
                        # 如果执行失败，尝试随机动作
                        alternative_actions = [0, 1, 2, 3, 4]
                        random.shuffle(alternative_actions)
                        for alt_action in alternative_actions:
                            if alt_action != agent_action:
                                # 确保替代动作不会导致碰撞
                                alt_action_vector = agent.ACTIONS[alt_action]
                                alt_new_pos = agent.pos + alt_action_vector
                                
                                alt_collision = False
                                for j, other_agent in enumerate(agents):
                                    if i != j and np.array_equal(alt_new_pos, other_agent.pos):
                                        alt_collision = True
                                        break
                                        
                                if not alt_collision and agent.execute_action(alt_action):
                                    break
                else:
                    # 如果相撞情况下，选择停留原地
                    agent.execute_action(0)  # 0 表示不动
            
            # 检查任务到达起点和终点状态
            for agent in agents:
                if agent.current_task:
                    # 检查是否到达起点
                    if not agent.current_task.started and np.array_equal(agent.pos, agent.current_task.start_pos):
                        agent.current_task.started = True
                        tasks_started += 1
                        print(f"Agent {agent.id} reached start point of Task {agent.current_task.id}, reward: {10.0 * agent.current_task.priority.value:.1f}")
                    
                    # 检查是否到达终点
                    elif agent.current_task.started and np.array_equal(agent.pos, agent.current_task.goal_pos):
                        agent.current_task.status = TaskStatus.COMPLETED
                        tasks_completed += 1
                        print(f"Agent {agent.id} completed Task {agent.current_task.id}, reward: {50.0 * agent.current_task.priority.value:.1f}")
                        agent.current_task = None
            
            # 分配所有未分配的任务
            # 获取可用的任务和智能体
            available_tasks = [task for task in task_allocator.tasks.values() 
                              if task.status == TaskStatus.PENDING]
            available_agents = [agent for agent in agents if agent.current_task is None]
            
            # 如果有可用的任务和智能体，尝试分配
            if available_tasks and available_agents:
                # 为每个智能体计算与每个任务的得分
                assignment_scores = {}
                for agent in available_agents:
                    assignment_scores[agent.id] = {}
                    for task in available_tasks:
                        # 获取模型输出的任务评分
                        agent_idx = agent.id  # 假设agent.id对应了索引
                        task_idx = task.id  # 假设task.id对应了索引
                        if task_idx < task_assignment_np.shape[-1]:
                            score = task_assignment_np[0, agent_idx, task_idx]
                            assignment_scores[agent.id][task.id] = score
                            
                # 按得分从高到低为每个智能体分配任务
                for agent in available_agents:
                    if not assignment_scores[agent.id]:
                        continue
                        
                    # 找到得分最高的任务
                    best_task_id = max(assignment_scores[agent.id].items(), 
                                      key=lambda x: x[1])[0]
                    best_task = task_allocator.tasks[best_task_id]
                    
                    # 如果任务仍然可用，分配给智能体
                    if best_task.status == TaskStatus.PENDING:
                        task_allocator.assign_task(agent, best_task)
                        print(f"分配任务: Task {best_task.id} (优先级: {best_task.priority.name}) 分配给 Agent {agent.id}")
                        
                        # 从可用任务列表中移除
                        available_tasks.remove(best_task)
                        
                        # 如果没有更多可用任务，退出循环
                        if not available_tasks:
                            break
            
            # 更新任务分配器并获取奖励
            task_allocator_rewards = task_allocator.update()
            
            # 合并碰撞惩罚和任务分配器返回的奖励
            if len(task_allocator_rewards) > 0:
                for i in range(min(len(rewards), len(task_allocator_rewards))):
                    rewards[i] += task_allocator_rewards[i]
            
            # 计算平均奖励
            reward = np.mean(rewards) if len(rewards) > 0 else 0.0
            episode_reward += reward
            
            # 检查是否完成
            done = task_allocator.is_done()
            
            # 获取下一个状态
            # 更新观察和状态
            next_agent_obs = np.zeros_like(agent_obs)  # 应该由环境提供，这里简化处理
            next_agent_states = np.zeros_like(agent_states)
            
            # 更新智能体状态
            for i, agent in enumerate(agents):
                # 更新电池状态
                next_agent_states[0, i, 0] = agent.current_battery / agent.max_battery
                # 更新经验值
                next_agent_states[0, i, 1] = agent.state.get('experience', 0) / 10.0
            
            # 更新任务特征
            next_task_features = np.zeros_like(task_features)
            # 这里应该根据实际任务更新，但简化处理
            
            # 更新通信掩码
            next_comm_mask = task_allocator.generate_communication_mask()
            next_comm_mask_tensor = torch.tensor(next_comm_mask, dtype=torch.float32).to(device)
            
            # 存储经验
            buffer.add(
                observation=agent_obs,
                action=actions,
                reward=rewards,
                next_observation=next_agent_obs,
                done=np.array([done]),
                hidden_state=hidden.detach().cpu().numpy(),
                comm_mask=comm_mask,
                agent_states=agent_states,
                task_features=task_features
            )
            
            # 更新状态
            agent_obs = next_agent_obs
            agent_states = next_agent_states
            task_features = next_task_features
            comm_mask = next_comm_mask
            comm_mask_tensor = next_comm_mask_tensor
            
            episode_steps += 1
            total_steps += 1
            
            # 学习步骤
            if buffer.get_size() >= batch_size:
                # 从缓冲区采样
                batch = buffer.sample(batch_size)
                
                # 转换为张量
                obs_batch = torch.tensor(batch['observations'], dtype=torch.float32).to(device)
                action_batch = torch.tensor(batch['actions'], dtype=torch.int64).to(device)
                reward_batch = torch.tensor(batch['rewards'], dtype=torch.float32).to(device)
                next_obs_batch = torch.tensor(batch['next_observations'], dtype=torch.float32).to(device)
                done_batch = torch.tensor(batch['dones'], dtype=torch.float32).to(device)
                
                # 处理action_batch的形状问题
                if len(action_batch.shape) == 3:
                    # 如果形状是[batch_size, 1, num_agents]，将其转换为[batch_size, num_agents]
                    if action_batch.shape[1] == 1:
                        action_batch = action_batch.squeeze(1)
                    # 如果形状是[batch_size, num_agents, 1]，将其转换为[batch_size, num_agents]
                    elif action_batch.shape[2] == 1:
                        action_batch = action_batch.squeeze(2)
                
                # 处理隐藏状态和通信掩码
                hidden_batch = None
                comm_mask_batch = None
                agent_states_batch = None
                task_features_batch = None
                
                if 'hidden_states' in batch:
                    hidden_batch = torch.tensor(batch['hidden_states'], dtype=torch.float32).to(device)
                if 'comm_masks' in batch:
                    comm_mask_batch = torch.tensor(batch['comm_masks'], dtype=torch.float32).to(device)
                if 'agent_states' in batch:
                    agent_states_batch = torch.tensor(batch['agent_states'], dtype=torch.float32).to(device)
                if 'task_features' in batch:
                    task_features_batch = torch.tensor(batch['task_features'], dtype=torch.float32).to(device)
                
                # 计算Q值
                current_action_values, current_task_assignment, _ = model(
                    obs=obs_batch, 
                    steps=None, 
                    hidden=hidden_batch, 
                    comm_mask=comm_mask_batch,
                    tasks=task_features_batch,
                    agent_states=agent_states_batch
                )
                
                # 计算目标Q值
                with torch.no_grad():
                    next_action_values, next_task_assignment, _ = target_model(
                        obs=next_obs_batch, 
                        steps=None, 
                        hidden=hidden_batch, 
                        comm_mask=comm_mask_batch,
                        tasks=task_features_batch,
                        agent_states=agent_states_batch
                    )
                    
                    # 取最大Q值
                    max_next_action_values = next_action_values.max(dim=-1, keepdim=True)[0]
                    expected_action_values = reward_batch.unsqueeze(-1) + (0.99 * max_next_action_values * (1 - done_batch.unsqueeze(-1)))
                
                # 选择实际执行的动作的Q值
                # 根据不同的维度情况进行处理
                if len(current_action_values.shape) == 3:  # [batch_size, num_agents, num_actions]
                    # 最常见的情况
                    # 确保action_batch形状是[batch_size, num_agents]
                    if len(action_batch.shape) == 2:
                        indices = action_batch.unsqueeze(-1)  # [batch_size, num_agents, 1]
                        current_q_values = current_action_values.gather(2, indices)  # [batch_size, num_agents, 1]
                    else:
                        # 如果action_batch不是2D，调整形状
                        reshaped_actions = action_batch.reshape(current_action_values.size(0), -1)
                        indices = reshaped_actions.unsqueeze(-1)
                        current_q_values = current_action_values.gather(2, indices)
                        
                elif len(current_action_values.shape) == 4:  # [batch_size, num_agents, num_actions, 1]
                    # 移除最后一个维度
                    current_action_values_3d = current_action_values.squeeze(-1)  # [batch_size, num_agents, num_actions]
                    
                    # 确保action_batch形状是[batch_size, num_agents]
                    if len(action_batch.shape) == 2:
                        indices = action_batch.unsqueeze(-1)  # [batch_size, num_agents, 1]
                        current_q_values = current_action_values_3d.gather(2, indices).unsqueeze(-1)  # [batch_size, num_agents, 1, 1]
                    else:
                        # 如果action_batch不是2D，调整形状
                        reshaped_actions = action_batch.reshape(current_action_values.size(0), -1)
                        indices = reshaped_actions.unsqueeze(-1)
                        current_q_values = current_action_values_3d.gather(2, indices).unsqueeze(-1)
                        
                elif len(current_action_values.shape) == 2:  # [batch_size * num_agents, num_actions]
                    # 处理扁平化的情况
                    # 确保action_batch形状正确
                    flattened_actions = action_batch.flatten()  # [batch_size * num_agents]
                    indices = flattened_actions.unsqueeze(-1)  # [batch_size * num_agents, 1]
                    current_q_values = current_action_values.gather(1, indices)  # [batch_size * num_agents, 1]
                    
                    # 可能需要重新调整形状以匹配expected_action_values
                    if expected_action_values.dim() > 2:
                        batch_size = expected_action_values.size(0)
                        num_agents = expected_action_values.size(1)
                        current_q_values = current_q_values.view(batch_size, num_agents, -1)
                
                else:
                    # 处理其他不常见的情况
                    raise ValueError(f"意外的current_action_values维度: {current_action_values.shape}")
                
                # 确保expected_action_values维度匹配
                if expected_action_values.dim() != current_q_values.dim():
                    if expected_action_values.dim() < current_q_values.dim():
                        # 如果current_q_values维度更高，增加expected_action_values的维度
                        for _ in range(current_q_values.dim() - expected_action_values.dim()):
                            expected_action_values = expected_action_values.unsqueeze(-1)
                    else:
                        # 如果expected_action_values维度更高，增加current_q_values的维度
                        for _ in range(expected_action_values.dim() - current_q_values.dim()):
                            current_q_values = current_q_values.unsqueeze(-1)
                
                # 最终检查
                
                # 计算损失
                action_value_loss = nn.MSELoss()(current_q_values, expected_action_values)
                
                # 计算任务分配损失 (如果有实际标签的话)
                # 这里使用一个简单的示例
                task_assignment_loss = nn.CrossEntropyLoss()(
                    current_task_assignment.view(-1, current_task_assignment.size(-1)),
                    torch.argmax(next_task_assignment, dim=-1).view(-1)
                )
                
                # 总损失
                loss = action_value_loss + 0.1 * task_assignment_loss
                
                # 优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 更新目标网络
                if total_steps % update_target_interval == 0:
                    target_model.load_state_dict(model.state_dict())
                
                # 更新学习率
                if total_steps % 1000 == 0:
                    scheduler.step()
                
                # 记录训练信息
                if total_steps % 100 == 0 and len(episode_rewards) > 0:
                    writer.add_scalar('Training/AverageReward', np.nanmean(episode_rewards[-100:]), total_steps)
                    writer.add_scalar('Training/SuccessRate', np.nanmean(episode_successes[-100:]), total_steps)
                    writer.add_scalar('Training/AverageLength', np.nanmean(episode_lengths[-100:]), total_steps)
                    writer.add_scalar('Training/LearningRate', scheduler.get_last_lr()[0], total_steps)
                
                # 保存最佳模型
                if len(episode_rewards) >= 100:
                    current_avg_reward = np.nanmean(episode_rewards[-100:])
                    if not np.isnan(current_avg_reward) and current_avg_reward > best_reward:
                        best_reward = current_avg_reward
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_reward': best_reward,
                    }, os.path.join(model_dir, 'model_best.pt'))
                
                # 定期保存模型
                if epoch > 0 and epoch % save_interval == 0:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'episode_rewards': episode_rewards,
                        'episode_successes': episode_successes,
                        'episode_lengths': episode_lengths,
                    }, os.path.join(model_dir, f'model_epoch_{epoch}.pt'))
                
                # 更新进度条
                progress_bar.set_postfix({
                    'reward': f'{np.nanmean(episode_rewards) if len(episode_rewards) > 0 else 0.0:.2f}',
                    'success': f'{np.nanmean(episode_successes) if len(episode_successes) > 0 else 0.0:.2%}',
                    'length': f'{np.nanmean(episode_lengths) if len(episode_lengths) > 0 else 0.0:.1f}'
                })
            
        # 记录回合统计信息
        episode_rewards.append(episode_reward)
        episode_successes.append(float(not done))
        episode_lengths.append(episode_steps)
        
        # 记录任务完成情况统计
        if total_tasks > 0:
            start_success_rate = tasks_started / total_tasks
            end_success_rate = tasks_completed / total_tasks
            start_point_successes.append(start_success_rate)
            end_point_successes.append(end_success_rate)
            total_rewards_history.append(episode_reward)
            
            # 打印任务完成情况
            print(f"Epoch {epoch}: Start point success rate: {start_success_rate:.2%}, End point success rate: {end_success_rate:.2%}")
        
        # 计算移动平均
        if len(episode_rewards) > 100:
            episode_rewards.pop(0)
            episode_successes.pop(0)
            episode_lengths.pop(0)
            if len(start_point_successes) > 100:
                start_point_successes.pop(0)
                end_point_successes.pop(0)
                total_rewards_history.pop(0)
        
        # 记录统计信息
        if len(episode_rewards) > 0:
            writer.add_scalar('Reward/train', np.nanmean(episode_rewards), epoch)
            writer.add_scalar('Success/train', np.nanmean(episode_successes), epoch)
            writer.add_scalar('Length/train', np.nanmean(episode_lengths), epoch)
            # 添加新的统计指标到TensorBoard
            if len(start_point_successes) > 0:
                writer.add_scalar('Success/start_point', np.nanmean(start_point_successes), epoch)
                writer.add_scalar('Success/end_point', np.nanmean(end_point_successes), epoch)
                writer.add_scalar('Reward/total', np.nanmean(total_rewards_history), epoch)
        
        # 定期评估
        if (epoch + 1) % eval_interval == 0:
            # 临时保存当前模型
            eval_model_path = os.path.join(model_dir, 'model_eval_temp.pt')
            torch.save(model.state_dict(), eval_model_path)
            
            # 评估模型
            try:
                from pathfinding.models.dhc.evaluate import evaluate_model
                print(f"\n正在评估模型 (epoch {epoch+1})...")
                eval_stats = evaluate_model(
                    model_path=eval_model_path,
                    num_agents=num_agents,
                    map_size=map_size,
                    num_tasks=num_tasks,
                    num_episodes=5,  # 评估5个回合
                    render=False
                )
                
                # 记录评估指标
                for key, value in eval_stats.items():
                    writer.add_scalar(f'Eval/{key}', value, epoch)
                    
                print(f"评估结果: 平均奖励 = {eval_stats.get('avg_reward', 0):.2f}, "
                      f"成功率 = {eval_stats.get('success_rate', 0):.2%}\n")
            except Exception as e:
                print(f"评估中出错: {e}")
            finally:
                # 删除临时模型
                if os.path.exists(eval_model_path):
                    os.remove(eval_model_path)
    
    # 保存最终模型
    print("正在保存最终模型...")
    try:
        # 首先保存模型状态字典
        torch.save(model.state_dict(), os.path.join(model_dir, 'model_final.pt'))
        print(f"模型状态字典已保存到 {os.path.join(model_dir, 'model_final.pt')}")
        
        # 然后保存训练信息
        training_info = {
            'epoch': epochs,
            'episode_rewards': episode_rewards,
            'episode_successes': episode_successes,
            'episode_lengths': episode_lengths,
            'start_point_successes': start_point_successes,
            'end_point_successes': end_point_successes,
            'total_rewards_history': total_rewards_history
        }
        torch.save(training_info, os.path.join(model_dir, 'training_info.pt'))
        print("训练信息已保存")
    except Exception as e:
        print(f"保存模型时出错: {e}")
    
    # 关闭SummaryWriter
    writer.close()

    return {
        'log_dir': log_dir,
        'history': {
            'start_point_successes': start_point_successes,
            'end_point_successes': end_point_successes,
            'total_rewards': total_rewards_history
        }
    }

def train_dhc_model(
    num_agents: int = 4,
    map_size: int = 40,
    num_tasks: int = 8,
    max_steps: int = 200,
    epochs: int = 50,
    log_dir: str = 'logs',
    seed: int = 42,
    plot_results: bool = True
):
    """训练DHC模型的便捷接口
    Args:
        num_agents: 智能体数量
        map_size: 地图大小
        num_tasks: 任务数量
        max_steps: 最大步数
        epochs: 训练epoch数
        log_dir: 日志目录
        seed: 随机种子
        plot_results: 是否在训练后绘制结果图表
    """
    # 设置参数
    learning_rate = TRAIN_CONFIG.get("learning_rate", 1e-4)
    batch_size = TRAIN_CONFIG.get("batch_size", 64)
    buffer_capacity = TRAIN_CONFIG.get("buffer_capacity", 10000)
    update_target_interval = TRAIN_CONFIG.get("update_target_interval", 100)
    save_interval = TRAIN_CONFIG.get("save_interval", 500)
    eval_interval = TRAIN_CONFIG.get("eval_interval", 200)
    
    # 创建时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_log_dir = os.path.join(log_dir, f"dhc_{timestamp}")
    
    # 打印训练配置
    print("="*50)
    print("DHC Model Training Configuration:")
    print(f"Epochs: {epochs}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Batch Size: {batch_size}")
    print(f"Number of Agents: {num_agents}")
    print(f"Map Size: {map_size}")
    print(f"Number of Tasks: {num_tasks}")
    print(f"Max Steps: {max_steps}")
    print(f"Buffer Capacity: {buffer_capacity}")
    print(f"Target Network Update Interval: {update_target_interval}")
    print(f"Model Save Interval: {save_interval}")
    print(f"Evaluation Interval: {eval_interval}")
    print(f"Random Seed: {seed}")
    print(f"Log Directory: {current_log_dir}")
    print(f"Plot Results: {plot_results}")
    print("="*50)
    
    # 开始训练
    print("Starting training...")
    run_result = None
    try:
        run_result = train(
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            num_agents=num_agents,
            map_size=map_size,
            num_tasks=num_tasks,
            max_steps=max_steps,
            buffer_capacity=buffer_capacity,
            update_target_interval=update_target_interval,
            save_interval=save_interval,
            eval_interval=eval_interval,
            seed=seed,
            log_dir=log_dir
        )
        print("Training completed!")
    except Exception as e:
        print(f"Error during training: {e}")
    
    # 训练完成后绘制结果
    if plot_results and run_result:
        try:
            print("Generating performance charts...")
            from pathfinding.models.dhc.visualize import plot_training_results
            
            # 获取实际的日志目录
            actual_log_dir = run_result.get('log_dir', current_log_dir)
            
            # 创建可视化输出目录
            visualization_dir = os.path.join('visualizations', f"training_plots_{timestamp}")
            os.makedirs(visualization_dir, exist_ok=True)
            
            # 绘制训练结果
            plot_training_results(
                log_dir=actual_log_dir,
                output_dir=visualization_dir,
                dpi=120
            )
            
            # 生成直接可视化图表
            import matplotlib.pyplot as plt
            import numpy as np
            
            # 获取训练历史数据
            history = run_result.get('history', {})
            if history:
                # 创建图表
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), dpi=120)
                
                # 成功率图表
                epochs_range = range(1, len(history['start_point_successes'])+1)
                ax1.plot(epochs_range, history['start_point_successes'], 'b-', label='Start Point Success Rate')
                ax1.plot(epochs_range, history['end_point_successes'], 'g-', label='End Point Success Rate')
                
                # 设置图表属性
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Success Rate')
                ax1.set_title('Task Completion Success Rates')
                ax1.grid(True)
                ax1.legend()
                
                # 奖励图表
                ax2.plot(epochs_range, history['total_rewards'], 'r-', label='Total Reward')
                
                # 平滑后的奖励趋势线
                if len(history['total_rewards']) > 5:
                    window_size = min(10, len(history['total_rewards']) // 5)
                    weights = np.ones(window_size) / window_size
                    smoothed_rewards = np.convolve(history['total_rewards'], weights, mode='valid')
                    smoothed_range = range(window_size, len(history['total_rewards'])+1)
                    ax2.plot(smoothed_range, smoothed_rewards, 'k--', label=f'Moving Average (window={window_size})')
                
                # 设置图表属性
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Reward')
                ax2.set_title('Training Reward Trend')
                ax2.grid(True)
                ax2.legend()
                
                # 调整布局并保存
                plt.tight_layout()
                plt.savefig(os.path.join(visualization_dir, 'performance_summary.png'))
                plt.close()
                
                print(f"Performance charts saved to {visualization_dir}")
            else:
                print("No training history data available for direct plotting")
            
        except Exception as e:
            print(f"Error generating charts: {e}")
    
    return run_result

def main():
    """主函数"""
    # 创建命令行解析器
    parser = argparse.ArgumentParser(description='训练分布式分层协作模型')
    parser.add_argument('--epochs', type=int, default=10000, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--num_agents', type=int, default=4, help='智能体数量')
    parser.add_argument('--map_size', type=int, default=40, help='地图大小')
    parser.add_argument('--num_tasks', type=int, default=8, help='任务数量')
    parser.add_argument('--max_steps', type=int, default=200, help='最大步数')
    parser.add_argument('--buffer_capacity', type=int, default=10000, help='经验回放缓冲区容量')
    parser.add_argument('--update_target_interval', type=int, default=100, help='更新目标网络的间隔')
    parser.add_argument('--save_interval', type=int, default=500, help='保存模型的间隔')
    parser.add_argument('--eval_interval', type=int, default=200, help='评估模型的间隔')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--no_plot', action='store_true', help='训练后不绘制结果图表')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 打印训练配置
    print("="*50)
    print("DHC Model Training Configuration:")
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.lr}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Number of Agents: {args.num_agents}")
    print(f"Map Size: {args.map_size}")
    print(f"Number of Tasks: {args.num_tasks}")
    print(f"Max Steps: {args.max_steps}")
    print(f"Buffer Capacity: {args.buffer_capacity}")
    print(f"Target Network Update Interval: {args.update_target_interval}")
    print(f"Model Save Interval: {args.save_interval}")
    print(f"Evaluation Interval: {args.eval_interval}")
    print(f"Random Seed: {args.seed}")
    print(f"Plot Results: {not args.no_plot}")
    print("="*50)
    
    # 开始训练
    print("Starting training...")
    try:
        train_dhc_model(
            epochs=args.epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            num_agents=args.num_agents,
            map_size=args.map_size,
            num_tasks=args.num_tasks,
            max_steps=args.max_steps,
            buffer_capacity=args.buffer_capacity,
            update_target_interval=args.update_target_interval,
            save_interval=args.save_interval,
            eval_interval=args.eval_interval,
            seed=args.seed,
            plot_results=not args.no_plot
        )
        print("Training completed!")
    except Exception as e:
        print(f"Error during training: {e}")

if __name__ == "__main__":
    main()
