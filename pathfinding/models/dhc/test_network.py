import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import random

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from pathfinding.models.dhc.model import Network
from pathfinding.models.dhc.agent import Agent
from pathfinding.models.dhc.task import Task, TaskPriority, TaskStatus
from pathfinding.models.dhc.task_allocator import TaskAllocator
from pathfinding.models.dhc.task_generator import TaskGenerator

def setup_environment(seed=42):
    """设置随机种子，确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_model(model_path):
    """加载训练好的模型
    
    Args:
        model_path: 模型文件路径
        
    Returns:
        加载的模型
    """
    print(f"正在加载模型: {model_path}")
    
    # 初始化模型
    model = Network()
    
    # 加载模型参数
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 检查设备
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("使用CUDA进行评估")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("使用MPS进行评估")
    else:
        device = torch.device("cpu")
        print("使用CPU进行评估")
    
    model.to(device)
    model.eval()
    
    print("模型加载成功！")
    print(f"训练轮次: {checkpoint['epoch']}")
    print(f"总步数: {checkpoint['total_steps']}")
    
    return model, device

def create_scenario(num_agents=4, map_size=40, num_tasks=8, num_charging_stations=2):
    """创建测试场景
    
    Args:
        num_agents: 智能体数量
        map_size: 地图大小
        num_tasks: 任务数量
        num_charging_stations: 充电站数量
        
    Returns:
        任务分配器、智能体列表和任务列表
    """
    # 创建任务分配器
    task_allocator = TaskAllocator(num_agents, map_size)
    
    # 创建智能体
    agents = []
    for i in range(num_agents):
        # 随机初始位置
        pos = np.random.randint(0, map_size, size=2)
        # 随机初始电量 (60-100%)
        battery = random.uniform(60, 100)
        # 创建智能体
        agent = Agent(
            id=i,
            pos=tuple(pos),
            max_battery=100,
            communication_range=10.0
        )
        # 设置初始电量
        agent.current_battery = battery
        agents.append(agent)
        task_allocator.add_agent(agent)
    
    # 创建充电站
    charging_stations = []
    for _ in range(num_charging_stations):
        station_pos = tuple(np.random.randint(0, map_size, size=2))
        charging_stations.append(station_pos)
    task_allocator.set_charging_stations(charging_stations)
    
    # 创建任务生成器
    task_generator = TaskGenerator(
        map_size=map_size,
        num_agents=num_agents,
        max_tasks=num_tasks
    )
    
    # 生成初始任务
    tasks = []
    for _ in range(num_tasks):
        task = task_generator.generate_task()
        tasks.append(task)
        task_allocator.add_task(task)
    
    return task_allocator, agents, tasks

def evaluate_model(model, device, num_agents=4, map_size=40, num_tasks=8, max_steps=100):
    """评估模型性能
    
    Args:
        model: 训练好的模型
        device: 计算设备
        num_agents: 智能体数量
        map_size: 地图大小
        num_tasks: 任务数量
        max_steps: 最大步数
        
    Returns:
        评估结果
    """
    # 创建场景
    task_allocator, agents, tasks = create_scenario(num_agents, map_size, num_tasks)
    
    # 初始化隐藏状态
    hidden = None
    
    # 记录统计信息
    episode_reward = 0
    episode_steps = 0
    task_completion_times = []
    completed_tasks = 0
    
    print(f"\n开始评估模型，智能体数量: {num_agents}，任务数量: {num_tasks}")
    print("-" * 50)
    
    # 模拟循环
    done = False
    while not done and episode_steps < max_steps:
        # 获取智能体状态和任务特征
        agent_features = task_allocator.build_agent_features()
        task_features = task_allocator.build_task_features()
        
        # 转换为张量
        agent_features_tensor = torch.tensor(agent_features, dtype=torch.float32).unsqueeze(0).to(device)  # 添加批次维度
        task_features_tensor = torch.tensor(task_features, dtype=torch.float32).unsqueeze(0).to(device)  # 添加批次维度
        
        # 创建通信掩码
        comm_mask = task_allocator.generate_communication_mask()
        comm_mask_tensor = torch.tensor(comm_mask, dtype=torch.float32).unsqueeze(0).to(device)  # 添加批次维度
        
        # 计算任务评分和动作
        with torch.no_grad():
            # 从智能体特征中提取位置和状态
            pos = agent_features_tensor[:, :, :2].clone()  # 提取位置
            states = agent_features_tensor[:, :, 2:4].clone()  # 提取状态 (例如电量)
            
            # 使用网络进行推理
            actions, hidden, task_scores = model.step(
                obs=agent_features_tensor,  # 观察
                pos=pos,                    # 位置
                tasks=task_features_tensor, # 任务
                agent_states=states,        # 智能体状态
                hidden=hidden,              # 隐藏状态
                comm_mask=comm_mask_tensor  # 通信掩码
            )
            
            # 选择动作
            actions_np = actions.squeeze(0).cpu().numpy()  # 移除批次维度
            actions_indices = np.argmax(actions_np, axis=1)
            
            # 任务分配
            task_scores_np = task_scores.squeeze(0).cpu().numpy()  # 移除批次维度
        
        # 更新任务分配
        task_allocator.compute_compatibility_matrix(task_scores_np)
        task_allocator.execute_assignments()
        
        # 执行动作
        rewards = task_allocator.update()
        reward = np.mean(rewards)
        episode_reward += reward
        
        # 检查任务完成情况
        now_completed = 0
        for task in task_allocator.tasks:
            if task.status == TaskStatus.COMPLETED:
                now_completed += 1
                if task.completion_time and task.start_time:
                    task_completion_times.append(task.completion_time - task.start_time)
        
        # 检查是否有新完成的任务
        if now_completed > completed_tasks:
            print(f"步骤 {episode_steps}: 完成了 {now_completed - completed_tasks} 个新任务")
            completed_tasks = now_completed
        
        # 打印智能体位置和电量
        if episode_steps % 10 == 0:
            print(f"\n步骤 {episode_steps}:")
            for i, agent in enumerate(agents):
                print(f"智能体 {i}: 位置 {agent.pos}, 电量 {agent.current_battery:.1f}%, " + 
                     (f"执行任务 {agent.current_task.id}" if agent.current_task else "空闲"))
        
        # 检查是否完成所有任务
        done = all(task.status == TaskStatus.COMPLETED for task in task_allocator.tasks)
        
        episode_steps += 1
    
    # 计算结果
    success_rate = completed_tasks / num_tasks
    mean_completion_time = np.mean(task_completion_times) if task_completion_times else 0
    
    print("\n" + "=" * 50)
    print("评估结果:")
    print(f"总步数: {episode_steps}")
    print(f"总奖励: {episode_reward:.2f}")
    print(f"完成任务数: {completed_tasks}/{num_tasks} ({success_rate:.1%})")
    print(f"平均任务完成时间: {mean_completion_time:.2f} 步")
    print(f"平均智能体电量: {np.mean([agent.current_battery for agent in agents]):.1f}%")
    print("=" * 50)
    
    return {
        'steps': episode_steps,
        'reward': episode_reward,
        'success_rate': success_rate,
        'completion_time': mean_completion_time,
        'battery_level': np.mean([agent.current_battery for agent in agents])
    }

if __name__ == "__main__":
    # 设置环境
    setup_environment(seed=42)
    
    # 加载模型
    model_path = "models/model_final.pt"
    model, device = load_model(model_path)
    
    # 测试模型
    results = evaluate_model(model, device)
