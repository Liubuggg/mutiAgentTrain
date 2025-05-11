import argparse
import numpy as np
import os
import random
import time
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Tuple, Any

from pathfinding.models.dhc.model import DHCModel
from pathfinding.models.dhc.agent import Agent
from pathfinding.models.dhc.task import Task, TaskPriority
from pathfinding.models.dhc.task_generator import TaskGenerator
from pathfinding.models.dhc.task_allocator import TaskAllocator
from pathfinding.settings import yaml_data as settings

def setup_environment(seed: int = 42):
    """设置环境和随机种子
    
    Args:
        seed: 随机种子
    """
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # 创建结果目录
    os.makedirs("results", exist_ok=True)

def create_scenario(
    num_agents: int = 4, 
    map_size: int = 40, 
    num_tasks: int = 8,
    num_charging_stations: int = 2
) -> Tuple[TaskAllocator, List[Agent], List[Task]]:
    """创建评估场景
    
    Args:
        num_agents: 智能体数量
        map_size: 地图大小
        num_tasks: 任务数量
        num_charging_stations: 充电站数量
    
    Returns:
        Tuple[TaskAllocator, List[Agent], List[Task]]: 任务分配器、智能体列表和任务列表
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
            pos=pos,
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
        station_pos = np.random.randint(0, map_size, size=2)
        charging_stations.append(station_pos)
    task_allocator.set_charging_stations(charging_stations)
    
    # 创建任务
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

def evaluate_model(
    model_path: str,
    num_agents: int = 4,
    map_size: int = 40,
    num_tasks: int = 8,
    num_episodes: int = 10,
    max_steps: int = 200,
    render: bool = False,
    seed: int = 42
) -> Dict[str, Any]:
    """评估模型
    
    Args:
        model_path: 模型路径
        num_agents: 智能体数量
        map_size: 地图大小
        num_tasks: 任务数量
        num_episodes: 评估回合数
        max_steps: 每个回合的最大步数
        render: 是否渲染
        seed: 随机种子
    
    Returns:
        Dict[str, Any]: 评估结果
    """
    # 设置环境
    setup_environment(seed)
    
    # 加载模型
    model = DHCModel.load(model_path)
    model.eval()
    
    # 检查是否有可用CUDA设备或Apple MPS
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("使用CUDA评估")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("使用MPS评估")
    else:
        device = torch.device("cpu")
        print("使用CPU评估")
    
    model.to(device)
    
    # 准备记录统计信息
    stats = {
        'rewards': [],
        'success_rates': [],
        'task_completion_times': [],
        'battery_levels': [],
        'episode_lengths': []
    }
    
    # 评估多个回合
    for episode in tqdm(range(num_episodes), desc="评估进度"):
        # 创建场景
        task_allocator, agents, tasks = create_scenario(num_agents, map_size, num_tasks)
        
        # 初始化隐藏状态
        model.init_hidden(batch_size=1, num_agents=num_agents, hidden_dim=model.communication.hidden_dim)
        
        # 记录当前回合统计信息
        episode_reward = 0
        episode_steps = 0
        success = False
        task_completion_times = []
        
        # 回合循环
        done = False
        while not done and episode_steps < max_steps:
            # 获取智能体状态和任务特征
            agent_features = task_allocator.build_agent_features()
            task_features = task_allocator.build_task_features()
            
            # 转换为张量
            agent_features_tensor = torch.tensor(agent_features, dtype=torch.float32).to(device)
            task_features_tensor = torch.tensor(task_features, dtype=torch.float32).to(device)
            
            # 创建通信掩码
            comm_mask = task_allocator.generate_communication_mask()
            comm_mask_tensor = torch.tensor(comm_mask, dtype=torch.float32).to(device)
            
            # 计算任务评分
            with torch.no_grad():
                actions, _ = model.get_action(
                    obs=agent_features_tensor, 
                    comm_mask=comm_mask_tensor,
                    deterministic=True  # 使用确定性策略
                )
                task_scores = model.compute_task_scores(
                    agent_features=agent_features_tensor,
                    task_features=task_features_tensor
                )
            
            # 转换为numpy数组
            actions_np = actions.detach().cpu().numpy()
            task_scores_np = task_scores.detach().cpu().numpy()
            
            # 根据任务评分分配任务
            task_allocator.compute_compatibility_matrix(task_scores_np)
            task_allocator.execute_assignments()
            
            # 更新任务分配器
            rewards = task_allocator.update()
            reward = np.mean(rewards)
            episode_reward += reward
            
            # 检查是否完成
            done = task_allocator.is_done()
            if done:
                success = True
            
            # 收集完成的任务
            for task in task_allocator.tasks:
                if task.status == TaskStatus.COMPLETED and task.completion_time is not None:
                    task_completion_times.append(task.completion_time - task.start_time)
            
            episode_steps += 1
            
            # 渲染
            if render:
                render_scenario(task_allocator, agents, tasks, episode_steps)
        
        # 记录回合统计信息
        stats['rewards'].append(episode_reward)
        stats['success_rates'].append(float(success))
        stats['episode_lengths'].append(episode_steps)
        
        # 记录任务完成时间
        if task_completion_times:
            stats['task_completion_times'].append(np.mean(task_completion_times))
        
        # 记录电池电量
        battery_levels = [agent.current_battery for agent in agents]
        stats['battery_levels'].append(np.mean(battery_levels))
    
    # 计算平均统计信息
    results = {
        'mean_reward': np.mean(stats['rewards']),
        'mean_success_rate': np.mean(stats['success_rates']),
        'mean_episode_length': np.mean(stats['episode_lengths']),
        'mean_battery_level': np.mean(stats['battery_levels'])
    }
    
    if stats['task_completion_times']:
        results['mean_task_completion_time'] = np.mean(stats['task_completion_times'])
    
    return results

def render_scenario(task_allocator, agents, tasks, step):
    """渲染场景
    
    Args:
        task_allocator: 任务分配器
        agents: 智能体列表
        tasks: 任务列表
        step: 当前步数
    """
    # 创建画布
    plt.figure(figsize=(10, 10))
    
    # 绘制地图边界
    plt.xlim(0, task_allocator.map_size)
    plt.ylim(0, task_allocator.map_size)
    
    # 绘制智能体
    for agent in agents:
        x, y = agent.pos
        plt.scatter(x, y, c='blue', s=100, label=f'Agent {agent.id}' if agent.id == 0 else "")
        plt.text(x, y + 0.5, f'{agent.id}', ha='center')
        
        # 绘制电池电量
        plt.text(x, y - 1.0, f'{agent.current_battery:.0f}%', ha='center', fontsize=8)
        
        # 如果有任务，绘制连线
        if agent.current_task:
            task_goal = agent.current_task.goal_pos
            plt.plot([x, task_goal[0]], [y, task_goal[1]], 'b--', alpha=0.3)
    
    # 绘制任务
    for task in tasks:
        start_x, start_y = task.start_pos
        goal_x, goal_y = task.goal_pos
        
        # 根据任务状态选择颜色
        if task.status == TaskStatus.COMPLETED:
            color = 'green'
        elif task.status == TaskStatus.IN_PROGRESS:
            color = 'orange'
        elif task.status == TaskStatus.PENDING:
            color = 'red'
        else:
            color = 'gray'
        
        plt.scatter(start_x, start_y, c=color, marker='s', s=50)
        plt.scatter(goal_x, goal_y, c=color, marker='*', s=100)
        plt.plot([start_x, goal_x], [start_y, goal_y], c=color, alpha=0.5)
    
    # 绘制充电站
    for station in task_allocator.charging_stations:
        x, y = station
        plt.scatter(x, y, c='purple', marker='^', s=100)
    
    # 设置标题和图例
    plt.title(f'步数: {step}')
    plt.grid(True)
    plt.legend()
    
    # 保存图片
    os.makedirs('frames', exist_ok=True)
    plt.savefig(f'frames/step_{step:04d}.png')
    plt.close()

def main():
    """主函数"""
    # 创建命令行解析器
    parser = argparse.ArgumentParser(description='评估分布式分层协作模型')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--num_agents', type=int, default=4, help='智能体数量')
    parser.add_argument('--map_size', type=int, default=40, help='地图大小')
    parser.add_argument('--num_tasks', type=int, default=8, help='任务数量')
    parser.add_argument('--num_episodes', type=int, default=10, help='评估回合数')
    parser.add_argument('--max_steps', type=int, default=200, help='每个回合的最大步数')
    parser.add_argument('--render', action='store_true', help='是否渲染')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 打印评估配置
    print("="*50)
    print("DHC模型评估配置:")
    print(f"模型路径: {args.model_path}")
    print(f"智能体数量: {args.num_agents}")
    print(f"地图大小: {args.map_size}")
    print(f"任务数量: {args.num_tasks}")
    print(f"评估回合数: {args.num_episodes}")
    print(f"每个回合的最大步数: {args.max_steps}")
    print(f"是否渲染: {args.render}")
    print(f"随机种子: {args.seed}")
    print("="*50)
    
    # 开始评估
    print("开始评估...")
    try:
        results = evaluate_model(
            model_path=args.model_path,
            num_agents=args.num_agents,
            map_size=args.map_size,
            num_tasks=args.num_tasks,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            render=args.render,
            seed=args.seed
        )
        
        # 打印评估结果
        print("="*50)
        print("评估结果:")
        for key, value in results.items():
            print(f"{key}: {value:.4f}")
        print("="*50)
        
        # 将结果保存到文件
        with open(os.path.join("results", "evaluation_results.txt"), "w") as f:
            f.write("评估结果:\n")
            for key, value in results.items():
                f.write(f"{key}: {value:.4f}\n")
        
        print("评估完成!")
        
        # 如果渲染了场景，生成GIF
        if args.render:
            try:
                import imageio
                import glob
                
                print("正在生成GIF...")
                frames = []
                frame_files = sorted(glob.glob("frames/step_*.png"))
                for frame_file in frame_files:
                    frames.append(imageio.imread(frame_file))
                
                imageio.mimsave(os.path.join("results", "evaluation.gif"), frames, fps=5)
                print(f"GIF已保存到 results/evaluation.gif")
            except ImportError:
                print("未安装imageio，无法生成GIF。请使用pip install imageio安装。")
    
    except Exception as e:
        print(f"评估过程中发生错误: {e}")

if __name__ == "__main__":
    main()
