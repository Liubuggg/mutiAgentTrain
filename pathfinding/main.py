import argparse
import os
import sys
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import time

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathfinding.environment import Environment
from pathfinding.config.config import load_config
from pathfinding.models.dhc.train import train_dhc_model
from pathfinding.models.dhc.visualize import simulate_model, visualize_training_trajectory
from pathfinding.models.dhc.test import test_dhc_model
from pathfinding.models.dhc.model import Network

# 设置设备
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("使用 MPS 设备")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("使用 CUDA 设备")
else:
    device = torch.device("cpu")
    print("使用 CPU 设备")

def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='多智能体寻路与任务分配')
    
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'visualize', 'visualize_trajectory'],
                        help='运行模式')
    parser.add_argument('--model_path', type=str, default=None,
                        help='模型路径（测试或可视化模式）')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='日志目录（训练模式）')
    parser.add_argument('--num_agents', type=int, default=4,
                        help='智能体数量')
    parser.add_argument('--map_size', type=int, default=40,
                        help='地图大小')
    parser.add_argument('--num_tasks', type=int, default=8,
                        help='任务数量')
    parser.add_argument('--steps', type=int, default=200,
                        help='最大步数')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练epoch数')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--save_static_frames', action='store_true',
                        help='是否保存静态帧图像')
    parser.add_argument('--static_frame_interval', type=int, default=20,
                        help='静态帧保存间隔')
    parser.add_argument('--output_format', type=str, default='gif', choices=['gif', 'mp4'],
                        help='输出格式（gif或mp4）')
    
    return parser.parse_args()

def test_environment(num_agents=4, map_size=40, max_steps=100, render=True, seed=42):
    """测试环境"""
    # 设置随机种子
    np.random.seed(seed)
    
    # 创建环境
    env = Environment(num_agents=num_agents, map_length=map_size)
    env.reset()
    
    # 生成随机任务
    for _ in range(5):
        start_pos = np.random.randint(0, map_size, size=2)
        goal_pos = np.random.randint(0, map_size, size=2)
        env.add_task(start_pos, goal_pos)
    
    # 运行环境
    done = False
    total_rewards = np.zeros(num_agents)
    step = 0
    
    # 初始化可视化
    if render:
        plt.figure(figsize=(10, 10))
    
    while not done and step < max_steps:
        # 随机选择动作
        actions = np.random.randint(0, 5, size=num_agents)
        
        # 执行动作
        obs, rewards, done, _ = env.step(actions)
        
        # 累计奖励
        total_rewards += rewards
        
        # 显示环境
        if render:
            plt.clf()
            # 在这里添加自定义的可视化代码
            # 绘制地图
            plt.imshow(env.map, cmap='binary', alpha=0.3)
            
            # 绘制智能体
            for i, pos in enumerate(env.agents_pos):
                plt.scatter(pos[1], pos[0], c='blue', s=100)
                plt.text(pos[1], pos[0], f'A{i}', fontsize=12, ha='center', va='center')
                
            # 绘制目标
            for i, pos in enumerate(env.goals_pos):
                plt.scatter(pos[1], pos[0], c='green', s=100)
                plt.text(pos[1], pos[0], f'G{i}', fontsize=12, ha='center', va='center')
                
            # 绘制充电站
            for pos in env.charging_stations:
                plt.scatter(pos[1], pos[0], c='purple', s=100, marker='s')
                
            # 绘制任务
            for task_id, task in env.tasks.items():
                start_color = 'red' if task.status.value <= 2 else 'gray'  # PENDING or ASSIGNED
                goal_color = 'orange' if task.status.value == 3 else ('green' if task.status.value == 4 else 'gray')  # IN_PROGRESS or COMPLETED
                
                plt.scatter(task.start_pos[1], task.start_pos[0], c=start_color, s=80, marker='^')
                plt.scatter(task.goal_pos[1], task.goal_pos[0], c=goal_color, s=80, marker='v')
                plt.plot([task.start_pos[1], task.goal_pos[1]], [task.start_pos[0], task.goal_pos[0]], 
                         c='gray', alpha=0.5, linestyle='--')
            
            plt.title(f'Step: {step}, Rewards: {total_rewards.sum():.2f}')
            plt.pause(0.1)
        
        step += 1
        print(f"Step {step}: Rewards {rewards}, Total {total_rewards}")
    
    print(f"Done after {step} steps. Total rewards: {total_rewards}")
    return total_rewards

def visualize(args):
    """可视化模型
    
    Args:
        args: 命令行参数
    """
    print(f"正在加载模型: {args.model_path}")
    try:
        # 使用Network类的load_model方法加载模型
        model = Network.load_model(args.model_path)
        model.to(device)
        model.eval()
        
        print("模型加载成功，开始可视化...")
        simulate_model(
            model_path=args.model_path,
            num_agents=args.num_agents,
            map_size=args.map_size,
            num_tasks=args.num_tasks,
            max_steps=args.steps,
            seed=args.seed,
            save_static_frames=args.save_static_frames,
            static_frame_interval=args.static_frame_interval,
            output_format=args.output_format
        )
    except Exception as e:
        print(f"可视化过程出错: {e}")
        raise

def main():
    args = parse_args()
    config = load_config()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 根据模式执行
    if args.mode == 'train':
        # 训练模式
        print(f"开始训练 DHC 模型 ({args.epochs} epochs)")
        train_dhc_model(
            num_agents=args.num_agents,
            map_size=args.map_size,
            num_tasks=args.num_tasks,
            max_steps=args.steps,
            epochs=args.epochs,
            log_dir=args.log_dir,
            seed=args.seed
        )
    elif args.mode == 'visualize':
        # 可视化模式
        if not args.model_path:
            raise ValueError("在可视化模式下需要提供模型路径")
        
        visualize(args)
    elif args.mode == 'visualize_trajectory':
        # 可视化训练轨迹
        visualize_training_trajectory(args.log_dir)
    else:
        # 测试模式
        if not args.model_path:
            raise ValueError("在测试模式下需要提供模型路径")
        
        test_dhc_model(
            model_path=args.model_path,
            num_agents=args.num_agents,
            map_size=args.map_size,
            num_tasks=args.num_tasks,
            max_steps=args.steps,
            seed=args.seed
        )

if __name__ == "__main__":
    main()
