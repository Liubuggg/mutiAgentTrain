import os
import numpy as np
import torch
from tqdm import tqdm

from pathfinding.models.dhc.model import Network
from pathfinding.models.dhc.task_allocator import TaskAllocator
from pathfinding.models.dhc.agent import Agent
from pathfinding.models.dhc.task import Task
from pathfinding.models.dhc.task_generator import TaskGenerator

def test_dhc_model(
    model_path: str,
    num_agents: int = 4,
    map_size: int = 40,
    num_tasks: int = 8,
    max_steps: int = 200,
    seed: int = 42
):
    """测试DHC模型性能
    
    Args:
        model_path: 模型路径
        num_agents: 智能体数量
        map_size: 地图大小
        num_tasks: 任务数量
        max_steps: 最大步数
        seed: 随机种子
    """
    # 设置随机种子
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # 判断设备
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA for testing")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS for testing")
    else:
        device = torch.device("cpu")
        print("Using CPU for testing")
    
    # 加载模型
    try:
        model = Network.load(model_path)
        model.to(device)
        model.eval()
        print(f"Successfully loaded model: {model_path}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    # 创建场景
    task_allocator = TaskAllocator(communication_range=10)
    
    # 创建智能体
    agents = []
    for i in range(num_agents):
        # 随机位置
        pos = (np.random.randint(0, map_size), np.random.randint(0, map_size))
        agent = Agent(id=i, pos=pos, battery=100.0, experience=0.0)
        agents.append(agent)
        task_allocator.add_agent(agent)
    
    # 创建任务
    task_generator = TaskGenerator(map_size=map_size)
    for i in range(num_tasks):
        task = task_generator.generate_task()
        task_allocator.add_task(task)
    
    # 初始化模型隐藏状态
    hidden = None
    
    # 主循环
    step = 0
    done = False
    total_rewards = []
    
    print("Starting test...")
    progress_bar = tqdm(total=max_steps)
    
    while not done and step < max_steps:
        # 获取智能体观察和特征
        agent_obs = np.zeros((1, num_agents, 6, 9, 9))  # 假设观察形状为 [B, N, 6, 9, 9]
        agent_states = np.zeros((1, num_agents, 2))     # [B, N, 2] - battery和experience
        task_features = np.zeros((1, num_agents, 4))    # [B, N, 4] - 每个智能体的任务特征
        
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
        
        # 选择动作和任务
        actions = np.argmax(action_values_np, axis=-1)
        task_choices = np.argmax(task_assignment_np, axis=-1)
        
        # 更新任务分配器
        # 这里需要根据模型输出更新任务分配
        for agent_idx, agent in enumerate(agents):
            available_tasks = task_allocator.get_available_tasks(agent)
            if available_tasks and not agent.current_task:
                # 根据任务选择分配任务
                task_idx = task_choices[0, agent_idx] % len(available_tasks)
                task = available_tasks[task_idx]
                task_allocator.assign_task(agent, task)
        
        # 更新智能体位置
        for agent_idx, agent in enumerate(agents):
            action = actions[0, agent_idx]
            if action == 0:  # 上
                new_pos = (max(0, agent.pos[0] - 1), agent.pos[1])
            elif action == 1:  # 下
                new_pos = (min(map_size - 1, agent.pos[0] + 1), agent.pos[1])
            elif action == 2:  # 左
                new_pos = (agent.pos[0], max(0, agent.pos[1] - 1))
            elif action == 3:  # 右
                new_pos = (agent.pos[0], min(map_size - 1, agent.pos[1] + 1))
            else:  # 不动
                new_pos = agent.pos
            
            agent.move_to(new_pos)
        
        # 更新任务分配器
        rewards = task_allocator.update()
        total_rewards.append(np.mean(rewards))
        
        # 检查是否完成
        done = task_allocator.is_done()
        step += 1
        progress_bar.update(1)
        
        # 在测试过程中根据需要添加新任务
        if step % 20 == 0 and not done and len(task_allocator.tasks) < 2 * num_tasks:
            new_task = task_generator.generate_task()
            task_allocator.add_task(new_task)
            print(f"Step {step}: Added new task {new_task.id}")
    
    progress_bar.close()
    
    # 打印评估结果
    avg_reward = np.mean(total_rewards)
    completed_tasks = len([t for t in task_allocator.tasks.values() if t.is_completed()])
    completion_rate = completed_tasks / len(task_allocator.tasks) * 100
    
    print(f"\nTest Results:")
    print(f"Steps: {step}")
    print(f"Average reward per step: {avg_reward:.4f}")
    print(f"Completed tasks: {completed_tasks}/{len(task_allocator.tasks)} ({completion_rate:.2f}%)")
    
    return avg_reward, completion_rate 