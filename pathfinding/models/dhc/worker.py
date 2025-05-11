import numpy as np
import os
import random
import ray
import threading
import time
import torch
import torch.nn as nn
from copy import deepcopy
from torch.amp import GradScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from typing import Tuple
from scipy.optimize import linear_sum_assignment

# 延迟导入Environment，避免循环导入
# from pathfinding.environment import Environment
from pathfinding.models.dhc.buffer import SumTree, LocalBuffer
from pathfinding.models.dhc.model import Network
from pathfinding.settings import yaml_data as settings

WRK_CONFIG = settings["dhc"]["worker"]
GENERAL_CONFIG = settings["dhc"]

if torch.cuda.is_available():
    from torch.amp import autocast
else:
    from torch.amp import autocast

@ray.remote(num_cpus=1)
class GlobalBuffer:
    def __init__(
        self,
        episode_capacity=WRK_CONFIG["episode_capacity"],
        local_buffer_capacity=GENERAL_CONFIG["max_episode_length"],
        init_env_settings=WRK_CONFIG["init_env_settings"],
        max_comm_agents=WRK_CONFIG["max_comm_agents"],
        alpha=WRK_CONFIG["prioritized_replay_alpha"],
        beta=WRK_CONFIG["prioritized_replay_beta"],
        max_num_agents=GENERAL_CONFIG["max_num_agents"],
    ):

        self.capacity = episode_capacity
        self.local_buffer_capacity = local_buffer_capacity
        self.size = 0
        self.ptr = 0

        # prioritized experience replay
        self.priority_tree = SumTree(episode_capacity * local_buffer_capacity)
        self.alpha = alpha
        self.beta = beta

        self.counter = 0
        self.batched_data = []
        self.stat_dict = {init_env_settings: []}
        self.lock = threading.Lock()
        self.env_settings_set = ray.put([init_env_settings])

        self.obs_buf = np.zeros(
            (
                (local_buffer_capacity + 1) * episode_capacity,
                max_num_agents,
                *GENERAL_CONFIG["observation_shape"],
            ),
            dtype=np.bool,
        )
        self.act_buf = np.zeros(
            (local_buffer_capacity * episode_capacity), dtype=np.uint8
        )
        self.rew_buf = np.zeros(
            (local_buffer_capacity * episode_capacity), dtype=np.float32
        )
        self.hid_buf = np.zeros(
            (
                local_buffer_capacity * episode_capacity,
                max_num_agents,
                GENERAL_CONFIG["hidden_dim"],
            ),
            dtype=np.float32,
        )
        self.done_buf = np.zeros(episode_capacity, dtype=np.bool)
        self.size_buf = np.zeros(episode_capacity, dtype=np.uint)
        self.comm_mask_buf = np.zeros(
            (
                (local_buffer_capacity + 1) * episode_capacity,
                max_num_agents,
                max_num_agents,
            ),
            dtype=np.bool,
        )

        self.last_print_time = time.time()
        self.print_interval = 15.0  # 每15秒最多打印一次状态

    def __len__(self):
        return self.size

    def run(self):
        self.background_thread = threading.Thread(target=self.prepare_data, daemon=True)
        self.background_thread.start()

    def prepare_data(self):
        while True:
            if len(self.batched_data) <= 4:
                data = self.sample_batch(GENERAL_CONFIG["batch_size"])
                data_id = ray.put(data)
                self.batched_data.append(data_id)
            else:
                time.sleep(0.1)

    def get_data(self):

        if len(self.batched_data) == 0:
            print("no prepared data")
            data = self.sample_batch(GENERAL_CONFIG["batch_size"])
            data_id = ray.put(data)
            return data_id
        else:
            return self.batched_data.pop(0)

    def add(self, data: Tuple):
        """
        添加数据到全局缓冲区
        Args:
            data: 包含经验数据的元组，具体项目为:
                - obs_buf: 观察数据
                - act_buf: 动作数据
                - rew_buf: 奖励数据
                - hid_buf: 隐藏状态数据
                - comm_mask_buf: 通信掩码数据
                - 如果有任务数据，则包含:
                  - task_features: 任务特征
                  - agent_states: 智能体状态
                  - task_assignments: 任务分配情况
        """
        with self.lock:
            # 提取基本数据
            obs_buf, act_buf, rew_buf, hid_buf, comm_mask_buf = data[:5]
            
            # 获取此批次的大小
            batch_size = obs_buf.shape[0] - 1  # 观察数组有额外的最终状态
            num_agents = obs_buf.shape[1]
            
            # 计算索引
            idx_start = self.ptr * self.local_buffer_capacity
            idx_end = idx_start + batch_size
            idxes = np.arange(idx_start, idx_end)
            
            # 更新缓冲区大小
            self.size -= self.size_buf[self.ptr].item() if self.size_buf[self.ptr].item() > 0 else 0
            self.size += batch_size
            self.counter += batch_size
            
            # 确保priorities形状与idxes匹配
            priorities = np.ones(len(idxes))
            self.priority_tree.batch_update(idxes, priorities ** self.alpha)
            
            # 更新缓冲区数据
            # 注意：使用idx_start和batch_size来确保索引正确
            self.obs_buf[idx_start : idx_start + batch_size + 1, :num_agents] = obs_buf
            self.act_buf[idx_start : idx_start + batch_size] = act_buf
            self.rew_buf[idx_start : idx_start + batch_size] = rew_buf
            self.hid_buf[idx_start : idx_start + batch_size, :num_agents] = hid_buf
            self.done_buf[self.ptr] = True  # 假设每批次最后都是终止状态
            self.size_buf[self.ptr] = batch_size
            
            # 更新通信掩码
            self.comm_mask_buf[idx_start : idx_start + batch_size + 1] = 0
            self.comm_mask_buf[idx_start : idx_start + batch_size + 1, :num_agents, :num_agents] = comm_mask_buf
            
            # 更新指针
            self.ptr = (self.ptr + 1) % self.capacity
            
            # 记录统计信息
            env_setting = (num_agents, self.map_size)
            if env_setting in self.stat_dict:
                self.stat_dict[env_setting].append(True)  # 假设成功
                if len(self.stat_dict[env_setting]) > 200:
                    self.stat_dict[env_setting].pop(0)

    def sample_batch(self, batch_size: int) -> Tuple:

        b_obs, b_action, b_reward, b_done, b_steps, b_seq_len, b_comm_mask = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        idxes, priorities = [], []
        b_hidden = []

        with self.lock:

            idxes, priorities = self.priority_tree.batch_sample(batch_size)
            global_idxes = idxes // self.local_buffer_capacity
            local_idxes = idxes % self.local_buffer_capacity

            for idx, global_idx, local_idx in zip(
                idxes.tolist(), global_idxes.tolist(), local_idxes.tolist()
            ):

                assert (
                    local_idx < self.size_buf[global_idx]
                ), f"index is {local_idx} but size is {self.size_buf[global_idx]}"

                conf_seq_len = WRK_CONFIG["seq_len"]
                fwd_steps = WRK_CONFIG["forward_steps"]

                steps = min(fwd_steps, (self.size_buf[global_idx].item() - local_idx))
                seq_len = min(local_idx + 1, conf_seq_len)

                if local_idx < conf_seq_len - 1:
                    obs = self.obs_buf[
                        global_idx * (self.local_buffer_capacity + 1) : idx
                        + global_idx
                        + 1
                        + steps
                    ]
                    comm_mask = self.comm_mask_buf[
                        global_idx * (self.local_buffer_capacity + 1) : idx
                        + global_idx
                        + 1
                        + steps
                    ]
                    hidden = np.zeros(
                        (
                            GENERAL_CONFIG["max_num_agents"],
                            GENERAL_CONFIG["hidden_dim"],
                        ),
                        dtype=np.float32,
                    )
                elif local_idx == conf_seq_len - 1:
                    obs = self.obs_buf[
                        idx
                        + global_idx
                        + 1
                        - conf_seq_len : idx
                        + global_idx
                        + 1
                        + steps
                    ]
                    comm_mask = self.comm_mask_buf[
                        global_idx * (self.local_buffer_capacity + 1) : idx
                        + global_idx
                        + 1
                        + steps
                    ]
                    hidden = np.zeros(
                        (
                            GENERAL_CONFIG["max_num_agents"],
                            GENERAL_CONFIG["hidden_dim"],
                        ),
                        dtype=np.float32,
                    )
                else:
                    obs = self.obs_buf[
                        idx
                        + global_idx
                        + 1
                        - conf_seq_len : idx
                        + global_idx
                        + 1
                        + steps
                    ]
                    comm_mask = self.comm_mask_buf[
                        idx
                        + global_idx
                        + 1
                        - conf_seq_len : idx
                        + global_idx
                        + 1
                        + steps
                    ]
                    hidden = self.hid_buf[idx - conf_seq_len]

                if obs.shape[0] < conf_seq_len + fwd_steps:
                    pad_len = conf_seq_len + fwd_steps - obs.shape[0]
                    obs = np.pad(obs, ((0, pad_len), (0, 0), (0, 0), (0, 0), (0, 0)))
                    comm_mask = np.pad(comm_mask, ((0, pad_len), (0, 0), (0, 0)))

                action = self.act_buf[idx]
                reward = 0
                for i in range(steps):
                    reward += self.rew_buf[idx + i] * 0.99**i

                if (
                    self.done_buf[global_idx]
                    and local_idx >= self.size_buf[global_idx] - fwd_steps
                ):
                    done = True
                else:
                    done = False

                b_obs.append(obs)
                b_action.append(action)
                b_reward.append(reward)
                b_done.append(done)
                b_steps.append(steps)
                b_seq_len.append(seq_len)
                b_hidden.append(hidden)
                b_comm_mask.append(comm_mask)

            # importance sampling weight
            min_p = np.min(priorities)
            weights = np.power(priorities / min_p, -self.beta).astype(np.float32)

            data = (
                torch.from_numpy(np.stack(b_obs).astype(np.float32)),
                torch.LongTensor(b_action).unsqueeze(1),
                torch.FloatTensor(b_reward).unsqueeze(1),
                torch.FloatTensor(b_done).unsqueeze(1),
                torch.FloatTensor(b_steps).unsqueeze(1),
                torch.LongTensor(b_seq_len),
                torch.from_numpy(np.concatenate(b_hidden).astype(np.float32)),
                torch.from_numpy(np.stack(b_comm_mask)),
                idxes,
                torch.from_numpy(weights).unsqueeze(1),
                self.ptr,
            )

            return data

    def update_priorities(
        self, idxes: np.ndarray, priorities: np.ndarray, old_ptr: int
    ):
        """Update priorities of sampled transitions"""
        with self.lock:

            # discard the indices that already been discarded in replay buffer during training
            if self.ptr > old_ptr:
                # range from [old_ptr, self.ptr)
                mask = (idxes < old_ptr * self.local_buffer_capacity) | (
                    idxes >= self.ptr * self.local_buffer_capacity
                )
                idxes = idxes[mask]
                priorities = priorities[mask]
            elif self.ptr < old_ptr:
                # range from [0, self.ptr) & [old_ptr, self,capacity)
                mask = (idxes < old_ptr * self.local_buffer_capacity) & (
                    idxes >= self.ptr * self.local_buffer_capacity
                )
                idxes = idxes[mask]
                priorities = priorities[mask]

            self.priority_tree.batch_update(
                np.copy(idxes), np.copy(priorities) ** self.alpha
            )

    def stats(self, interval: int):
        current_time = time.time()
        stats = {
            'size': self.size,
            'update_speed': self.counter / interval if interval > 0 else 0
        }
        
        if current_time - self.last_print_time >= self.print_interval:
            if self.size == 0:
                print("(GlobalBuffer) 缓冲区为空，等待数据...")
            else:
                print(f"(GlobalBuffer) 当前大小: {self.size:,} | 更新速度: {stats['update_speed']:.2f}/s")
            self.last_print_time = current_time
            
        return stats

    def ready(self):
        if len(self) >= WRK_CONFIG["learning_starts"]:
            return True
        else:
            return False

    def get_env_settings(self):
        return self.env_settings_set

    def check_done(self):

        for i in range(GENERAL_CONFIG["max_num_agents"]):
            if (i + 1, WRK_CONFIG["max_map_length"]) not in self.stat_dict:
                return False

            l = self.stat_dict[(i + 1, WRK_CONFIG["max_map_length"])]

            if len(l) < 200:
                return False
            elif sum(l) < 200 * WRK_CONFIG["pass_rate"]:
                return False

        return True


@ray.remote(num_cpus=1)
class Learner:
    def __init__(self, buffer: GlobalBuffer, training_steps=10000):
        # 检查是否可以使用MPS加速
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
            
        self.model = Network()
        self.model.to(self.device)
        self.tar_model = deepcopy(self.model)
        self.optimizer = Adam(self.model.parameters(), lr=1e-4)
        self.scheduler = MultiStepLR(
            self.optimizer, milestones=[200000, 400000], gamma=0.5
        )
        self.buffer = buffer
        self.counter = 0
        self.last_counter = 0
        self.done = False
        self.loss = 0
        self.actors = []  # 存储所有Actor的引用

        self.steps = training_steps
        self.weights = None
        self.store_weights()

        self.last_print_time = time.time()
        self.print_interval = 15.0  # 每15秒最多打印一次状态

    def add_actor(self, actor):
        """添加Actor引用"""
        self.actors.append(actor)

    def get_weights(self):
        """获取网络权重"""
        return self.weights

    def store_weights(self):
        """存储网络权重"""
        state_dict = self.model.state_dict()
        for k, v in state_dict.items():
            state_dict[k] = v.cpu()
        self.weights = state_dict

    def run(self):
        self.learning_thread = threading.Thread(target=self.train, daemon=True)
        self.learning_thread.start()

    def train(self):
        """训练循环"""
        print("开始训练...")
        while not self.done and self.counter < self.steps:
            try:
                # 从经验回放缓冲区采样
                batch = ray.get(self.buffer.sample.remote())
                if batch is None:
                    time.sleep(0.1)
                    continue

                # 解包数据
                b_obs, b_action, b_reward, b_done, b_steps, b_seq_len, b_hidden, b_comm_mask, idxes, weights, old_ptr = batch

                # 将数据移到设备上并检查NaN
                b_obs = torch.FloatTensor(b_obs).to(self.device)
                b_action = torch.LongTensor(b_action).to(self.device)
                b_reward = torch.FloatTensor(b_reward).to(self.device)
                b_done = torch.FloatTensor(b_done).to(self.device)
                b_steps = torch.FloatTensor(b_steps).to(self.device)
                weights = torch.FloatTensor(weights).to(self.device)
                
                if b_hidden is not None:
                    b_hidden = torch.FloatTensor(b_hidden).to(self.device)
                b_comm_mask = torch.FloatTensor(b_comm_mask).to(self.device)

                # 检查输入数据是否包含NaN
                if (torch.isnan(b_obs).any() or torch.isnan(b_reward).any() or 
                    torch.isnan(b_done).any() or torch.isnan(b_steps).any()):
                    print("警告：输入数据包含NaN值，跳过此批次")
                    continue

                # 计算target Q值
                with torch.no_grad():
                    next_values, _, _ = self.tar_model(
                        b_obs[:, -1],  # 使用最后一个时间步
                        b_steps,
                        b_hidden,
                        b_comm_mask[:, -1]
                    )  # [B, N, 1]
                    
                    # 处理可能的NaN值
                    next_values = torch.nan_to_num(next_values, 0.0)
                    b_q_ = (1 - b_done.unsqueeze(1)) * next_values.squeeze(-1)  # [B, N]

                # 计算当前Q值
                current_values, _, _ = self.model(
                    b_obs[:, 0],  # 使用第一个时间步
                    b_steps,
                    b_hidden,
                    b_comm_mask[:, 0]
                )  # [B, N, A]
                
                # 处理可能的NaN值
                current_values = torch.nan_to_num(current_values, 0.0)
                
                # 收集每个智能体选择的动作的Q值
                b_q = current_values.gather(2, b_action.unsqueeze(-1)).squeeze(-1)  # [B, N]

                # 计算TD误差，避免除以零
                td_error = b_q - (b_reward + (0.99 ** b_steps.unsqueeze(1)) * b_q_)  # [B, N]
                
                # 计算每个批次的平均TD误差作为优先级
                priorities = td_error.abs().mean(dim=1).clamp(1e-4).cpu().numpy()  # [B]

                # 计算损失 (使用Huber损失)
                loss = (weights.unsqueeze(1) * self.huber_loss(td_error)).mean()

                # 检查损失值是否为NaN
                if torch.isnan(loss):
                    print("警告：损失值为NaN，跳过此批次")
                    continue

                # 优化器步骤
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 40)
                self.optimizer.step()
                self.scheduler.step()

                # 更新优先级
                self.buffer.update_priorities.remote(idxes, priorities, old_ptr)

                # 存储新的权重
                if self.counter % 5 == 0:
                    self.store_weights()

                self.counter += 1
                self.loss += loss.item()

                # 更新目标网络
                if self.counter % WRK_CONFIG["target_network_update_freq"] == 0:
                    self.tar_model.load_state_dict(self.model.state_dict())

                # 打印训练状态
                current_time = time.time()
                if current_time - self.last_print_time >= self.print_interval:
                    steps_per_sec = (self.counter - self.last_counter) / (current_time - self.last_print_time)
                    avg_loss = self.loss / (self.counter - self.last_counter) if self.counter > self.last_counter else 0
                    print(f"训练步数: {self.counter}, 损失: {avg_loss:.4f}, 速度: {steps_per_sec:.2f} steps/s")
                    self.last_counter = self.counter
                    self.loss = 0
                    self.last_print_time = current_time

            except Exception as e:
                print(f"训练过程中出现错误: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

    def huber_loss(self, td_error, kappa=1.0):
        """Huber损失函数
        
        Args:
            td_error: TD误差
            kappa: Huber损失的阈值
            
        Returns:
            损失值
        """
        abs_td = torch.abs(td_error)
        quadratic = torch.min(abs_td, torch.tensor(kappa, device=td_error.device))
        linear = abs_td - quadratic
        return 0.5 * quadratic.pow(2) + kappa * linear

    def stats(self, interval: int):
        current_time = time.time()
        
        # 计算损失值
        if self.counter != self.last_counter:
            current_loss = self.loss / (self.counter - self.last_counter)
        else:
            current_loss = 0.0

        # 重置计数器和损失
        self.last_counter = self.counter
        self.loss = 0

        # 从所有Actor收集统计信息
        actor_stats_futures = [actor.get_stats.remote() for actor in self.actors]
        try:
            actor_stats = ray.get(actor_stats_futures, timeout=2.0)
            actor_stats = [s for s in actor_stats if s is not None]
        except Exception:
            actor_stats = []

        # 计算平均统计信息
        if actor_stats:
            mean_stats = {
                'mean_reward': float(np.mean([s['mean_reward'] for s in actor_stats])),
                'mean_episode_length': float(np.mean([s['mean_episode_length'] for s in actor_stats])),
                'success_rate': float(np.mean([s['success_rate'] for s in actor_stats]))
            }
        else:
            mean_stats = {
                'mean_reward': 0.0,
                'mean_episode_length': 0.0,
                'success_rate': 0.0
            }

        if current_time - self.last_print_time >= self.print_interval:
            print(f"\n(Learner) 训练进度: {self.counter}/{WRK_CONFIG['training_times']} | "
                  f"损失值: {current_loss:.4f} | "
                  f"更新速度: {(self.counter - self.last_counter) / interval:.2f}/s")
            self.last_print_time = current_time

        return {
            'done': self.done,
            'update_speed': (self.counter - self.last_counter) / interval if interval > 0 else 0,
            'loss': float(current_loss),
            'mean_reward': mean_stats['mean_reward'],
            'mean_episode_length': mean_stats['mean_episode_length'],
            'success_rate': mean_stats['success_rate']
        }


@ray.remote(num_cpus=1)
class Actor:
    def __init__(self, worker_id, epsilon, learner, buffer):
        self.worker_id = worker_id
        self.epsilon = epsilon
        self.learner = learner
        self.buffer = buffer
        
        # 延迟导入Environment，避免循环导入
        from pathfinding.environment import Environment
        
        # 初始化环境
        self.env = Environment()
        self.map_size = self.env.map_length
        
        # 初始化任务生成器
        self.task_generator = TaskGenerator(self.map_size)
        
        # 初始化本地缓冲区
        self.local_buffer = LocalBuffer()
        
        # 初始化网络
        self.network = Network()
        weights_ref = ray.get(self.learner.get_weights.remote())
        self.network.load_state_dict(weights_ref)
        self.network.eval()
        
        # 统计信息
        self.rewards = []
        self.episode_lengths = []
        self.successes = []
        self.max_stats_len = 100  # 最多保存100个回合的统计信息

    def run(self):
        """运行Actor"""
        while True:
            # 重置环境
            obs = self.env.reset()
            done = False
            hidden = None
            comm_mask = None
            
            # 跟踪当前回合的统计数据
            episode_reward = 0
            episode_length = 0
            
            # 生成新任务
            task_generator = TaskGenerator(self.env.map_length)
            task_generator.set_obstacle_map(self.env.obstacle_map)
            tasks = task_generator.generate_tasks(self.env.num_agents)
            for start_pos, goal_pos, priority in tasks:
                self.env.add_task(start_pos, goal_pos, priority)
            
            while not done:
                # 获取智能体状态
                agent_states = self._get_agent_states()
                
                # 获取任务特征
                task_features = self._get_task_features(self.env.tasks)
                
                # 转换为张量
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # 添加批次维度
                pos_tensor = torch.tensor(self.env.agents_pos, dtype=torch.float32).unsqueeze(0)  # 添加批次维度
                task_tensor = torch.tensor(task_features, dtype=torch.float32)
                state_tensor = torch.tensor(agent_states, dtype=torch.float32)
                
                # 选择动作
                if np.random.random() < self.epsilon:
                    actions = np.random.randint(0, 5, size=self.env.num_agents)
                    hidden = None
                    comm_mask = None
                else:
                    with torch.no_grad():
                        q_val, new_hidden, new_comm_mask, task_scores = self.network.step(
                            obs_tensor,
                            pos_tensor,
                            task_tensor,
                            state_tensor,
                            hidden,  # 传入上一步的hidden state
                            comm_mask  # 传入上一步的comm_mask
                        )
                        actions = torch.argmax(q_val.squeeze(0), 1).cpu().numpy()
                        hidden = new_hidden  # 更新hidden state
                        comm_mask = new_comm_mask  # 更新comm_mask
                
                # 执行动作
                next_obs, rewards, done, _ = self.env.step(actions)
                
                # 更新统计信息
                episode_reward += np.mean(rewards)
                episode_length += 1
                
                # 存储经验
                self.local_buffer.add(
                    obs=obs,
                    action=actions,
                    reward=rewards,
                    hidden=hidden.detach().cpu().numpy() if hidden is not None else None,
                    comm_mask=comm_mask.detach().cpu().numpy() if comm_mask is not None else None,
                    task_features=task_features,
                    agent_states=agent_states,
                    task_assignments=self._get_task_assignments()
                )
                
                obs = next_obs
                
                # 如果buffer满了或者回合结束，发送数据到全局buffer
                if self.local_buffer.size >= self.local_buffer.capacity or done:
                    data = self.local_buffer.get()
                    self.buffer.add.remote(data)
                    self.local_buffer = LocalBuffer()
                    
                    # 更新网络权重
                    weights_ref = ray.get(self.learner.get_weights.remote())
                    self.network.load_state_dict(weights_ref)
                    self.network.eval()
            
            # 回合结束，更新统计信息
            self.rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.successes.append(self.env.is_success())
            
            # 保持统计列表在固定长度
            if len(self.rewards) > self.max_stats_len:
                self.rewards.pop(0)
                self.episode_lengths.pop(0)
                self.successes.pop(0)
                
    def get_stats(self):
        """返回Actor的统计信息"""
        if not self.rewards:  # 如果还没有数据
            return {
                'mean_reward': 0.0,
                'mean_episode_length': 0.0,
                'success_rate': 0.0
            }
        
        return {
            'mean_reward': float(np.mean(self.rewards)),
            'mean_episode_length': float(np.mean(self.episode_lengths)),
            'success_rate': float(np.mean(self.successes))
        }

    def _get_agent_states(self):
        """获取所有智能体的状态"""
        states = []
        for agent in self.env.agents:
            state = agent.get_state()
            states.append([
                state['battery'] / 100.0,  # 归一化电量
                state['experience'] / 10.0,  # 归一化经验值
                float(state['available'])    # 是否可用
            ])
        states = np.array(states)
        # 添加批次维度
        states = np.expand_dims(states, axis=0)
        return states
        
    def _get_task_features(self, tasks):
        """获取任务特征"""
        features = []
        for task_id, task in tasks.items():
            features.append([
                *task.start_pos,  # 起始位置
                *task.goal_pos    # 目标位置
            ])
        features = np.array(features)
        # 添加批次维度
        features = np.expand_dims(features, axis=0)
        return features
        
    def _get_task_assignments(self):
        """获取当前任务分配情况"""
        assignments = np.zeros((len(self.env.agents), len(self.env.tasks)))
        for i, agent in enumerate(self.env.agents):
            if agent.current_task is not None:
                for j, (task_id, task) in enumerate(self.env.tasks.items()):
                    if task_id == agent.current_task.id:
                        assignments[i, j] = 1
        return assignments

class TaskGenerator:
    """任务生成器"""
    def __init__(self, map_size):
        self.map_size = map_size
        self.obstacle_map = None

    def set_obstacle_map(self, obstacle_map):
        """设置障碍物地图"""
        self.obstacle_map = obstacle_map

    def _get_valid_positions(self):
        """获取所有非障碍物位置"""
        if self.obstacle_map is None:
            return [(i, j) for i in range(self.map_size) for j in range(self.map_size)]
        return [(i, j) for i in range(self.map_size) for j in range(self.map_size) 
                if self.obstacle_map[i, j] == 0]

    def generate_tasks(self, num_agents):
        """生成任务"""
        from pathfinding.models.dhc.task import TaskPriority
        tasks = []
        priorities = list(TaskPriority)  # 获取所有优先级
        
        # 获取所有有效位置
        valid_positions = self._get_valid_positions()
        if len(valid_positions) < 2:
            print("Warning: Not enough valid positions to generate tasks")
            return tasks
        
        for _ in range(num_agents):
            if len(valid_positions) < 2:
                break
                
            # 随机选择起点
            start_idx = np.random.randint(0, len(valid_positions))
            start_pos = valid_positions.pop(start_idx)
            
            # 随机选择终点
            goal_idx = np.random.randint(0, len(valid_positions))
            goal_pos = valid_positions.pop(goal_idx)
            
            # 随机选择优先级
            priority = np.random.choice(priorities)
            
            tasks.append((start_pos, goal_pos, priority))
            
        return tasks
