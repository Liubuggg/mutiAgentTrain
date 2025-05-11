# credits to https://github.com/ZiyuanMa/DHC/blob/master/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
from typing import Tuple, Optional, Dict, Any
import torch.serialization

# 添加numpy.core.multiarray.scalar到安全全局变量列表
torch.serialization.add_safe_globals(['numpy.core.multiarray.scalar'])

# 设置设备
device_name = "cpu"  # 默认为CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
    device_name = "mps"
    print("使用 MPS (Apple Silicon GPU) 进行训练")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    device_name = "cuda"
    print("使用 CUDA (NVIDIA GPU) 进行训练")
else:
    device = torch.device("cpu")
    print("使用 CPU 进行训练")

from pathfinding.settings import yaml_data as settings

# 获取DHC配置
try:
    DHC_CONFIG = settings["dhc"]
except KeyError:
    print("警告：设置中不存在'dhc'键，使用默认值")
    DHC_CONFIG = {
        "train": {
            "epochs": 10000,
            "learning_rate": 1e-4,
            "batch_size": 64,
            "buffer_capacity": 10000,
            "update_target_interval": 100,
            "save_interval": 500,
            "eval_interval": 200
        },
        "model": {
            "hidden_dim": 64,
            "comm_heads": 4,
            "key_dim": 16,
            "input_shape": (8, 9, 9),
            "cnn_channels": 128,
            "hidden_dim": 256,
            "max_comm_agents": 3,
            "latent_dim": 784,
            "batch_size": 64
        },
        "communication": {
            "num_comm_heads": 4,
            "key_dim": 16,
            "comm_hidden_dim": 64
        },
        "batch_size": 64
    }

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = out + residual
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W_Q = nn.Linear(input_dim, output_dim * num_heads)
        self.W_K = nn.Linear(input_dim, output_dim * num_heads)
        self.W_V = nn.Linear(input_dim, output_dim * num_heads)
        self.W_O = nn.Linear(output_dim * num_heads, output_dim, bias=False)

    def forward(self, input, attn_mask):
        # input: [batch_size x num_agents x input_dim]
        batch_size, num_agents, input_dim = input.size()
        assert input_dim == self.input_dim

        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = (
            self.W_Q(input)
            .view(batch_size, num_agents, self.num_heads, -1)
            .transpose(1, 2)
        )  # q_s: [batch_size x num_heads x num_agents x output_dim]
        k_s = (
            self.W_K(input)
            .view(batch_size, num_agents, self.num_heads, -1)
            .transpose(1, 2)
        )  # k_s: [batch_size x num_heads x num_agents x output_dim]
        v_s = (
            self.W_V(input)
            .view(batch_size, num_agents, self.num_heads, -1)
            .transpose(1, 2)
        )  # v_s: [batch_size x num_heads x num_agents x output_dim]

        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
        assert (
            attn_mask.size(0) == batch_size
        ), f"mask dim {attn_mask.size(0)} while batch size {batch_size}"

        attn_mask = attn_mask.unsqueeze(1).repeat_interleave(
            self.num_heads, 1
        )  # attn_mask : [batch_size x num_heads x num_agents x num_agents]
        assert attn_mask.size() == (batch_size, self.num_heads, num_agents, num_agents)

        # context: [batch_size x num_heads x num_agents x output_dim]
        with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=False):
            scores = torch.matmul(q_s.float(), k_s.float().transpose(-1, -2)) / (
                self.output_dim**0.5
            )  # scores : [batch_size x n_heads x num_agents x num_agents]
            scores.masked_fill_(
                attn_mask, -1e9
            )  # Fills elements of self tensor with value where mask is one.
            attn = F.softmax(scores, dim=-1)

        context = torch.matmul(attn, v_s)
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, num_agents, self.num_heads * self.output_dim)
        )  # context: [batch_size x len_q x n_heads * d_v]
        output = self.W_O(context)

        return output  # output: [batch_size x num_agents x output_dim]


class CommBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim=64,
        num_heads=4,
        num_layers=2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.self_attn = MultiHeadAttention(input_dim, output_dim, num_heads)

        self.update_cell = nn.GRUCell(output_dim, input_dim)

    def forward(self, latent, comm_mask):
        """
        latent shape: batch_size x num_agents x latent_dim

        """
        batch_size, num_agents, _ = latent.size()

        # agent indices of agent that use communication
        update_mask = comm_mask.sum(dim=-1) > 1
        comm_idx = update_mask.nonzero(as_tuple=True)

        # no agent use communication, return
        if len(comm_idx[0]) == 0:
            return latent

        if len(comm_idx) > 1:
            update_mask = update_mask.unsqueeze(2)

        attn_mask = comm_mask == False

        for _ in range(self.num_layers):

            info = self.self_attn(latent, attn_mask=attn_mask)
            if len(comm_idx) == 1:

                batch_idx = torch.zeros(len(comm_idx[0]), dtype=torch.long)
                latent[batch_idx, comm_idx[0]] = self.update_cell(
                    info[batch_idx, comm_idx[0]], latent[batch_idx, comm_idx[0]]
                )
            else:
                update_info = self.update_cell(
                    info.view(-1, self.output_dim), latent.view(-1, self.input_dim)
                ).view(batch_size, num_agents, self.input_dim)
                latent = torch.where(update_mask, update_info, latent)

        return latent


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # 从配置文件中读取参数
        try:
            model_config = DHC_CONFIG.get('model', {})
            self.input_shape = tuple(model_config.get('input_shape', (8, 9, 9)))
            self.cnn_channels = model_config.get('cnn_channels', 128)
            self.hidden_dim = model_config.get('hidden_dim', 256)
            self.max_comm_agents = model_config.get('max_comm_agents', 3)
            self.latent_dim = model_config.get('latent_dim', 784)
            self._batch_size = model_config.get('batch_size', 64)
        except Exception as e:
            print(f"读取配置文件失败: {e}，使用默认值")
            self.input_shape = (8, 9, 9)
            self.cnn_channels = 128
            self.hidden_dim = 256
            self.max_comm_agents = 3
            self.latent_dim = 784
            self._batch_size = 64

        # 观测编码器
        self.obs_encoder = nn.Sequential(
            nn.Conv2d(self.input_shape[0], self.cnn_channels, 3, 1, padding=1),
            ResBlock(self.cnn_channels),
            ResBlock(self.cnn_channels),
            ResBlock(self.cnn_channels),
            nn.Conv2d(self.cnn_channels, 16, 1, 1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )

        # 任务编码器
        self.task_encoder = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 32),
            nn.ReLU(inplace=True),
        )
        
        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 16),
            nn.ReLU(inplace=True),
        )
        
        flat_cnn_size = 16 * self.input_shape[1] * self.input_shape[2]
        encoder_output_size = flat_cnn_size + 32 + 16
        
        # GRU单元
        self.gru_cell = nn.GRUCell(
            input_size=encoder_output_size, 
            hidden_size=self.latent_dim
        )
        
        # 通信块
        self.comm = CommBlock(
            input_dim=self.latent_dim, 
            output_dim=self.latent_dim // 4, 
            num_heads=4, 
            num_layers=2
        )
        
        # 任务分配网络
        self.task_assignment = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 5),  # 5个任务选项
        )
        
        # 行动值网络
        self.value = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 5),  # 5个动作：0=不动，1=上，2=右，3=下，4=左
        )
        
        self.hidden = None
        self._xavier_init()

    def _xavier_init(self):
        for _, m in self.named_modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def compute_task_scores(self, agent_states, task_features):
        """计算任务分配分数"""
        batch_size = agent_states.size(0)
        num_agents = agent_states.size(1)
        num_tasks = task_features.size(1)

        # 编码智能体状态
        agent_encodings = self.state_encoder(agent_states)  # [B, N, H]
        
        # 编码任务
        task_encodings = self.task_encoder(task_features)   # [B, M, H]

        # 计算所有智能体-任务对的分数
        scores = torch.zeros(batch_size, num_agents, num_tasks, device=agent_states.device)
        
        # 扩展维度以进行广播
        agent_encodings = agent_encodings.unsqueeze(2).expand(-1, -1, num_tasks, -1)  # [B, N, M, H]
        task_encodings = task_encodings.unsqueeze(1).expand(-1, num_agents, -1, -1)   # [B, N, M, H]
        
        # 计算所有对的分数
        pair_features = torch.cat([agent_encodings, task_encodings], dim=-1)  # [B, N, M, 2H]
        pair_features = pair_features.view(-1, 48)  # [B*N*M, 48] (16 + 32 = 48)
        
        # 计算分数
        scores = self.task_assignment(pair_features).view(batch_size, num_agents, num_tasks)
        
        return scores

    @torch.no_grad()
    def step(self, obs, pos, tasks=None, agent_states=None, hidden=None, comm_mask=None):
        """单步执行
        Args:
            obs: 观察 [B, N, C, H, W]
            pos: 位置 [B, N, 2]
            tasks: 任务特征 [B, M, 4]
            agent_states: 智能体状态 [B, N, 2]
            hidden: 上一步的隐藏状态 [B*N, latent_dim]
            comm_mask: 上一步的通信掩码 [B, N, N]
        Returns:
            action_values: 动作值 [B, N, 1]
            new_hidden: 新的隐藏状态 [B*N, latent_dim]
            task_assignment: 任务分配 [B, N, 5]
        """
        batch_size, num_agents = obs.shape[:2]
        device = obs.device
        
        # 重塑观察张量以适应卷积层
        obs_reshaped = obs.view(batch_size * num_agents, *obs.shape[2:])  # [B*N, C, H, W]
        
        # 编码观测
        cnn_out = self.obs_encoder(obs_reshaped)  # [B*N, cnn_out_dim]
        
        # 编码任务（如果有）
        task_encoding = torch.zeros(batch_size * num_agents, 32, device=device)
        if tasks is not None:
            tasks_flat = tasks.view(-1, 4)  # [B*N, 4]
            task_encoding = self.task_encoder(tasks_flat)  # [B*N, 32]
        
        # 编码智能体状态（如果有）
        state_encoding = torch.zeros(batch_size * num_agents, 16, device=device)
        if agent_states is not None:
            states_flat = agent_states.view(-1, 2)  # [B*N, 2]
            state_encoding = self.state_encoder(states_flat)  # [B*N, 16]
        
        # 合并所有编码
        obs_encoding = torch.cat([cnn_out, task_encoding, state_encoding], dim=1)  # [B*N, encoder_output_size]
        
        # 如果没有隐藏状态，初始化为零
        if hidden is None:
            hidden = torch.zeros(batch_size * num_agents, self.latent_dim, device=device)
        # 确保hidden是2D张量 [B*N, latent_dim]
        elif hidden.dim() > 2:
            hidden = hidden.view(-1, self.latent_dim)
        
        # 更新隐藏状态
        new_hidden = self.gru_cell(obs_encoding, hidden)  # [B*N, latent_dim]
        
        # 重塑隐藏状态用于通信
        latent = new_hidden.view(batch_size, num_agents, self.latent_dim)  # [B, N, latent_dim]
        
        # 通信
        if comm_mask is not None:
            # 确保comm_mask维度正确
            if comm_mask.dim() < 3:
                comm_mask = comm_mask.view(batch_size, num_agents, -1)
            # 确保comm_mask在正确的设备上
            comm_mask = comm_mask.to(device)
        else:
            # 创建默认的通信掩码
            comm_mask = torch.ones(batch_size, num_agents, num_agents, device=device)
            comm_mask.fill_diagonal_(0)  # 对角线设为0，智能体不与自己通信
        
        latent = self.comm(latent, comm_mask)  # [B, N, latent_dim]
        
        # 计算任务分配
        task_assignment = self.task_assignment(latent)  # [B, N, 5]
        
        # 计算动作值
        action_values = self.value(latent)  # [B, N, 1]
        
        return action_values, new_hidden, task_assignment

    def reset(self):
        self.hidden = None

    @torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu')
    def forward(self, obs, steps, hidden, comm_mask, tasks=None, agent_states=None):
        """前向传播
        
        Args:
            obs: 观察张量 [B, T, N, C, H, W] 或 [B*N, C, H, W]
            steps: 步数
            hidden: 隐藏状态 [B*N, latent_dim]
            comm_mask: 通信掩码 [B, T, N, N]
            tasks: 任务特征 [B, M, 4] (可选)
            agent_states: 智能体状态 [B, N, 2] (可选)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                action_values: [B, N, 1] - 动作值
                task_assignment: [B, N, 5] - 任务分配分数
                new_hidden: [B*N, latent_dim] - 新的隐藏状态
        """
        # 获取批次大小和智能体数量
        if len(obs.shape) == 6:  # [B, T, N, C, H, W]
            batch_size, time_steps, num_agents = obs.shape[:3]
            # 重塑观察张量以适应卷积层
            obs_reshaped = obs.view(batch_size * time_steps * num_agents, *obs.shape[3:])  # [B*T*N, C, H, W]
        elif len(obs.shape) == 5:  # [B, N, C, H, W]
            batch_size, num_agents = obs.shape[:2]
            obs_reshaped = obs.view(batch_size * num_agents, *obs.shape[2:])  # [B*N, C, H, W]
        else:  # [B*N, C, H, W]
            batch_size = self._batch_size
            num_agents = obs.shape[0] // batch_size
            obs_reshaped = obs
        
        # 确保数据在正确的设备上
        device = obs.device
        
        # 编码观测
        cnn_out = self.obs_encoder(obs_reshaped)  # [B*T*N, cnn_out_dim] 或 [B*N, cnn_out_dim]
        
        # 如果是时序数据，只使用最后一个时间步
        if len(obs.shape) == 6:
            cnn_out = cnn_out.view(batch_size, time_steps, num_agents, -1)[:, -1]  # [B, N, cnn_out_dim]
            cnn_out = cnn_out.reshape(batch_size * num_agents, -1)  # [B*N, cnn_out_dim]
        
        # 编码任务（如果有）
        task_encoding = torch.zeros(batch_size * num_agents, 32, device=device)
        if tasks is not None:
            tasks_flat = tasks.view(-1, 4)  # [B*N, 4]
            task_encoding = self.task_encoder(tasks_flat)  # [B*N, 32]
        
        # 编码智能体状态（如果有）
        state_encoding = torch.zeros(batch_size * num_agents, 16, device=device)
        if agent_states is not None:
            states_flat = agent_states.view(-1, 2)  # [B*N, 2]
            state_encoding = self.state_encoder(states_flat)  # [B*N, 16]
        
        # 合并所有编码
        obs_encoding = torch.cat([cnn_out, task_encoding, state_encoding], dim=1)  # [B*N, encoder_output_size]
        
        # 确保hidden是2D张量 [B*N, latent_dim]
        if hidden is not None:
            if hidden.dim() > 2:
                hidden = hidden.view(-1, self.latent_dim)
        else:
            hidden = torch.zeros(batch_size * num_agents, self.latent_dim, device=device)
        
        # 更新隐藏状态
        new_hidden = self.gru_cell(obs_encoding, hidden)  # [B*N, latent_dim]
        
        # 重塑隐藏状态用于通信
        latent = new_hidden.view(batch_size, num_agents, self.latent_dim)  # [B, N, latent_dim]
        
        # 通信
        if comm_mask is not None:
            # 如果是时序数据，只使用最后一个时间步
            if comm_mask.dim() == 4:  # [B, T, N, N]
                comm_mask = comm_mask[:, -1]  # [B, N, N]
            # 确保comm_mask维度正确
            if comm_mask.dim() < 3:
                comm_mask = comm_mask.view(batch_size, num_agents, -1)
            # 确保comm_mask在正确的设备上
            comm_mask = comm_mask.to(device)
        else:
            # 创建默认的通信掩码
            comm_mask = torch.ones(batch_size, num_agents, num_agents, device=device)
            comm_mask.fill_diagonal_(0)  # 对角线设为0,智能体不与自己通信
        
        latent = self.comm(latent, comm_mask)  # [B, N, latent_dim]
        
        # 计算任务分配值
        task_assignment = self.task_assignment(latent)  # [B, N, 5]
        
        # 计算Q值
        action_values = self.value(latent)  # [B, N, 1]
        
        # 确保没有NaN值
        if torch.isnan(action_values).any() or torch.isnan(task_assignment).any():
            print("警告：检测到NaN值")
            action_values = torch.nan_to_num(action_values, 0.0)
            task_assignment = torch.nan_to_num(task_assignment, 0.0)
        
        return action_values, task_assignment, new_hidden

    def communicate(self, latent, comm_mask):
        """通信函数"""
        # 重新计算实际的批量大小
        n_agents = 0
        
        # 处理不同形状的通信掩码
        if comm_mask.dim() == 4:  # [B, h, N, N]
            batch_size = comm_mask.shape[0]
            n_agents = comm_mask.shape[2]
            # 重塑为 [B, N, N]，忽略头维度
            comm_mask = comm_mask[:, 0, :, :]
        elif comm_mask.dim() == 3:  # [B, N, N]
            batch_size = comm_mask.shape[0]
            n_agents = comm_mask.shape[1]
        else:
            # 如果掩码形状不符预期，尝试自适应
            batch_size = latent.shape[0] // (comm_mask.shape[-1] if comm_mask.dim() >= 2 else 4)
            n_agents = comm_mask.shape[-1] if comm_mask.dim() >= 2 else 4
        
        # 通信掩码检查
        assert comm_mask.shape[-1] == comm_mask.shape[-2], f"通信掩码大小应为正方形，但实际形状为{comm_mask.shape}"
        
        # 转换latent以适应通信
        latent = latent.view(batch_size, n_agents, self.latent_dim)
        
        # 使用通信模块处理latent状态
        latent = self.comm(latent, comm_mask)
        
        # 返回处理后的latent状态
        return latent.view(batch_size * n_agents, self.latent_dim)

    @staticmethod
    def load_model(model_path: str) -> 'Network':
        """加载模型
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            Network: 加载的模型
        """
        print(f"正在加载模型: {model_path}")
        try:
            # 加载模型文件
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            
            # 创建新的模型实例
            model = Network()
            
            # 如果是字典格式，检查是否包含模型状态字典
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    # 如果存在model_state_dict键，直接加载
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"成功加载模型 (训练轮次: {checkpoint.get('epoch', 'unknown')})")
                elif 'state_dict' in checkpoint:
                    # 如果存在state_dict键，直接加载
                    model.load_state_dict(checkpoint['state_dict'])
                    print(f"成功加载模型 (训练轮次: {checkpoint.get('epoch', 'unknown')})")
                else:
                    # 如果没有找到模型状态字典，尝试直接加载整个checkpoint
                    try:
                        model.load_state_dict(checkpoint)
                        print("成功加载模型状态字典")
                    except Exception as e:
                        print(f"警告：直接加载失败，尝试初始化新模型: {e}")
                        # 如果加载失败，返回新初始化的模型
                        return model
            else:
                # 如果不是字典格式，尝试直接加载
                try:
                    model.load_state_dict(checkpoint)
                    print("成功加载模型状态字典")
                except Exception as e:
                    print(f"警告：直接加载失败，尝试初始化新模型: {e}")
                    # 如果加载失败，返回新初始化的模型
                    return model
            
            return model
        except Exception as e:
            print(f"模型加载失败: {e}")
            print("返回新初始化的模型")
            return Network()


class CommunicationModule(nn.Module):
    """通信模块
    
    负责智能体之间的信息交换和协作
    """
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int,
                 num_heads: int = 4,
                 dropout: float = 0.1):
        """初始化通信模块
        
        Args:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            num_heads: 注意力头数
            dropout: Dropout比率
        """
        super().__init__()
        
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        
        # 多头注意力层
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, 
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量 [batch_size, num_agents, hidden_dim]
            mask: 注意力掩码 [batch_size, num_agents, num_agents]
            
        Returns:
            torch.Tensor: 输出张量 [batch_size, num_agents, hidden_dim]
        """
        batch_size, num_agents, _ = x.size()
        
        # 重塑输入以适应多头注意力
        x = x.transpose(0, 1)  # [num_agents, batch_size, hidden_dim]
        
        # 应用注意力
        attn_output, _ = self.attention(x, x, x, key_padding_mask=mask)
        attn_output = self.dropout(attn_output)
        
        # 第一个残差连接和层归一化
        x = self.norm1(x + attn_output)
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        ff_output = self.dropout(ff_output)
        
        # 第二个残差连接和层归一化
        x = self.norm2(x + ff_output)
        
        # 重塑回原始维度
        x = x.transpose(0, 1)  # [batch_size, num_agents, hidden_dim]
        
        return x


class TaskScoringModule(nn.Module):
    """任务评分模块
    
    负责评估任务对智能体的适合程度
    """
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int = 1,
                 dropout: float = 0.1):
        """初始化任务评分模块
        
        Args:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
            dropout: Dropout比率
        """
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量 [batch_size, input_dim]
            
        Returns:
            torch.Tensor: 评分张量 [batch_size, output_dim]
        """
        return self.network(x)


class DHCModel(nn.Module):
    """分布式分层协作模型
    
    实现无标签化的多智能体任务分配
    """
    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 hidden_dim: int = 128,
                 num_heads: int = 4,
                 dropout: float = 0.1):
        """初始化模型
        
        Args:
            obs_dim: 观察空间维度
            act_dim: 动作空间维度
            hidden_dim: 隐藏层维度
            num_heads: 注意力头数
            dropout: Dropout比率
        """
        super().__init__()
        
        # 特征提取网络
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 通信模块
        self.communication = CommunicationModule(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # 任务评分模块
        self.task_scoring = TaskScoringModule(
            input_dim=hidden_dim * 2,  # 智能体特征 + 任务特征
            hidden_dim=hidden_dim,
            output_dim=1,
            dropout=dropout
        )
        
        # 动作网络
        self.action_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, act_dim)
        )
        
        # 值函数网络
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # 隐藏状态
        self.hidden = None
        
    def init_hidden(self, batch_size: int, num_agents: int, hidden_dim: int):
        """初始化隐藏状态
        
        Args:
            batch_size: 批次大小
            num_agents: 智能体数量
            hidden_dim: 隐藏层维度
        """
        self.hidden = torch.zeros(batch_size, num_agents, hidden_dim)
        
    def forward(self,
                obs: torch.Tensor,
                comm_mask: Optional[torch.Tensor] = None,
                hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播
        
        Args:
            obs: 观察张量 [batch_size, num_agents, obs_dim]
            comm_mask: 通信掩码 [batch_size, num_agents, num_agents]
            hidden: 隐藏状态 [batch_size, num_agents, hidden_dim]
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                (动作, 值函数, 新的隐藏状态)
        """
        batch_size, num_agents, _ = obs.size()
        
        # 特征提取
        features = self.feature_extractor(obs)  # [batch_size, num_agents, hidden_dim]
        
        # 通信
        if comm_mask is not None:
            comm_mask = comm_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            comm_mask = comm_mask.reshape(batch_size * self.num_heads, num_agents, num_agents)
            
        comm_features = self.communication(features, comm_mask)
        
        # 合并特征
        combined_features = torch.cat([features, comm_features], dim=-1)
        
        # 计算任务评分
        task_scores = self.task_scoring(combined_features)
        
        # 计算动作
        actions = self.action_net(comm_features)
        
        # 计算值函数
        values = self.value_net(comm_features)
        
        # 更新隐藏状态
        if hidden is not None:
            self.hidden = hidden
            
        return actions, values, self.hidden
        
    def compute_task_scores(self,
                          agent_features: torch.Tensor,
                          task_features: torch.Tensor) -> torch.Tensor:
        """计算任务评分
        
        Args:
            agent_features: 智能体特征 [batch_size, num_agents, feature_dim]
            task_features: 任务特征 [batch_size, num_tasks, feature_dim]
            
        Returns:
            torch.Tensor: 任务评分矩阵 [batch_size, num_agents, num_tasks]
        """
        batch_size, num_agents, agent_dim = agent_features.size()
        _, num_tasks, task_dim = task_features.size()
        
        # 扩展维度以进行广播
        agent_features = agent_features.unsqueeze(2)  # [batch_size, num_agents, 1, agent_dim]
        task_features = task_features.unsqueeze(1)    # [batch_size, 1, num_tasks, task_dim]
        
        # 合并特征
        combined_features = torch.cat([agent_features, task_features], dim=-1)
        combined_features = combined_features.reshape(
            batch_size * num_agents * num_tasks, -1
        )
        
        # 计算评分
        scores = self.task_scoring(combined_features)
        scores = scores.reshape(batch_size, num_agents, num_tasks)
        
        return scores
        
    def get_action(self,
                  obs: torch.Tensor,
                  comm_mask: Optional[torch.Tensor] = None,
                  deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取动作
        
        Args:
            obs: 观察张量 [batch_size, num_agents, obs_dim]
            comm_mask: 通信掩码 [batch_size, num_agents, num_agents]
            deterministic: 是否使用确定性策略
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (动作, 动作概率)
        """
        actions, values, self.hidden = self(obs, comm_mask, self.hidden)
        
        if deterministic:
            action_probs = torch.ones_like(actions)
        else:
            action_probs = F.softmax(actions, dim=-1)
            
        return actions, action_probs
        
    def evaluate_actions(self,
                        obs: torch.Tensor,
                        actions: torch.Tensor,
                        comm_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """评估动作
        
        Args:
            obs: 观察张量 [batch_size, num_agents, obs_dim]
            actions: 动作张量 [batch_size, num_agents, act_dim]
            comm_mask: 通信掩码 [batch_size, num_agents, num_agents]
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (动作概率, 值函数)
        """
        _, values, self.hidden = self(obs, comm_mask, self.hidden)
        action_probs = F.softmax(actions, dim=-1)
        
        return action_probs, values
        
    def save(self, path: str):
        """保存模型
        
        Args:
            path: 保存路径
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'obs_dim': self.feature_extractor[0].in_features,
                'act_dim': self.action_net[-1].out_features,
                'hidden_dim': self.feature_extractor[0].out_features,
                'num_heads': self.communication.num_heads,
                'dropout': self.feature_extractor[2].p
            }
        }, path)
        
    @classmethod
    def load(cls, path: str) -> 'DHCModel':
        """加载模型
        
        Args:
            path: 模型路径
            
        Returns:
            DHCModel: 加载的模型
        """
        checkpoint = torch.load(path)
        config = checkpoint['config']
        
        model = cls(
            obs_dim=config['obs_dim'],
            act_dim=config['act_dim'],
            hidden_dim=config['hidden_dim'],
            num_heads=config['num_heads'],
            dropout=config['dropout']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
