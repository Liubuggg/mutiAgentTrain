# 多智能体寻路系统

## 环境配置

本项目需要以下依赖:

- Python 3.8+
- PyTorch 2.0.0+ (支持CUDA和MPS)
- 其他依赖见requirements.txt

### 安装指南

可以使用以下命令一键安装所有依赖:

```bash
# 创建虚拟环境 (可选)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或者
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 硬件加速支持

本项目支持以下硬件加速:

- NVIDIA GPU (通过CUDA)
- Apple Silicon GPU (通过MPS)
- 如果以上都不可用，将自动使用CPU

## 训练模型

可以使用以下命令训练模型:

```bash
# 训练模型 (使用默认参数)
python -m pathfinding.models.dhc.train --epochs 50 --num_agents 4 --map_size 20 --num_tasks 4 --max_steps 500

# 或者使用简化命令
python -c "from pathfinding.models.dhc.train import train_dhc_model; train_dhc_model(epochs=50, num_agents=4, map_size=20, num_tasks=4, max_steps=500)"
```

## 可视化训练结果

训练完成后，可以使用以下命令可视化训练结果:

```bash
# 可视化训练结果
python -m pathfinding.models.dhc.visualize plot --log_dir logs/dhc_YYYYMMDD_HHMMSS
```

## 使用训练好的模型

可以使用以下命令运行训练好的模型:

```bash
# 可视化模型仿真
python -m pathfinding.models.dhc.visualize simulate --model_path models/dhc_YYYYMMDD_HHMMSS/model_final.pt --num_agents 4 --map_size 20 --num_tasks 4 --max_steps 300
```

python pathfinding/main.py --mode visualize --model_path models/model_final_fixed.pt --num_agents 4 --map_size 40 --num_tasks 8 --steps 200 --output_format gif

python -c "from pathfinding.models.dhc.train import train_dhc_model; train_dhc_model(epochs=1, num_agents=4, map_size=20, num_tasks=4, max_steps=100)"