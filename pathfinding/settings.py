import os
import yaml
import numpy as np

# 添加元组支持
def construct_python_tuple(self, node):
    return tuple(self.construct_sequence(node))

yaml.SafeLoader.add_constructor(u'tag:yaml.org,2002:python/tuple', construct_python_tuple)

# 获取当前文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录路径
root_dir = os.path.dirname(current_dir)
# 配置文件路径
config_path = os.path.join(root_dir, "config.yaml")

# 尝试加载配置文件
yaml_data = {}
try:
    with open(config_path, "r") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        print(f"配置文件加载成功: {config_path}")
except FileNotFoundError:
    # 尝试从当前目录加载
    try:
        with open("config.yaml", "r") as yaml_file:
            yaml_data = yaml.safe_load(yaml_file)
            print(f"配置文件从当前目录加载成功")
    except FileNotFoundError:
        # 尝试从上级目录加载
        try:
            parent_dir = os.path.dirname(os.getcwd())
            alternative_path = os.path.join(parent_dir, "config.yaml")
            with open(alternative_path, "r") as yaml_file:
                yaml_data = yaml.safe_load(yaml_file)
                print(f"配置文件从上级目录加载成功: {alternative_path}")
        except FileNotFoundError:
            print("无法找到配置文件，使用默认配置")
            # 设置默认配置
            yaml_data = {
                "environment": {
                    "init_env_settings": [4, 40],
                    "num_charging_stations": 4,
                    "observation_radius": 4,
                    "reward_fn": {
                        "charging": 0.1,
                        "collision": -0.5,
                        "complete_task": 5.0,
                        "low_battery": -0.2,
                        "move": -0.1,
                        "stay_off_goal": -0.1,
                        "stay_on_goal": 0.0,
                        "step": -0.01
                    }
                },
                "model": {
                    "communication": {
                        "comm_hidden_dim": 64,
                        "key_dim": 16,
                        "num_comm_heads": 4
                    },
                    "hidden_size": 64,
                    "num_layers": 2
                },
                "training": {
                    "batch_size": 64,
                    "buffer_capacity": 10000,
                    "epochs": 10000,
                    "eval_interval": 200,
                    "learning_rate": 0.0001,
                    "max_training_steps": 100,
                    "save_interval": 500,
                    "update_target_interval": 100
                },
                "dhc": {
                    "train": {
                        "batch_size": 64,
                        "learning_rate": 0.0003,
                        "num_epochs": 4,
                        "gamma": 0.99,
                        "clip_epsilon": 0.2
                    }
                }
            }

# 获取配置值的辅助函数
def get_config(key_path, default=None):
    """获取配置值
    
    Args:
        key_path: 配置键路径，如 "environment.reward_fn.step"
        default: 默认值
        
    Returns:
        配置值
    """
    keys = key_path.split(".")
    value = yaml_data
    
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default

__all__ = ["yaml_data", "get_config"]
