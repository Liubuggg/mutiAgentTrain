import os
import yaml

def load_config(config_path=None):
    """加载配置文件
    Args:
        config_path: 配置文件路径，如果未指定则使用默认路径
    Returns:
        配置字典
    """
    if config_path is None:
        # 使用默认配置文件
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, 'default.yaml')
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"已加载配置文件: {config_path}")
        return config
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        return {} 