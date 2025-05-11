import matplotlib.pyplot as plt
import numpy as np
import random
from typing import List, Union

plt.ion()
from matplotlib import colors

from pathfinding.settings import yaml_data as settings

ENV_CONFIG = settings["environment"]

action_list = np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int)

color_map = np.array(
    [
        [255, 255, 255],  # white - empty
        [190, 190, 190],  # gray - obstacle
        [0, 191, 255],    # blue - agent
        [255, 165, 0],    # orange - goal
        [0, 250, 154],    # green - charging station
        [255, 0, 0],      # red - low battery
    ]
)


def map_partition(map):
    """
    将地图分割成独立的区域
    """
    empty_pos = np.argwhere(map == 0).astype(np.int).tolist()
    empty_pos = [tuple(pos) for pos in empty_pos]

    if not empty_pos:
        raise RuntimeError("no empty position")

    partition_list = []
    while empty_pos:
        partition = []
        start_pos = empty_pos.pop(0)
        partition.append(start_pos)

        open_list = [start_pos]
        while open_list:
            current_pos = open_list.pop(0)
            x, y = current_pos

            # 检查上方
            up = (x - 1, y)
            if up[0] >= 0 and map[up[0], up[1]] == 0 and up in empty_pos:
                empty_pos.remove(up)
                open_list.append(up)
                partition.append(up)

            # 检查下方
            down = (x + 1, y)
            if down[0] < map.shape[0] and map[down[0], down[1]] == 0 and down in empty_pos:
                empty_pos.remove(down)
                open_list.append(down)
                partition.append(down)

            # 检查左方
            left = (x, y - 1)
            if left[1] >= 0 and map[left[0], left[1]] == 0 and left in empty_pos:
                empty_pos.remove(left)
                open_list.append(left)
                partition.append(left)

            # 检查右方
            right = (x, y + 1)
            if right[1] < map.shape[1] and map[right[0], right[1]] == 0 and right in empty_pos:
                empty_pos.remove(right)
                open_list.append(right)
                partition.append(right)

        if len(partition) >= 2:
            partition_list.append(partition)

    return partition_list


class Environment:
    def __init__(
        self,
        num_agents: int = ENV_CONFIG["init_env_settings"][0],
        map_length: int = ENV_CONFIG["init_env_settings"][1],
        obs_radius: int = ENV_CONFIG["observation_radius"],
        reward_fn: dict = ENV_CONFIG["reward_fn"],
        fix_density=None,
        curriculum=False,
        init_env_settings_set=ENV_CONFIG["init_env_settings"],
    ):
        """初始化环境"""
        self.num_agents = num_agents
        self.map_length = map_length
        self.map_size = map_length  # 为了兼容性添加map_size
        self.obs_radius = obs_radius
        self.reward_fn = reward_fn
        self.fix_density = fix_density
        self.curriculum = curriculum
        self.init_env_settings_set = init_env_settings_set

        # 初始化地图
        self.map = np.zeros((self.map_length, self.map_length), dtype=np.int32)
        self.agents_pos = np.zeros((self.num_agents, 2), dtype=np.int32)
        self.goals_pos = np.zeros((self.num_agents, 2), dtype=np.int32)

        # 添加新的组件
        self.task_allocator = None
        self.agents = []
        self.tasks = {}  # 使用字典存储任务，键为任务ID
        self.charging_stations = []
        
        # 初始化任务层
        self.task_layer = np.zeros((self.map_length, self.map_length), dtype=np.int32)
        
        # 初始化任务分配器
        self._init_task_allocator()
        
        # 初始化智能体
        self._init_agents()
        
        # 初始化充电站
        self._init_charging_stations()
        
        # 初始化渲染
        self.fig = None
        self.ax = None

        self.steps = 0

    def _init_charging_stations(self):
        """初始化充电站"""
        from pathfinding.settings import yaml_data as settings
        
        # 获取环境配置
        env_config = settings["environment"]
        num_charging_stations = env_config.get("num_charging_stations", 2)
        
        # 随机生成充电站位置
        charging_stations = []
        for _ in range(num_charging_stations):
            # 随机选择一个空位置
            available_positions = []
            for i in range(self.map_length):
                for j in range(self.map_length):
                    if self.map[i, j] == 0:  # 空位置
                        # 检查是否已被智能体或目标占用
                        is_occupied = False
                        for agent_pos in self.agents_pos:
                            if np.array_equal([i, j], agent_pos):
                                is_occupied = True
                                break
                        for goal_pos in self.goals_pos:
                            if np.array_equal([i, j], goal_pos):
                                is_occupied = True
                                break
                        if not is_occupied:
                            available_positions.append((i, j))
            
            if available_positions:
                # 随机选择一个位置
                station_pos = random.choice(available_positions)
                charging_stations.append(station_pos)
                # 标记地图上的充电站位置
                self.map[station_pos] = 4  # 使用4表示充电站
        
        # 设置充电站
        self.charging_stations = charging_stations
        if self.task_allocator:
            self.task_allocator.set_charging_stations(charging_stations)

    def _init_agents(self):
        """初始化智能体"""
        from pathfinding.models.dhc.agent import Agent
        self.agents = []
        for i in range(self.num_agents):
            # 创建智能体
            position = tuple(self.agents_pos[i])
            goal = tuple(self.goals_pos[i])
            
            # 随机初始电量 (60-100%)
            battery_level = random.uniform(60, 100)
            
            # 创建智能体
            agent = Agent(
                id=i,
                pos=position,
                max_battery=100,
                communication_range=10.0
            )
            
            # 添加到列表
            self.agents.append(agent)
            
            # 添加到任务分配器
            if self.task_allocator:
                self.task_allocator.add_agent(agent)

    def _update_agent_states(self):
        """更新智能体状态"""
        for i, agent in enumerate(self.agents):
            # 更新位置
            agent.update_position(self.agents_pos[i])
            
            # 更新电量 (根据移动距离消耗)
            if not np.array_equal(self.agents_pos[i], agent.pos):
                distance = np.linalg.norm(self.agents_pos[i] - agent.pos)
                consumption = distance * 0.1  # 每单位距离消耗0.1电量
                agent.update_battery(consumption)
                
            # 检查是否在充电站
            if any(np.array_equal(self.agents_pos[i], station) for station in self.charging_stations):
                agent.current_battery = min(agent.max_battery, agent.current_battery + 5)  # 每步充电5单位
                
    def add_task(self, start_pos, goal_pos, priority=None):
        """添加新任务到环境中"""
        from pathfinding.models.dhc.task import Task, TaskPriority
        
        if priority is None:
            priority = TaskPriority.MEDIUM
            
        task_id = len(self.tasks)
        task = Task(
            id=task_id, 
            start_pos=start_pos, 
            goal_pos=goal_pos, 
            priority=priority
        )
        task.creation_time = self.steps
        self.tasks[task_id] = task
        
        # 添加到任务分配器
        if self.task_allocator:
            self.task_allocator.add_task(task)
        
        # 更新任务层
        self.task_layer[start_pos[0], start_pos[1]] = priority.value  # 使用.value获取枚举值
        self.task_layer[goal_pos[0], goal_pos[1]] = priority.value    # 使用.value获取枚举值
        
        return task_id
        
    def update_env_settings_set(self, new_env_settings_set):
        self.init_env_settings_set = new_env_settings_set

    def reset(self, num_agents=None, map_length=None):
        """重置环境"""
        if num_agents is not None:
            self.num_agents = num_agents
        if map_length is not None:
            self.map_length = map_length
            self.map_size = map_length

        # 初始化地图
        self.map = np.zeros((self.map_length, self.map_length), dtype=np.int32)
        
        # 初始化任务层
        self.task_layer = np.zeros((self.map_length, self.map_length), dtype=np.int32)
        
        # 随机生成障碍物
        obstacle_density = np.random.triangular(0, 0.33, 0.5)
        obstacle_mask = np.random.choice(
            [0, 1],
            size=(self.map_length, self.map_length),
            p=[1 - obstacle_density, obstacle_density]
        )
        self.map[obstacle_mask == 1] = 1

        # 初始化智能体和目标位置
        self.agents_pos = np.zeros((self.num_agents, 2), dtype=np.int32)
        self.goals_pos = np.zeros((self.num_agents, 2), dtype=np.int32)

        # 获取可用位置
        empty_positions = np.argwhere(self.map == 0).tolist()
        if len(empty_positions) < self.num_agents * 2:
            return self.reset(num_agents, map_length)  # 如果空位不够，重新生成地图

        # 随机选择起点和终点
        positions = random.sample(empty_positions, self.num_agents * 2)
        for i in range(self.num_agents):
            self.agents_pos[i] = positions[i]
            self.goals_pos[i] = positions[i + self.num_agents]

        # 重置其他状态
        self.steps = 0
        self.tasks = {}
        self._init_agents()
        self._init_charging_stations()
        self._init_task_allocator()

        # 返回观察
        return self.observe()

    def load(self, map: np.ndarray, agents_pos: np.ndarray, goals_pos: np.ndarray):
        """载入已有地图和位置"""
        self.map = np.copy(map)
        self.agents_pos = np.copy(agents_pos)
        self.goals_pos = np.copy(goals_pos)

        self.num_agents = agents_pos.shape[0]
        self.map_size = (self.map.shape[0], self.map.shape[1])

        self.steps = 0
        self.imgs = []

        # 初始化任务分配器等
        self._init_task_allocator()
        self._init_agents()
        self._init_charging_stations()

    def step(self, actions):
        """执行动作"""
        from pathfinding.models.dhc.task import TaskStatus
        
        next_pos = np.copy(self.agents_pos)
        rewards = np.zeros(self.num_agents)
        done = False

        # 更新智能体位置
        for agent_id in range(self.num_agents):
            action = actions[agent_id]
            # 获取智能体和可能的下一个位置
            agent = self.agents[agent_id]
            current_pos = self.agents_pos[agent_id]
            
            # 标记是否尝试无效移动
            attempted_invalid_move = False

            if action > 0:  # 如果不是停止动作
                next_pos[agent_id] = current_pos + action_list[action]

                # 检查是否超出地图边界
                if (next_pos[agent_id] < 0).any() or \
                   (next_pos[agent_id][0] >= self.map_length) or \
                   (next_pos[agent_id][1] >= self.map_length):
                    next_pos[agent_id] = current_pos
                    rewards[agent_id] = self.reward_fn["collision"] 
                    attempted_invalid_move = True
                    continue

                # 检查是否与其他智能体碰撞
                collision = False
                for other_id in range(self.num_agents):
                    if other_id != agent_id and np.array_equal(next_pos[agent_id], next_pos[other_id]):
                        collision = True
                        break
                if collision:
                    next_pos[agent_id] = current_pos
                    rewards[agent_id] = self.reward_fn["collision"]
                    attempted_invalid_move = True
                    continue

                # 检查是否与障碍物碰撞
                if self.map[next_pos[agent_id][0], next_pos[agent_id][1]] == 1:
                    next_pos[agent_id] = current_pos
                    rewards[agent_id] = self.reward_fn["collision"]
                    attempted_invalid_move = True
                    continue

                # 更新电量
                if action > 0:  # 如果移动了
                    distance = np.linalg.norm(next_pos[agent_id] - current_pos)
                    consumption = distance * 0.1  # 每单位距离消耗0.1电量
                    agent.update_battery(-consumption)
                    
                    # 如果电量过低，给予惩罚
                    if agent.current_battery < 20:  # 低于20%视为低电量
                        rewards[agent_id] += self.reward_fn.get("low_battery", -0.2)

            # 记录是否尝试无效移动（供任务分配器使用）
            if hasattr(agent, 'attempted_invalid_move'):
                agent.attempted_invalid_move = attempted_invalid_move

            # 检查是否在充电站
            if any(np.array_equal(next_pos[agent_id], station) for station in self.charging_stations):
                charge_amount = 5  # 每步充电5单位
                agent.update_battery(charge_amount)
                rewards[agent_id] += self.reward_fn.get("charging", 0.1)

            # 检查是否完成任务
            if agent.current_task is not None:
                if np.array_equal(next_pos[agent_id], agent.current_task.goal_pos):
                    rewards[agent_id] += self.reward_fn.get("complete_task", 5.0)
                    agent.complete_task()
                else:
                    # 常规移动奖励
                    rewards[agent_id] += self.reward_fn.get("move", -0.1)
            else:
                # 根据智能体是否在目标位置给不同的奖励
                if np.array_equal(next_pos[agent_id], self.goals_pos[agent_id]):
                    rewards[agent_id] += self.reward_fn.get("stay_on_goal", 0.0)
                else:
                    rewards[agent_id] += self.reward_fn.get("stay_off_goal", -0.1)

        # 更新智能体位置
        self.agents_pos = next_pos
        
        # 更新智能体状态
        for i, agent in enumerate(self.agents):
            agent.update_position(self.agents_pos[i])
            # 更新状态字典中的当前步骤
            if hasattr(agent, 'state'):
                agent.state['current_step'] = self.steps

        # 更新步数
        self.steps += 1
        
        # 更新任务分配器
        if self.task_allocator:
            self.task_allocator.update(self.steps)

        # 检查是否所有任务都完成
        all_tasks_completed = all(task.status == TaskStatus.COMPLETED for task in self.tasks.values()) if self.tasks else False
        
        # 检查是否超过最大步数
        if self.steps >= 100 or all_tasks_completed:
            done = True

        return self.observe(), rewards, done, {}

    def observe(self):
        """获取环境观察"""
        # 创建观察空间，增加两个新通道
        obs = np.zeros(
            (self.num_agents, 8, 2 * self.obs_radius + 1, 2 * self.obs_radius + 1),
            dtype=np.float32
        )

        # 为每个智能体生成观察
        for i in range(self.num_agents):
            # 获取智能体位置
            x, y = self.agents_pos[i]
            
            # 提取观察窗口
            x_min = max(0, x - self.obs_radius)
            x_max = min(self.map_length, x + self.obs_radius + 1)
            y_min = max(0, y - self.obs_radius)
            y_max = min(self.map_length, y + self.obs_radius + 1)
            
            # 计算填充范围
            pad_x_min = self.obs_radius - (x - x_min)
            pad_x_max = self.obs_radius + 1 + (x_max - x - 1)
            pad_y_min = self.obs_radius - (y - y_min)
            pad_y_max = self.obs_radius + 1 + (y_max - y - 1)
            
            # 地图层
            obs[i, 0, pad_x_min:pad_x_max, pad_y_min:pad_y_max] = self.map[x_min:x_max, y_min:y_max]
            
            # 智能体层
            agent_map = np.zeros_like(self.map)
            for j, pos in enumerate(self.agents_pos):
                if j != i:  # 不包括自己
                    agent_map[pos[0], pos[1]] = 1
            obs[i, 1, pad_x_min:pad_x_max, pad_y_min:pad_y_max] = agent_map[x_min:x_max, y_min:y_max]
            
            # 目标层
            goal_map = np.zeros_like(self.map)
            goal_map[self.goals_pos[i, 0], self.goals_pos[i, 1]] = 1
            obs[i, 2, pad_x_min:pad_x_max, pad_y_min:pad_y_max] = goal_map[x_min:x_max, y_min:y_max]
            
            # 充电站层
            station_map = np.zeros_like(self.map)
            for station in self.charging_stations:
                station_map[station[0], station[1]] = 1
            obs[i, 3, pad_x_min:pad_x_max, pad_y_min:pad_y_max] = station_map[x_min:x_max, y_min:y_max]
            
            # 电量层
            battery_map = np.zeros_like(self.map)
            battery_map[x, y] = self.agents[i].current_battery / self.agents[i].max_battery
            obs[i, 4, pad_x_min:pad_x_max, pad_y_min:pad_y_max] = battery_map[x_min:x_max, y_min:y_max]
            
            # 任务层
            task_map = np.zeros_like(self.map)
            if self.agents[i].current_task is not None:
                task = self.agents[i].current_task
                task_map[task.start_pos[0], task.start_pos[1]] = 0.5
                task_map[task.goal_pos[0], task.goal_pos[1]] = 1.0
            obs[i, 5, pad_x_min:pad_x_max, pad_y_min:pad_y_max] = task_map[x_min:x_max, y_min:y_max]
            
            # 生成路径引导层并添加到观察中
            path_guidance_x = np.zeros_like(self.map)
            path_guidance_y = np.zeros_like(self.map)
            
            # 获取当前智能体的路径引导
            if self.agents[i].current_task is not None:
                # 获取对应智能体的路径规划器
                pathfinder = self.agents[i].pathfinder
                
                # 如果任务未开始，生成到任务起点的引导
                if not self.agents[i].current_task.started:
                    # 生成从当前位置到任务起点的路径引导
                    guidance_layer = pathfinder.generate_path_guidance_layer(
                        (x, y), self.agents[i].current_task.start_pos
                    )
                # 如果任务已开始，生成到任务终点的引导
                else:
                    # 生成从当前位置到任务终点的路径引导
                    guidance_layer = pathfinder.generate_path_guidance_layer(
                        (x, y), self.agents[i].current_task.goal_pos
                    )
                
                # 提取X和Y方向的引导
                path_guidance_x = guidance_layer[:, :, 0]
                path_guidance_y = guidance_layer[:, :, 1]
            # 如果没有当前任务，尝试生成到目标位置的引导
            elif hasattr(self, 'goals_pos') and len(self.goals_pos) > i:
                # 获取对应智能体的路径规划器
                pathfinder = self.agents[i].pathfinder
                
                # 生成从当前位置到目标的路径引导
                guidance_layer = pathfinder.generate_path_guidance_layer(
                    (x, y), tuple(self.goals_pos[i])
                )
                
                # 提取X和Y方向的引导
                path_guidance_x = guidance_layer[:, :, 0]
                path_guidance_y = guidance_layer[:, :, 1]
            
            # 添加X方向路径引导层
            obs[i, 6, pad_x_min:pad_x_max, pad_y_min:pad_y_max] = path_guidance_x[x_min:x_max, y_min:y_max]
            
            # 添加Y方向路径引导层
            obs[i, 7, pad_x_min:pad_x_max, pad_y_min:pad_y_max] = path_guidance_y[x_min:x_max, y_min:y_max]

        return obs

    def render(self):
        if not hasattr(self, "fig"):
            self.fig = plt.figure()

        map = np.copy(self.map)
        for agent_id in range(self.num_agents):
            if np.array_equal(self.agents_pos[agent_id], self.goals_pos[agent_id]):
                map[tuple(self.agents_pos[agent_id])] = 4
            else:
                map[tuple(self.agents_pos[agent_id])] = 2
                map[tuple(self.goals_pos[agent_id])] = 3

        map = map.astype(np.uint8)
        # plt.xlabel('step: {}'.format(self.steps))

        # add text in plot
        self.imgs.append([])
        if hasattr(self, "texts"):
            for i, ((agent_x, agent_y), (goal_x, goal_y)) in enumerate(
                zip(self.agents_pos, self.goals_pos)
            ):
                self.texts[i].set_position((agent_y, agent_x))
                self.texts[i].set_text(i)
        else:
            self.texts = []
            for i, ((agent_x, agent_y), (goal_x, goal_y)) in enumerate(
                zip(self.agents_pos, self.goals_pos)
            ):
                text = plt.text(
                    agent_y, agent_x, i, color="black", ha="center", va="center"
                )
                plt.text(goal_y, goal_x, i, color="black", ha="center", va="center")
                self.texts.append(text)

        plt.imshow(color_map[map], animated=True)

        plt.show()
        # plt.ion()
        plt.pause(0.5)

    def close(self, save=False):
        plt.close()
        del self.fig

    def _init_task_allocator(self):
        """初始化任务分配器"""
        from pathfinding.models.dhc.task_allocator import TaskAllocator
        from pathfinding.models.dhc.agent import Agent
        
        # 创建任务分配器
        self.task_allocator = TaskAllocator(
            num_agents=self.num_agents,
            map_size=self.map_length,
            communication_range=10.0
        )
        
        # 创建智能体并添加到任务分配器
        self.agents = []
        for agent_id in range(self.num_agents):
            # 创建智能体
            position = tuple(self.agents_pos[agent_id])
            goal = tuple(self.goals_pos[agent_id])
            
            # 随机初始电量 (60-100%)
            battery_level = random.uniform(60, 100)
            
            # 创建智能体
            agent = Agent(
                id=agent_id,
                pos=position,
                max_battery=100,
                communication_range=10.0
            )
            
            # 添加到列表
            self.agents.append(agent)
            
            # 添加到任务分配器
            self.task_allocator.add_agent(agent)
        
        # 初始化充电站
        self._init_charging_stations()
        
        # 设置当前步骤
        self.current_step = 0

    def is_success(self):
        """检查当前回合是否成功
        
        根据任务完成度计算成功率，如果没有任务则返回0
        
        Returns:
            float: 成功率，介于0到1之间。1表示全部成功，0表示全部失败
        """
        from pathfinding.models.dhc.task import TaskStatus
        
        if not self.tasks:
            return 0.0
            
        completed_tasks = sum(1 for task in self.tasks.values() 
                              if task.status == TaskStatus.COMPLETED)
        total_tasks = len(self.tasks)
        
        # 计算成功率
        success_rate = completed_tasks / total_tasks if total_tasks > 0 else 0.0
        
        # 打印任务完成情况，以便调试
        if self.steps % 50 == 0 or self.steps == 99:  # 每50步或最后一步打印一次
            print(f"当前步数: {self.steps}, 任务完成情况: {completed_tasks}/{total_tasks}, 成功率: {success_rate:.2%}")
            
        return success_rate
