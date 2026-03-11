# Wheel Control - 差分车轮机器人LQR运动控制系统

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

基于LQR的差分车轮式机器人轨迹跟踪控制系统，采用Frenet坐标系误差建模。

## 功能特性

- **LQR控制器**: 基于Frenet坐标系的线性二次调节器
- **轨迹生成**: 支持Bezier曲线、圆、正弦等多种轨迹
- **实时可视化**: 轨迹跟踪动画、误差曲线、速度曲线、曲率曲线
- **参数自整定**: 支持贝叶斯优化、网格搜索、随机搜索
- **模块化设计**: 易于扩展新的路径规划器、速度规划器、控制器

## 项目结构

```
wheel_project/
├── config/                 # YAML配置文件
│   ├── default.yaml       # 默认配置
│   ├── robot.yaml         # 机器人参数
│   ├── simulation.yaml    # 仿真参数
│   └── controller/
│       └── lqr.yaml       # LQR参数
├── wheel_control/         # 主包
│   ├── trajectory/        # 轨迹生成
│   ├── kinematics/        # 运动学模型
│   ├── planner/           # 规划器(路径+速度)
│   ├── control/           # 控制器
│   ├── utils/             # 工具(配置、日志、Frenet)
│   └── simulation/        # 仿真环境和可视化
├── examples/              # 示例脚本
├── tests/                 # 单元测试
└── logs/                  # 日志输出
```

## 安装

```bash
# 克隆仓库
git clone https://github.com/your-repo/wheel_project.git
cd wheel_project

# 安装依赖
pip install -r requirements.txt
```

## 快速开始

### 基础轨迹跟踪

```python
from wheel_control.trajectory import BezierSplineTrajectory
from wheel_control.kinematics import DiffDriveKinematics
from wheel_control.control import LQRController
from wheel_control.simulation import SimulationEnv

# 生成轨迹
trajectory = BezierSplineTrajectory().generate()

# 创建运动学模型
kinematics = DiffDriveKinematics(wheel_base=0.3, max_v=1.5, max_omega=3.0)

# 创建LQR控制器
controller = LQRController(Q=[1.0, 2.0, 0.5, 0.5], R=[0.1, 0.1])

# 创建仿真环境并运行
env = SimulationEnv(trajectory, controller, kinematics)
result = env.run_episode()

print(f"RMS横向误差: {result.metrics['rms_lateral_error']:.4f}")
```

### 运行示例

```bash
# 基础轨迹跟踪
python examples/basic_tracking.py

# LQR参数整定
python examples/lqr_tuning.py

# 自定义轨迹
python examples/custom_trajectory.py
```

## 配置说明

### LQR参数 (`config/controller/lqr.yaml`)

```yaml
# 状态权重 Q: [e_lat, e_yaw, e_v, e_omega]
Q: [1.0, 2.0, 0.5, 0.5]

# 控制权重 R: [delta_v, delta_omega]
R: [0.1, 0.1]

tuner:
  enabled: false
  method: bayesian  # bayesian, grid, random
  n_trials: 50
```

### 机器人参数 (`config/robot.yaml`)

```yaml
wheel_base: 0.3       # 轮距 (m)
max_v: 1.5            # 最大线速度 (m/s)
max_omega: 3.0        # 最大角速度 (rad/s)
tau_v: 0.1            # 线速度时间常数 (s)
tau_omega: 0.08       # 角速度时间常数 (s)
```

## 核心算法

### Frenet坐标系误差

- `e_lat`: 横向误差（垂直于轨迹切向）
- `e_yaw`: 航向误差
- `e_v`: 速度误差
- `e_omega`: 角速度误差

### LQR控制律

```
u = -K * e + feedforward
```

其中 `K` 通过离散时间Riccati方程求解。

## 扩展指南

### 添加新的控制器

```python
from wheel_control.control.base import ControllerBase, ControlOutput

class MyController(ControllerBase):
    def compute_control(self, state, ref_trajectory, nearest_idx):
        # 实现控制逻辑
        return ControlOutput(v_cmd=..., omega_cmd=...)
    
    def reset(self):
        pass
```

### 添加新的轨迹生成器

```python
from wheel_control.trajectory.base import TrajectoryBase

class MyTrajectory(TrajectoryBase):
    def generate(self, rng=None):
        # 生成 (N, 7) 数组: [x, y, yaw, vx, vy, omega, kappa]
        return trajectory
```

## 可视化

系统提供实时可视化，包含：
1. 轨迹跟踪图（XY平面，参考vs实际）
2. 横向误差曲线
3. 航向误差曲线
4. 速度曲线（参考vs实际）
5. 曲率曲线

```python
from wheel_control.simulation import RealtimeVisualizer

visualizer = RealtimeVisualizer(realtime=True)
visualizer.setup(trajectory)

# 在仿真循环中更新
visualizer.update(state, ref_point, errors, dt)

# 显示最终结果
visualizer.show()
```

## 运行测试

```bash
pytest tests/
```

## 依赖

- numpy >= 1.20.0
- scipy >= 1.7.0
- pyyaml >= 5.4.0
- matplotlib >= 3.4.0

可选：
- scikit-optimize: 用于贝叶斯优化参数整定
- pytest: 用于单元测试

## 许可证

MIT License

## 参考

- Frenet坐标系: 基于路径的自然坐标系
- LQR控制: 线性二次调节器
- 差分驱动运动学: 非完整约束机器人模型
