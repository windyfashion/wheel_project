# Wheel Control — 差分车轮机器人 LQR 轨迹跟踪控制系统

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

基于 LQR 的差分车轮式机器人轨迹跟踪控制系统。采用 Frenet 坐标系误差建模，线性化模型包含**曲率耦合**与**一阶执行器动态**，通过矩阵指数精确离散化，并带有零速 look-ahead 前馈。

## 功能特性

- **完整 LQR 控制器** — Frenet 误差状态 `[e_lat, e_yaw, e_v, e_ω]`，A/B 矩阵含曲率耦合项与执行器惯性，`expm` 精确离散化，输出限幅
- **物理一致的轨迹生成** — 所有生成器（Circle / Straight / Sine / Bezier）统一经过 curvature-limited + forward-backward 加减速规划
- **中点法运动学积分** — 减少前向欧拉的航向累积误差
- **增量式实时可视化** — 基于 `set_data()` 的多子图实时显示，无每帧 `clear()` 开销
- **可复现仿真** — `SimulationEnv` 支持 `seed` 参数，调参/测试结果完全可复现
- **参数自整定** — Bayesian / Grid / Random 三种方法，log-uniform 采样，复用 `SimulationEnv` 评估
- **集成测试** — 4 种轨迹端到端收敛验证，`pytest` 自动化回归

## 项目结构

```
wheel_project/
├── config/                  # YAML 配置文件
│   ├── robot.yaml           # 机器人参数（轮距、执行器时间常数等）
│   ├── simulation.yaml      # 仿真参数（dt、最大步数等）
│   └── controller/
│       └── lqr.yaml         # LQR 权重与调参配置
├── wheel_control/           # 主包
│   ├── trajectory/          # 轨迹生成（Base + Circle/Sine/Straight/Bezier/Random）
│   ├── kinematics/          # 差速运动学（含执行器动态、中点法积分）
│   ├── control/             # 控制器（LQR + Tuner）
│   ├── planner/             # 规划器（Bezier路径 + 梯形速度）
│   ├── utils/               # 工具（MathUtils / FrenetFrame / Config / Logger）
│   └── simulation/          # 仿真环境 / 可视化 / 指标评估
├── examples/                # 示例脚本
├── tests/                   # 单元测试 + 集成测试
├── docs/                    # 架构文档
└── requirements.txt
```

## 安装

```bash
git clone https://github.com/your-repo/wheel_project.git
cd wheel_project
pip install -r requirements.txt
```

## 快速开始

```python
from wheel_control.trajectory import BezierSplineTrajectory
from wheel_control.kinematics import DiffDriveKinematics
from wheel_control.control import LQRController
from wheel_control.simulation import SimulationEnv

# 生成轨迹（含加减速规划）
trajectory = BezierSplineTrajectory(max_v=1.5, max_omega=3.0, max_acc=1.0).generate()

# 创建运动学模型
kinematics = DiffDriveKinematics(
    wheel_base=0.3, max_v=1.5, max_omega=3.0,
    tau_v=0.1, tau_omega=0.08, dt=0.02,
)

# 创建 LQR 控制器（需要与运动学共享 tau 参数）
controller = LQRController(
    dt=0.02,
    Q=[1.0, 2.0, 0.5, 0.5],
    R=[0.1, 0.1],
    tau_v=0.1,
    tau_omega=0.08,
    max_v=1.5,
    max_omega=3.0,
)

# 仿真并获取结果
env = SimulationEnv(trajectory, controller, kinematics, config={"dt": 0.02})
result = env.run_episode(seed=42)

print(f"RMS 横向误差: {result.metrics['rms_lateral_error']:.4f} m")
print(f"完成: {result.success}")
```

### 运行示例

```bash
python examples/basic_tracking.py      # 基础轨迹跟踪 + 可视化
python examples/lqr_tuning.py          # LQR 参数自整定
python examples/custom_trajectory.py   # 自定义轨迹
```

## 核心算法

### Frenet 误差动力学（含曲率耦合与执行器惯性）

```
ė_lat   =  v_ref · e_yaw
ė_yaw   = -v_ref · κ² · e_lat  -  κ · e_v  +  e_ω
ė_v     = -e_v / τ_v   +  δ_v / τ_v
ė_ω     = -e_ω / τ_ω   +  δ_ω / τ_ω
```

### LQR 控制律

```
δ  = -K · e                  (反馈)
u  = u_ff + δ                (前馈 + 反馈)
u  = clip(u, ±u_max)         (限幅)
```

- **离散化**: 矩阵指数 (ZOH) `expm([[A,B],[0,0]]·dt)` — 非前向欧拉
- **K 缓存**: 当 `|Δκ| > 0.01` 或 `|Δv| > 0.05` 时重新求解 DARE
- **零速 look-ahead**: 轨迹起步/减速段向前看 30 点取前馈速度，避免系统不可控

### 运动学积分（中点法）

```
θ_mid = θ + ω·dt/2
x += v·cos(θ_mid)·dt
y += v·sin(θ_mid)·dt
θ += ω·dt
```

## 配置说明

### LQR 参数 (`config/controller/lqr.yaml`)

```yaml
Q: [1.0, 2.0, 0.5, 0.5]   # 状态权重 [e_lat, e_yaw, e_v, e_omega]
R: [0.1, 0.1]              # 控制权重 [delta_v, delta_omega]

tuner:
  enabled: false
  method: bayesian          # bayesian / grid / random
  n_trials: 50
  q_range: [0.1, 10.0]
  r_range: [0.01, 1.0]
```

### 机器人参数 (`config/robot.yaml`)

```yaml
wheel_base: 0.3       # 轮间距 (m)
wheel_radius: 0.05    # 轮半径 (m)
max_v: 1.5            # 最大线速度 (m/s)
max_omega: 3.0        # 最大角速度 (rad/s)
tau_v: 0.1            # 线速度执行器时间常数 (s)
tau_omega: 0.08       # 角速度执行器时间常数 (s)
v_acc_max: 3.0        # 最大线加速度 (m/s²)
w_acc_max: 10.0       # 最大角加速度 (rad/s²)
```

## 测试

```bash
# 运行全部测试（单元 + 集成）
python -m pytest tests/ -v

# 仅运行集成测试（端到端收敛验证）
python -m pytest tests/test_integration.py -v
```

集成测试覆盖 4 种轨迹类型（Circle / Straight / Sine / Bezier），验证完整跟踪与误差阈值，全部使用 `seed=42` 保证可复现。

## 扩展指南

### 添加新的控制器

```python
from wheel_control.control.base import ControllerBase, ControlOutput

class MyController(ControllerBase):
    def compute_control(self, state, ref_trajectory, nearest_idx):
        # state: [x, y, theta, vx, omega]
        # ref_trajectory: (N, 7) — [x, y, yaw, vx, vy, omega, kappa]
        return ControlOutput(v_cmd=..., omega_cmd=...)

    def reset(self):
        pass
```

### 添加新的轨迹生成器

```python
from wheel_control.trajectory.base import TrajectoryBase

class MyTrajectory(TrajectoryBase):
    def __init__(self, dt=0.02, max_v=1.5, max_omega=3.0, max_acc=1.0, n_points=600):
        self.dt, self.max_v, self.max_omega, self.max_acc, self.n_points = \
            dt, max_v, max_omega, max_acc, n_points

    def generate(self, rng=None):
        x, y = ...  # 生成几何路径
        yaw, vx, omega, kappa = self.compute_derivatives(
            x, y, self.dt, self.max_v, self.max_omega, self.max_acc
        )
        vy = np.zeros_like(x)
        return np.stack([x, y, yaw, vx, vy, omega, kappa], axis=1)
```

## 可视化

```python
from wheel_control.simulation import RealtimeVisualizer

visualizer = RealtimeVisualizer(realtime=True, update_interval=5)
visualizer.setup(trajectory)

# 在仿真循环中增量更新（内部使用 set_data，无重绘开销）
visualizer.update(state, ref_point, errors, dt)

visualizer.show()
visualizer.save_figure("result.png")
```

## 依赖

```
numpy    >= 1.20, < 2.0
scipy    >= 1.7,  < 2.0
pyyaml   >= 5.4,  < 7.0
matplotlib >= 3.4, < 4.0
pytest   >= 6.0,  < 9.0
```

可选：`scikit-optimize >= 0.9` — 用于贝叶斯优化参数整定

## 许可证

MIT License

## 参考

- Frenet 坐标系：基于路径的自然坐标系
- LQR 控制：线性二次调节器 + 离散 Riccati 方程
- 矩阵指数离散化：ZOH (Zero-Order Hold) 精确方法
- 差分驱动运动学：非完整约束机器人模型
- 详细架构文档：[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
