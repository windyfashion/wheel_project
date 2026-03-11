"""Custom trajectory example."""

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from wheel_control.trajectory import TrajectoryBase
from wheel_control.kinematics import DiffDriveKinematics
from wheel_control.control import LQRController
from wheel_control.simulation import SimulationEnv, RealtimeVisualizer, MetricEvaluator


class SpiralTrajectory(TrajectoryBase):
    """Custom spiral trajectory generator."""
    
    def __init__(
        self,
        dt: float = 0.02,
        max_v: float = 1.5,
        max_omega: float = 3.0,
        max_acc: float = 1.0,
        n_points: int = 600,
        spiral_growth: float = 0.5,
    ):
        self.dt = dt
        self.max_v = max_v
        self.max_omega = max_omega
        self.max_acc = max_acc
        self.n_points = n_points
        self.spiral_growth = spiral_growth
    
    def generate(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """Generate spiral trajectory."""
        t = np.linspace(0, 4 * np.pi, self.n_points)
        
        # Archimedean spiral: r = a + b * theta
        a = 0.5
        r = a + self.spiral_growth * t
        
        x = r * np.cos(t)
        y = r * np.sin(t)
        
        # Compute derivatives
        yaw, vx, omega, kappa = self.compute_derivatives(
            x, y, self.dt, self.max_v, self.max_omega, self.max_acc
        )
        vy = np.zeros(self.n_points)
        
        return np.stack([x, y, yaw, vx, vy, omega, kappa], axis=1)


def main():
    """Run custom trajectory example."""
    print("=" * 60)
    print("Custom Trajectory Example (Spiral)")
    print("=" * 60)
    
    # Parameters
    dt = 0.02
    max_v = 1.0
    max_omega = 2.0
    
    # Generate custom spiral trajectory
    print("\nGenerating spiral trajectory...")
    trajectory_gen = SpiralTrajectory(
        dt=dt,
        max_v=max_v,
        max_omega=max_omega,
        n_points=800,
        spiral_growth=0.3,
    )
    trajectory = trajectory_gen.generate()
    print(f"  Trajectory length: {len(trajectory)} points")
    
    # Create kinematics model
    kinematics = DiffDriveKinematics(
        wheel_base=0.3,
        max_v=max_v,
        max_omega=max_omega,
        tau_v=0.1,
        tau_omega=0.08,
        dt=dt,
    )
    
    # Create LQR controller
    controller = LQRController(
        dt=dt,
        Q=[1.0, 2.0, 0.5, 0.5],
        R=[0.1, 0.1],
        wheel_base=0.3,
    )
    
    # Create simulation environment
    env = SimulationEnv(
        trajectory=trajectory,
        controller=controller,
        kinematics=kinematics,
        config={
            "dt": dt,
            "max_steps": 1500,
            "lateral_limit": 0.5,
            "init_noise": 0.1,
        },
    )
    
    # Create visualizer
    visualizer = RealtimeVisualizer(realtime=True, update_interval=10)
    visualizer.setup(trajectory)
    
    # Run simulation
    print("\nRunning simulation...")
    step = 0
    state = env.reset()
    done = False
    
    while not done:
        state, done, info = env.step()
        
        nearest_idx = info.get("nearest_idx", 0)
        ref_point = trajectory[nearest_idx]
        
        visualizer.update(
            state=state,
            ref_point=ref_point,
            errors={
                "e_lat": info["e_lat"],
                "e_yaw": info["e_yaw"],
            },
            dt=dt,
        )
        
        step += 1
        
        if step % 200 == 0:
            print(f"  Step {step}: e_lat={info['e_lat']:.4f}")
    
    # Show final visualization
    print("\nShowing final visualization...")
    visualizer.show()
    
    # Compute metrics
    errors = env.get_tracking_errors()
    evaluator = MetricEvaluator()
    metrics = evaluator.compute(
        e_lat=np.array(errors["e_lat"]),
        e_yaw=np.array(errors["e_yaw"]),
        e_v=np.array(errors["e_v"]),
        e_omega=np.array(errors["e_omega"]),
    )
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Total steps: {step}")
    print(f"  RMS lateral error: {metrics['rms_lateral_error']:.4f} m")
    print(f"  Max lateral error: {metrics['max_lateral_error']:.4f} m")


if __name__ == "__main__":
    main()
