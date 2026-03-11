"""Basic trajectory tracking example using LQR controller."""

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from wheel_control.trajectory import BezierSplineTrajectory
from wheel_control.kinematics import DiffDriveKinematics
from wheel_control.control import LQRController
from wheel_control.simulation import SimulationEnv, RealtimeVisualizer, MetricEvaluator
from wheel_control.utils import ConfigManager


def main():
    """Run basic trajectory tracking example."""
    print("=" * 60)
    print("Basic Trajectory Tracking Example")
    print("=" * 60)
    
    # Load configuration
    config_path = Path(__file__).parent.parent / "config"
    config = ConfigManager(config_path)
    
    # Get parameters
    robot_cfg = config.load("robot")
    sim_cfg = config.load("simulation")
    lqr_cfg = config.load("controller/lqr")
    
    # Generate trajectory
    print("\nGenerating trajectory...")
    trajectory_gen = BezierSplineTrajectory(
        dt=sim_cfg["dt"],
        max_v=robot_cfg["max_v"],
        max_omega=robot_cfg["max_omega"],
        max_acc=robot_cfg.get("v_acc_max", 1.0),
        n_points=600,
    )
    trajectory = trajectory_gen.generate()
    print(f"  Trajectory length: {len(trajectory)} points")
    
    # Create kinematics model
    print("\nCreating kinematics model...")
    kinematics = DiffDriveKinematics(
        wheel_base=robot_cfg["wheel_base"],
        wheel_radius=robot_cfg.get("wheel_radius", 0.05),
        max_v=robot_cfg["max_v"],
        max_omega=robot_cfg["max_omega"],
        tau_v=robot_cfg["tau_v"],
        tau_omega=robot_cfg["tau_omega"],
        v_acc_max=robot_cfg["v_acc_max"],
        w_acc_max=robot_cfg["w_acc_max"],
        dt=sim_cfg["dt"],
    )
    
    # Create LQR controller
    print("\nCreating LQR controller...")
    Q = lqr_cfg.get("Q", [1.0, 2.0, 0.5, 0.5])
    R = lqr_cfg.get("R", [0.1, 0.1])
    
    controller = LQRController(
        dt=sim_cfg["dt"],
        Q=Q,
        R=R,
        wheel_base=robot_cfg["wheel_base"],
    )
    print(f"  Q: {Q}")
    print(f"  R: {R}")
    
    # Create simulation environment
    print("\nCreating simulation environment...")
    env = SimulationEnv(
        trajectory=trajectory,
        controller=controller,
        kinematics=kinematics,
        config={
            "dt": sim_cfg["dt"],
            "max_steps": sim_cfg["max_steps"],
            "lateral_limit": sim_cfg["lateral_limit"],
            "init_noise": sim_cfg.get("init_noise", 0.1),
        },
    )
    
    # Create visualizer
    visualizer = RealtimeVisualizer(realtime=True, update_interval=10)
    visualizer.setup(trajectory)
    
    # Run simulation with visualization
    print("\nRunning simulation...")
    print("  (Close the plot window to end simulation early)")
    
    step = 0
    state = env.reset()
    done = False
    
    while not done and step < sim_cfg["max_steps"]:
        state, done, info = env.step()
        
        # Get reference point for visualization
        nearest_idx = info.get("nearest_idx", 0)
        ref_point = trajectory[nearest_idx]
        
        # Update visualizer
        visualizer.update(
            state=state,
            ref_point=ref_point,
            errors={
                "e_lat": info["e_lat"],
                "e_yaw": info["e_yaw"],
            },
            dt=sim_cfg["dt"],
        )
        
        step += 1
        
        if step % 100 == 0:
            print(f"  Step {step}: e_lat={info['e_lat']:.4f}, e_yaw={info['e_yaw']:.4f}")
    
    # Show final visualization
    print("\nShowing final visualization...")
    visualizer.show()
    
    # Compute final metrics
    print("\nComputing metrics...")
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
    print(f"  Success: {info.get('finished', False)}")
    print(f"  Final lateral error: {info['e_lat']:.4f} m")
    print(f"  Final yaw error: {info['e_yaw']:.4f} rad")
    print(f"\n  RMS lateral error: {metrics['rms_lateral_error']:.4f} m")
    print(f"  Max lateral error: {metrics['max_lateral_error']:.4f} m")
    print(f"  RMS yaw error: {metrics['rms_yaw_error']:.4f} rad")
    print(f"  RMS velocity error: {metrics['rms_velocity_error']:.4f} m/s")
    
    # Export log
    log_path = env.export_log()
    print(f"\n  Log exported to: {log_path}")


if __name__ == "__main__":
    main()
