"""LQR parameter tuning example."""

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from wheel_control.trajectory import BezierSplineTrajectory
from wheel_control.kinematics import DiffDriveKinematics
from wheel_control.control import LQRController
from wheel_control.control.tuner import LQRTuner
from wheel_control.utils import ConfigManager


def main():
    """Run LQR parameter tuning example."""
    print("=" * 60)
    print("LQR Parameter Tuning Example")
    print("=" * 60)
    
    # Load configuration
    config_path = Path(__file__).parent.parent / "config"
    config = ConfigManager(config_path)
    
    # Get parameters
    robot_cfg = config.load("robot")
    sim_cfg = config.load("simulation")
    lqr_cfg = config.load("controller/lqr")
    
    # Generate trajectory for tuning
    print("\nGenerating tuning trajectory...")
    trajectory_gen = BezierSplineTrajectory(
        dt=sim_cfg["dt"],
        max_v=robot_cfg["max_v"],
        max_omega=robot_cfg["max_omega"],
        max_acc=robot_cfg.get("v_acc_max", 1.0),
        n_points=400,
    )
    trajectory = trajectory_gen.generate()
    
    # Create kinematics model
    kinematics = DiffDriveKinematics(
        wheel_base=robot_cfg["wheel_base"],
        max_v=robot_cfg["max_v"],
        max_omega=robot_cfg["max_omega"],
        tau_v=robot_cfg["tau_v"],
        tau_omega=robot_cfg["tau_omega"],
        v_acc_max=robot_cfg["v_acc_max"],
        w_acc_max=robot_cfg["w_acc_max"],
        dt=sim_cfg["dt"],
    )
    
    # Create controller with initial weights
    initial_Q = [1.0, 2.0, 0.5, 0.5]
    initial_R = [0.1, 0.1]
    
    controller = LQRController(
        dt=sim_cfg["dt"],
        Q=initial_Q,
        R=initial_R,
        tau_v=robot_cfg["tau_v"],
        tau_omega=robot_cfg["tau_omega"],
        max_v=robot_cfg["max_v"],
        max_omega=robot_cfg["max_omega"],
    )
    
    # Create tuner
    print("\nCreating LQR tuner...")
    tuner = LQRTuner(
        controller=controller,
        kinematics=kinematics,
        method="random",  # Use "bayesian" if scikit-optimize is installed
        metric="rms_lateral_error",
    )
    
    # Run tuning
    print("\nRunning parameter tuning (this may take a while)...")
    n_trials = 20  # Increase for better results
    
    result = tuner.tune(
        trajectory=trajectory,
        n_trials=n_trials,
        q_range=(0.1, 5.0),
        r_range=(0.01, 0.5),
        dt=sim_cfg["dt"],
        max_steps=500,
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("TUNING RESULTS")
    print("=" * 60)
    print(f"\n  Number of trials: {n_trials}")
    print(f"\n  Initial Q: {initial_Q}")
    print(f"  Initial R: {initial_R}")
    print(f"\n  Best Q: {result.best_Q}")
    print(f"  Best R: {result.best_R}")
    print(f"\n  Best metric ({tuner.metric}): {result.best_metric:.6f}")
    
    # Print trial history
    print("\n  Top 5 trials:")
    trials = sorted(result.all_trials, key=lambda x: x["metric_value"])
    for i, trial in enumerate(trials[:5]):
        print(f"    {i+1}. Q={trial['params']['Q']}, R={trial['params']['R']}")
        print(f"       Metric: {trial['metric_value']:.6f}")
    
    # Save best parameters
    print("\n  Saving best parameters to config...")
    lqr_cfg["Q"] = result.best_Q
    lqr_cfg["R"] = result.best_R
    config.save("controller/lqr", lqr_cfg)
    print("  Saved!")


if __name__ == "__main__":
    main()
