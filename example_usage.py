"""
Example Usage of DeepReach Model Deployment

This script demonstrates how to use the trained DeepReach model for:
1. Loading the model
2. Evaluating value functions
3. Computing optimal controls
4. Generating reachability plots
5. Analyzing trajectories

Make sure to copy your trained model checkpoint to this directory first:
cp ../deepreach/runs/dubins3d_gpu_final/training/checkpoints/model_current.pth ./
"""

import numpy as np
import matplotlib.pyplot as plt
from deepreach_deployment import DeepReachModel, load_deepreach_model


def main():
    print("DeepReach Model Deployment Example")
    print("==================================")
    
    # 1. Load the trained model
    print("\n1. Loading trained model...")
    try:
        model = load_deepreach_model("model_current.pth", device='cpu')
        print("✓ Model loaded successfully!")
    except FileNotFoundError:
        print("❌ Model file not found!")
        print("Please copy your trained model checkpoint to this directory:")
        print("cp ../deepreach/runs/dubins3d_gpu_final/training/checkpoints/model_current.pth ./")
        return
    
    # 2. Evaluate value function at specific points
    print("\n2. Evaluating value function...")
    test_points = [
        (0.0, [0.0, 0.0, 0.0]),      # Origin
        (0.0, [0.5, 0.0, 0.0]),      # Near target
        (0.0, [1.0, 0.0, 0.0]),      # Far from target
        (0.5, [0.0, 0.0, 0.0]),      # Origin at t=0.5
    ]
    
    for t, state in test_points:
        value = model.evaluate_value(t, state)
        safe = model.is_safe(t, state)
        print(f"  t={t:.1f}, state={state} → V={value:.4f}, safe={safe}")
    
    # 3. Compute optimal controls
    print("\n3. Computing optimal controls...")
    for t, state in test_points:
        control = model.get_optimal_control(t, state)
        print(f"  t={t:.1f}, state={state} → u*={control:.4f}")
    
    # 4. Generate reachability plots
    print("\n4. Generating reachability plots...")
    
    # Plot at t=0, θ=0
    print("  Generating plot at t=0, θ=0...")
    model.generate_reachability_plot(time=0.0, theta=0.0, resolution=100, 
                                   save_path="reachability_t0_theta0.png")
    
    # Plot at t=0.5, θ=0
    print("  Generating plot at t=0.5, θ=0...")
    model.generate_reachability_plot(time=0.5, theta=0.0, resolution=100, 
                                   save_path="reachability_t05_theta0.png")
    
    # 5. Analyze a sample trajectory
    print("\n5. Analyzing sample trajectory...")
    
    # Create a simple trajectory (straight line with constant heading)
    t_final = 1.0
    n_points = 50
    times = np.linspace(0, t_final, n_points)
    states = np.zeros((n_points, 3))
    
    # Straight line trajectory: x(t) = 0.5*t, y(t) = 0, θ(t) = 0
    states[:, 0] = 0.5 * times  # x
    states[:, 1] = 0.0          # y
    states[:, 2] = 0.0          # θ
    
    # Analyze trajectory
    analysis = model.analyze_trajectory(states, times)
    
    print(f"  Trajectory analysis:")
    print(f"    Safety ratio: {analysis['safety_ratio']:.2%}")
    print(f"    Value range: [{analysis['min_value']:.4f}, {analysis['max_value']:.4f}]")
    print(f"    Control range: [{np.min(analysis['controls']):.4f}, {np.max(analysis['controls']):.4f}]")
    
    # Plot trajectory analysis
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Trajectory in x-y space
    axes[0, 0].plot(states[:, 0], states[:, 1], 'b-', linewidth=2, label='Trajectory')
    axes[0, 0].scatter(states[~analysis['safe_states'], 0], states[~analysis['safe_states'], 1], 
                      c='red', s=20, label='Unsafe states')
    circle = plt.Circle((0, 0), model.dynamics.goalR, color='red', alpha=0.3, label='Target')
    axes[0, 0].add_patch(circle)
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    axes[0, 0].set_title('Trajectory in State Space')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axis('equal')
    
    # Value function over time
    axes[0, 1].plot(times, analysis['values'], 'g-', linewidth=2)
    axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5, label='V=0')
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Value Function V(t,x)')
    axes[0, 1].set_title('Value Function Along Trajectory')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Optimal control over time
    axes[1, 0].plot(times, analysis['controls'], 'r-', linewidth=2)
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Optimal Control u*')
    axes[1, 0].set_title('Optimal Control Along Trajectory')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Safety over time
    safe_colors = ['green' if safe else 'red' for safe in analysis['safe_states']]
    axes[1, 1].scatter(times, [1 if safe else 0 for safe in analysis['safe_states']], 
                      c=safe_colors, s=20)
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('Safe (1) / Unsafe (0)')
    axes[1, 1].set_title('Safety Along Trajectory')
    axes[1, 1].set_ylim(-0.1, 1.1)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("trajectory_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Example completed successfully!")
    print("Generated files:")
    print("  - reachability_t0_theta0.png")
    print("  - reachability_t05_theta0.png") 
    print("  - trajectory_analysis.png")


def demonstrate_api():
    """Demonstrate the main API functions."""
    print("\n" + "="*50)
    print("API DEMONSTRATION")
    print("="*50)
    
    # Load model
    model = load_deepreach_model("model_current.pth", device='cpu')
    
    # Example 1: Check if a state is safe
    print("\nExample 1: Safety Check")
    state = [0.3, 0.0, 0.0]  # Near the target
    is_safe = model.is_safe(0.0, state)
    print(f"State {state} at t=0 is {'SAFE' if is_safe else 'UNSAFE'}")
    
    # Example 2: Get optimal control
    print("\nExample 2: Optimal Control")
    control = model.get_optimal_control(0.0, state)
    print(f"Optimal control at state {state}: u* = {control:.4f}")
    
    # Example 3: Value function evaluation
    print("\nExample 3: Value Function")
    value = model.evaluate_value(0.0, state)
    print(f"Value function at state {state}: V = {value:.4f}")
    
    # Example 4: Batch evaluation
    print("\nExample 4: Batch Evaluation")
    states = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]])
    times = np.array([0.0, 0.0, 0.0])
    
    for i, (t, state) in enumerate(zip(times, states)):
        value = model.evaluate_value(t, state)
        control = model.get_optimal_control(t, state)
        safe = model.is_safe(t, state)
        print(f"  Point {i+1}: V={value:.4f}, u*={control:.4f}, safe={safe}")


if __name__ == "__main__":
    # Run main example
    main()
    
    # Run API demonstration
    demonstrate_api()
