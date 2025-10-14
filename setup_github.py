"""
GitHub Repository Setup Script

This script helps set up the DeepReach Model Deployment repository for GitHub.
Run this script to initialize git and prepare for GitHub upload.
"""

import os
import subprocess
import sys


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return None


def main():
    print("DeepReach Model Deployment - GitHub Setup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("deepreach_deployment.py"):
        print("❌ Please run this script from the DeepReach_Model_Deployment directory")
        return
    
    # Initialize git repository
    if not os.path.exists(".git"):
        run_command("git init", "Initialize git repository")
    else:
        print("✓ Git repository already initialized")
    
    # Add all files
    run_command("git add .", "Add all files to git")
    
    # Create initial commit
    run_command('git commit -m "Initial commit: DeepReach Model Deployment"', "Create initial commit")
    
    print("\n" + "=" * 50)
    print("NEXT STEPS:")
    print("=" * 50)
    print("1. Create a new repository on GitHub:")
    print("   - Go to https://github.com/new")
    print("   - Name it 'DeepReach-Model-Deployment' or similar")
    print("   - Don't initialize with README (we already have one)")
    print()
    print("2. Copy your trained model:")
    print("   cp ../deepreach/runs/dubins3d_gpu_final/training/checkpoints/model_current.pth ./")
    print()
    print("3. Connect to GitHub and push:")
    print("   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git")
    print("   git branch -M main")
    print("   git push -u origin main")
    print()
    print("4. Test the deployment:")
    print("   python example_usage.py")
    print()
    print("✓ Setup completed! Follow the steps above to upload to GitHub.")


if __name__ == "__main__":
    main()
