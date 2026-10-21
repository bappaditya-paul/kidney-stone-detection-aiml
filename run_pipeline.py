#!/usr/bin/env python3
"""
Kidney Stone Detection - Complete Pipeline

This script runs the complete pipeline for the Kidney Stone Detection project:
1. Data Preprocessing
2. Model Training
3. Model Evaluation
4. Launch Streamlit Web App (optional)
"""

import sys
import subprocess
from pathlib import Path
import time

def run_command(command, description):
    """Run a shell command with live console output and error handling."""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"⏳ Running: {command}")
    print("-" * 60)
    
    start_time = time.time()
    try:
        subprocess.run(
            command,
            shell=True,
            check=True
        )
        print(f"\n✅ Completed in {time.time() - start_time:.1f} seconds")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running command: {command}")
        print(f"Exit code: {e.returncode}")
        return False

def main():
    """Run the complete pipeline."""
    # Get the project root directory
    project_root = Path(__file__).resolve().parent
    src_dir = project_root / 'src'
    
    # 1. Run data preprocessing
    if not run_command(
        f'"{sys.executable}" "{src_dir}/preprocess.py"',
        "Step 1/3: Preprocessing Data"
    ):
        print("\n❌ Preprocessing failed. Exiting...")
        return 1
    
    # 2. Train the model
    if not run_command(
        f'"{sys.executable}" "{src_dir}/train_cnn.py"',
        "Step 2/3: Training Model"
    ):
        print("\n❌ Model training failed. Exiting...")
        return 1
    
    # 3. Evaluate the model
    if not run_command(
        f'"{sys.executable}" "{src_dir}/evaluate.py"',
        "Step 3/3: Evaluating Model"
    ):
        print("\n⚠️  Model evaluation completed with issues.")
    else:
        print("\n✅ Model evaluation completed successfully!")
    
    # 4. Ask to launch Streamlit app
    print("\n" + "="*60)
    print("🎉 Pipeline completed successfully!")
    print("="*60)
    
    launch = input("\n🚀 Would you like to launch the Streamlit app? (y/n): ")
    if launch.lower() == 'y':
        print("\nStarting Streamlit app... (Press Ctrl+C to stop)")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py"])
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
