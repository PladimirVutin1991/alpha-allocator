import os
import subprocess
import sys
import time

def setup_environment():

    #Checks for required packages and installs them if missing.
    #This ensures the app works on any new machine/environment immediately.

    print("\n" + "="*60)
    print("SYSTEM CHECK: Verifying Dependencies...")
    print("="*60)

    # packages
    required_packages = [
        "yfinance",
        "pandas",
        "numpy",
        "scikit-learn",   
        "joblib",
        "matplotlib",
        "seaborn",
        "streamlit",
        "plotly"
    ]

    for package in required_packages:
        try:
            # checks packages 
            import_name = "sklearn" if package == "scikit-learn" else package
            __import__(import_name)
            print(f"✅ {package} is already installed.")
        except ImportError:
            print(f"⚠️ {package} not found. Installing...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✅ {package} installed successfully!")
            except subprocess.CalledProcessError:
                print(f"❌ Failed to install {package}. Please install manually.")
                sys.exit(1)



# Runs Python script located in the src/ folder.
def run_module(script_name):
    
    print(f"\n" + "="*60)
    print(f"EXECUTING: src/{script_name}")
    print("="*60)
    
    # Check if file exists 
    if not os.path.exists(f"src/{script_name}"):
        print(f"❌ Error: src/{script_name} not found!")
        sys.exit(1)

    result = subprocess.run([sys.executable, f"src/{script_name}"], capture_output=False)
    
    if result.returncode != 0:
        print(f"❌ CRITICAL ERROR in {script_name}. Pipeline stopped.")
        sys.exit(1)
    else:
        print(f"✅ {script_name} completed successfully.")

def main():
    # Install everything 
    setup_environment()
    # Ensures directories exist
    if not os.path.exists('results'): os.makedirs('results') 

    print("\nStarting ISL Strategic Allocator Pipeline...\n")
    
    # Download Data & Engineer Features
    run_module("data_processing.py")
    
    # Train the Random Forest Model
    run_module("optimize_model.py")
    
    # Check performance 
    run_module("visualize_performance.py")
    
    print("\n" + "="*60)
    print("PIPELINE FINISHED")
    print("LAUNCHING DASHBOARD")
    print("PRESS CTRL+C TO STOP THE DASHBOARD")
    print("="*60 + "\n")
    
    time.sleep(3)
    
    # Dashboard
    subprocess.run([sys.executable, "-m", "streamlit", "run", "src/app.py"])

if __name__ == "__main__":
    main()