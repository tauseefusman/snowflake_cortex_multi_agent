#!/usr/bin/env python3
"""
Deployment script for Multi-Agent Snowflake Cortex Streamlit App

This script handles deployment for both standalone and Snowflake-hosted environments.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        "streamlit",
        "pandas", 
        "snowflake-connector-python",
        "langgraph",
        "langchain-core",
        "python-dotenv",
        "requests",
        "sseclient-py"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("📦 Installing missing packages...")
        
        for package in missing_packages:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        
        print("✅ All packages installed successfully!")
    else:
        print("✅ All dependencies are installed!")

def setup_environment():
    """Setup environment configuration"""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists():
        if env_example.exists():
            print("📝 Creating .env file from .env.example...")
            env_file.write_text(env_example.read_text())
            print("⚠️  Please edit .env file with your Snowflake credentials before running the app!")
        else:
            print("⚠️  No .env file found. Please create one with your Snowflake configuration.")
    else:
        print("✅ Environment file (.env) exists!")

def run_streamlit_app(port=8501, host="localhost"):
    """Run the Streamlit app"""
    app_file = "streamlit_app.py"
    
    if not Path(app_file).exists():
        print(f"❌ App file '{app_file}' not found!")
        return False
    
    print(f"🚀 Starting Streamlit app on http://{host}:{port}")
    print("📋 Make sure your .env file is configured with Snowflake credentials!")
    print("🔄 Use Ctrl+C to stop the app")
    
    try:
        subprocess.run([
            "streamlit", "run", app_file,
            "--server.port", str(port),
            "--server.address", host,
            "--server.headless", "true" if host != "localhost" else "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Error running app: {str(e)}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Deploy Multi-Agent Snowflake Cortex App")
    parser.add_argument("--port", type=int, default=8501, help="Port to run the app on")
    parser.add_argument("--host", default="localhost", help="Host to bind the app to")
    parser.add_argument("--check-deps", action="store_true", help="Only check dependencies")
    parser.add_argument("--setup-env", action="store_true", help="Only setup environment")
    
    args = parser.parse_args()
    
    print("🎯 Multi-Agent Snowflake Cortex - Deployment Script")
    print("=" * 50)
    
    # Check dependencies
    check_dependencies()
    
    if args.check_deps:
        print("✅ Dependency check complete!")
        return
    
    # Setup environment
    setup_environment()
    
    if args.setup_env:
        print("✅ Environment setup complete!")
        return
    
    # Run the app
    success = run_streamlit_app(port=args.port, host=args.host)
    
    if success:
        print("✅ App deployment successful!")
    else:
        print("❌ App deployment failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
