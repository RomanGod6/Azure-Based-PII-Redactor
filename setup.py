#!/usr/bin/env python3
"""
PII Redactor Pro - Modern Installation Script
Automatically installs Poetry and sets up the project
"""

import sys
import os
import subprocess
import platform
from pathlib import Path

def run_command(cmd, shell=False):
    """Run a command and return success status"""
    try:
        result = subprocess.run(cmd, shell=shell, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def check_python():
    """Check if Python 3.9+ is available"""
    print("🐍 Checking Python version...")
    
    if sys.version_info < (3, 9):
        print(f"❌ Python 3.9+ required, found {sys.version}")
        print("Pandas 2.1.4+ requires Python 3.9 or later")
        return False
    
    print(f"✅ Python {sys.version.split()[0]} detected")
    return True

def install_poetry():
    """Install Poetry if not present"""
    print("📦 Checking Poetry installation...")
    
    # Check if poetry is already installed
    success, stdout, stderr = run_command(["poetry", "--version"])
    if success:
        print(f"✅ Poetry already installed: {stdout.strip()}")
        return True
    
    print("🔄 Installing Poetry...")
    print("This may take a few minutes...")
    
    if platform.system() == "Windows":
        # Windows installation
        cmd = ['powershell', '-Command', 
               "(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -"]
        success, stdout, stderr = run_command(cmd, shell=True)
    else:
        # Unix/Linux/macOS installation
        cmd = ["curl", "-sSL", "https://install.python-poetry.org", "|", "python3", "-"]
        success, stdout, stderr = run_command(" ".join(cmd), shell=True)
    
    if not success:
        print(f"❌ Failed to install Poetry: {stderr}")
        print("Please install Poetry manually: https://python-poetry.org/docs/#installation")
        print("\nAlternative installation methods:")
        print("Windows: pip install poetry")
        print("macOS: brew install poetry") 
        print("Linux: pip install poetry")
        return False
    
    print("✅ Poetry installed successfully!")
    print("Note: You may need to restart your terminal for Poetry to be available in PATH")
    return True

def setup_environment():
    """Set up the .env file"""
    print("⚙️ Setting up environment...")
    
    env_file = Path(".env")
    env_template = Path(".env.template")
    
    if not env_file.exists() and env_template.exists():
        env_file.write_text(env_template.read_text())
        print("✅ Created .env from template")
    elif env_file.exists():
        print("✅ .env file already exists")
    else:
        print("⚠️ No .env.template found - you'll need to configure Azure credentials manually")
    
    return True

def install_dependencies():
    """Install project dependencies using Poetry"""
    print("📋 Installing dependencies...")
    print("This may take several minutes for the first run...")
    
    # First try regular install
    success, stdout, stderr = run_command(["poetry", "install"])
    if not success:
        print(f"❌ Failed to install dependencies: {stderr}")
        print("\nTrying alternative approach...")
        
        # Try installing without dev dependencies first
        success, stdout, stderr = run_command(["poetry", "install", "--only", "main"])
        if not success:
            print(f"❌ Failed to install main dependencies: {stderr}")
            return False
        else:
            print("✅ Main dependencies installed successfully!")
            print("⚠️ Development dependencies skipped due to version conflicts")
            return True
    
    print("✅ All dependencies installed successfully!")
    return True

def main():
    """Main installation process"""
    print("=" * 60)
    print("🎯 PII Redactor Pro - Modern Setup")
    print("=" * 60)
    print()
    
    # Check requirements
    if not check_python():
        print("\n💡 Tip: If you have multiple Python versions installed,")
        print("   try running with a specific version: python3.9 setup.py")
        sys.exit(1)
    
    # Install Poetry
    if not install_poetry():
        print("\n💡 If Poetry installation fails, you can:")
        print("   1. Install via pip: pip install poetry")
        print("   2. Use conda: conda install poetry")
        print("   3. Download from: https://python-poetry.org/docs/#installation")
        sys.exit(1)
    
    # Setup environment
    if not setup_environment():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\n💡 If dependencies fail to install:")
        print("   1. Try: poetry install --only main")
        print("   2. Use pip fallback: pip install -r requirements.txt")
        sys.exit(1)
    
    print()
    print("🎉 Setup Complete!")
    print("=" * 40)
    print()
    print("Next steps:")
    print("• Run the app: make run  (or: poetry run python pii_redactor_app.py)")
    print("• See all commands: make help")
    print("• Development mode: make dev")
    print()
    
    # Ask if user wants to run the app
    response = input("Would you like to start the app now? (y/N): ").strip().lower()
    if response in ['y', 'yes']:
        print("\n🚀 Starting PII Redactor Pro...")
        success, stdout, stderr = run_command(["poetry", "run", "python", "pii_redactor_app.py"])
        if not success:
            print(f"❌ Failed to start app: {stderr}")
            print("Try running manually: poetry run python pii_redactor_app.py")

if __name__ == "__main__":
    main()
