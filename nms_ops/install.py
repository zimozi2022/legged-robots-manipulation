#!/usr/bin/env python3
"""
Installation script for nms_ops C++ extension
"""
import subprocess
import sys
import os

def install_dependencies():
    """Install required Python packages"""
    print("Installing dependencies...")
    
    packages = ["numpy", "pybind11"]
    
    for package in packages:
        try:
            __import__(package)
            print(f"✅ {package} already installed")
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} installed")

def build_extension():
    """Build the C++ extension"""
    print("\nBuilding C++ extension...")
    
    # Change to the nms_ops directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Build the extension
    result = subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"], 
                          capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ C++ extension built successfully")
        return True
    else:
        print("❌ Failed to build C++ extension")
        print("Error output:")
        print(result.stderr)
        return False

def run_tests():
    """Run the test suite"""
    print("\nRunning tests...")
    
    result = subprocess.run([sys.executable, "test_nms_ops.py"], 
                          capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ All tests passed")
        return True
    else:
        print("❌ Some tests failed")
        print("Test output:")
        print(result.stdout)
        print(result.stderr)
        return False

def main():
    """Main installation process"""
    print("NMS Operations C++ Extension Installer")
    print("=" * 40)
    
    try:
        # Install dependencies
        install_dependencies()
        
        # Build extension
        if not build_extension():
            sys.exit(1)
        
        # Run tests
        if not run_tests():
            print("⚠️  Extension built but tests failed. Please check the implementation.")
        
        print("\n🎉 Installation completed successfully!")
        print("\nUsage example:")
        print("  import nms_ops_cpp")
        print("  help(nms_ops_cpp.scale_ray_nms_box3d)")
        print("\nRun demo:")
        print("  python demo.py")
        
    except KeyboardInterrupt:
        print("\n❌ Installation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Installation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()