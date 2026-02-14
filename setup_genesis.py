#!/usr/bin/env python3
"""
First-time setup wizard for Genesis.
Detects hardware, installs correct PyTorch, prepares environment.
"""
import sys
import subprocess
from pathlib import Path

SETUP_MARKER = Path.home() / ".genesis_setup_complete"

def print_header(text):
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def check_python_version():
    """Check if Python version is compatible with PyTorch."""
    print_header("Python Version Check")

    major = sys.version_info.major
    minor = sys.version_info.minor

    print(f"Python version: {major}.{minor}")

    # PyTorch pre-built wheels are available for Python 3.9-3.13
    if major != 3 or minor < 9:
        print("\n❌ Python 3.9 or newer is required")
        print("   Your version is too old")
        return False
    elif minor > 13:
        print("\n❌ Python 3.14+ is not yet supported by PyTorch")
        print("   PyTorch pre-built wheels are only available for Python 3.9-3.13")
        print("\n💡 Solution:")
        print("   Install Python 3.12 for this project:")
        print("   - Fedora: sudo dnf install python3.12")
        print("   - Ubuntu: sudo apt install python3.12")
        print("   - Then create a new venv: python3.12 -m venv .venv")
        return False
    elif minor < 11:
        print(f"\n⚠️  Warning: Python 3.{minor} is older")
        print("   Python 3.11-3.13 is recommended for best compatibility")
        response = input("Continue anyway? (y/n): ")
        return response.lower() == 'y'
    else:
        print(f"\n✓ Python 3.{minor} is supported")
        return True

def detect_gpu_hardware():
    """Detect available GPU hardware (NVIDIA or AMD)."""
    print_header("Hardware Detection")
    
    # Check for NVIDIA GPU
    nvidia_gpu = None
    nvidia_vram = 0
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            line = result.stdout.strip()
            if line:
                parts = line.split(',')
                nvidia_gpu = parts[0].strip()
                nvidia_vram = float(parts[1].strip().split()[0]) / 1024  # Convert MB to GB
                print(f"✓ NVIDIA GPU detected: {nvidia_gpu} ({nvidia_vram:.1f} GB)")
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass
    
    # Check for AMD GPU
    amd_gpu = None
    amd_vram = 0
    try:
        result = subprocess.run(['rocm-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            # Get detailed GPU info
            result2 = subprocess.run(
                ['rocm-smi', '--showproductname'],
                capture_output=True, text=True, timeout=5
            )
            if result2.returncode == 0:
                # Check for specific GPU models
                if '0x7590' in result2.stdout:
                    amd_gpu = "AMD Radeon RX 9060 XT"
                    amd_vram = 16.0
                elif '0x744c' in result2.stdout or '7900' in result2.stdout:
                    amd_gpu = "AMD Radeon RX 7900 XTX"
                    amd_vram = 24.0
                elif '0x73bf' in result2.stdout or '7900 XT' in result2.stdout:
                    amd_gpu = "AMD Radeon RX 7900 XT"
                    amd_vram = 20.0
                else:
                    amd_gpu = "AMD Radeon GPU"
                    amd_vram = 8.0  # Default estimate
            
            if amd_gpu:
                print(f"✓ AMD GPU detected: {amd_gpu} ({amd_vram:.1f} GB)")
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass
    
    if not nvidia_gpu and not amd_gpu:
        print("⚠️  No GPU detected")
        print("   Genesis will run on CPU (very slow)")
        return None
    
    # Return the detected GPU
    if nvidia_gpu and amd_gpu:
        print("\n⚠️  Both NVIDIA and AMD GPUs detected!")
        print("   Using NVIDIA by default")
        return ("nvidia", nvidia_gpu, nvidia_vram)
    elif nvidia_gpu:
        return ("nvidia", nvidia_gpu, nvidia_vram)
    else:
        return ("amd", amd_gpu, amd_vram)

def check_pytorch_installation(gpu_type):
    """Check if PyTorch is installed and if it matches the GPU."""
    print_header("PyTorch Check")
    
    try:
        import torch
        version = torch.__version__
        cuda_version = torch.version.cuda
        hip_version = getattr(torch.version, 'hip', None)
        
        print(f"✓ PyTorch installed: {version}")
        print(f"  Built with CUDA: {cuda_version if cuda_version else 'No'}")
        print(f"  Built with ROCm: {hip_version if hip_version else 'No'}")
        
        # Check PyTorch version (need 2.6+ for torch.load security fix CVE-2025-32434)
        version_parts = version.split('+')[0].split('.')  # Remove +rocm/+cu suffix
        major = int(version_parts[0])
        minor = int(version_parts[1]) if len(version_parts) > 1 else 0

        if major < 2 or (major == 2 and minor < 6):
            print(f"\n⚠️  WARNING: PyTorch {version} is older than 2.6")
            print("   PyTorch 2.6+ is required for torch.load security (CVE-2025-32434)")
            print("   Your current version may have security vulnerabilities")
            return False

        # Check if PyTorch matches GPU
        if gpu_type == "nvidia" and not cuda_version:
            print("\n❌ Wrong PyTorch: You have NVIDIA GPU but PyTorch is not built with CUDA")
            return False
        elif gpu_type == "amd" and not hip_version:
            print("\n❌ Wrong PyTorch: You have AMD GPU but PyTorch is not built with ROCm")
            return False
        elif gpu_type is None and (cuda_version or hip_version):
            print("\n⚠️  PyTorch has GPU support but no GPU detected")
            return True
        
        # Test if PyTorch can see the GPU
        if gpu_type and not torch.cuda.is_available():
            print("\n❌ PyTorch cannot detect GPU")
            return False
        
        print("\n✓ PyTorch configuration is correct")
        return True
        
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def install_pytorch(gpu_type):
    """Install the correct PyTorch version for the detected GPU."""
    print_header("PyTorch Installation")
    
    print("\nThis may take several minutes...")
    
    # Uninstall existing PyTorch
    print("\n1. Removing existing PyTorch...")
    subprocess.run(
        [sys.executable, '-m', 'pip', 'uninstall', '-y', 'torch', 'torchvision', 'torchaudio', 'triton'],
        capture_output=True
    )
    
    # Install new PyTorch
    print("2. Installing PyTorch...")

    if gpu_type == "nvidia":
        print("Installing PyTorch with CUDA 12.1 support...")
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install',
             'torch', 'torchvision', 'torchaudio',
             '--index-url', 'https://download.pytorch.org/whl/cu121'],
            capture_output=False
        )
    elif gpu_type == "amd":
        print("Installing PyTorch with ROCm 7.1 support...")
        # Use PyTorch's official ROCm index (supports multiple distributions and Python versions)
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install',
             'torch', 'torchvision', 'torchaudio',
             '--index-url', 'https://download.pytorch.org/whl/rocm7.1'],
            capture_output=False
        )
    else:
        print("Installing CPU-only PyTorch...")
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install',
             'torch', 'torchvision', 'torchaudio',
             '--index-url', 'https://download.pytorch.org/whl/cpu'],
            capture_output=False
        )

    if result.returncode != 0:
        print("\n❌ PyTorch installation failed")
        return False
    
    print("\n✓ PyTorch installation complete")
    return True

def verify_pytorch_works():
    """Verify PyTorch can detect GPU."""
    print_header("Verification")
    
    try:
        # Force reimport after installation
        if 'torch' in sys.modules:
            del sys.modules['torch']
        
        import torch
        print(f"PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print(f"✓ GPU detected: {device_count} device(s)")
            for i in range(device_count):
                name = torch.cuda.get_device_name(i)
                vram = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"  GPU {i}: {name} ({vram:.1f} GB)")
            
            # Detect platform
            platform = "CUDA"
            if hasattr(torch.version, 'hip') and torch.version.hip:
                platform = "ROCm"
            print(f"  Platform: {platform}")
            return True
        else:
            print("⚠️  No GPU detected by PyTorch")
            return False
            
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

def install_genesis_packages():
    """Install Genesis and its dependencies."""
    print_header("Genesis Installation")
    
    print("Installing Genesis dependencies...")
    result = subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '-e', '.'],
        capture_output=False
    )
    
    if result.returncode != 0:
        print("\n❌ Genesis installation failed")
        return False
    
    print("\n✓ Genesis dependencies installed")
    return True

def check_vram_warnings(vram_gb):
    """Show warnings based on VRAM size."""
    print_header("VRAM Configuration")
    
    print(f"Detected VRAM: {vram_gb:.1f} GB")
    
    if vram_gb < 8:
        print("\n⚠️  WARNING: Less than 8GB VRAM")
        print("   - Video generation will be very slow or may fail")
        print("   - Training is not recommended")
        print("   - Caption generation may work with small models")
        return False
    elif vram_gb < 16:
        print("\n⚠️  NOTICE: Less than 16GB VRAM")
        print("   - Use 1.3B model for generation (not 5B or 14B)")
        print("   - Training will be slow, use small batch sizes")
        print("   - Caption generation should work fine")
        return True
    elif vram_gb < 24:
        print("\n✓ Good: 16-24GB VRAM")
        print("   - 5B model recommended for generation")
        print("   - Training will work with gradient accumulation")
        print("   - All caption models supported")
        return True
    else:
        print("\n✓ Excellent: 24GB+ VRAM")
        print("   - All models supported (1.3B, 5B, 14B)")
        print("   - Training with larger batch sizes")
        print("   - All features fully supported")
        return True

def run_first_time_setup():
    """Run the complete first-time setup process."""
    print("\n" + "=" * 70)
    print("  🎬 GENESIS - FIRST TIME SETUP")
    print("=" * 70)
    print("\nThis wizard will:")
    print("  1. Check Python version compatibility")
    print("  2. Detect your GPU hardware")
    print("  3. Install the correct PyTorch version")
    print("  4. Install Genesis dependencies")
    print("  5. Verify everything works")
    print("\nThis is a one-time process (5-10 minutes)")
    print("\nNote: Models will be downloaded on first use (not during setup)")
    
    input("\nPress Enter to start setup...")
    
    # Step 0: Check Python version
    if not check_python_version():
        print("\n❌ Setup cannot continue with incompatible Python version")
        return False

    # Step 1: Detect GPU
    gpu_info = detect_gpu_hardware()
    
    if gpu_info:
        gpu_type, gpu_name, vram_gb = gpu_info
        can_continue = check_vram_warnings(vram_gb)
        
        if not can_continue:
            print("\n⚠️  Your GPU may not have enough VRAM")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                print("\nSetup cancelled")
                return False
    else:
        gpu_type = None
        print("\n⚠️  No GPU detected - Genesis will run on CPU")
        response = input("Continue with CPU-only mode? (y/n): ")
        if response.lower() != 'y':
            print("\nSetup cancelled")
            return False
    
    # Step 2: Check PyTorch
    pytorch_ok = check_pytorch_installation(gpu_type)
    
    if not pytorch_ok:
        print("\nPyTorch needs to be installed/reinstalled")
        response = input("Install correct PyTorch version? (y/n): ")
        if response.lower() != 'y':
            print("\nSetup cancelled")
            return False
        
        if not install_pytorch(gpu_type):
            return False
        
        if not verify_pytorch_works():
            print("\n❌ PyTorch verification failed")
            print("   Please check your GPU drivers and try again")
            return False
    
    # Step 3: Install Genesis packages
    print("\n")
    response = input("Install Genesis dependencies? (y/n): ")
    if response.lower() != 'y':
        print("\nSetup cancelled")
        return False
    
    if not install_genesis_packages():
        return False
    
    # Final verification with Genesis
    print_header("Final Verification")
    try:
        from src.core.config import get_device_info, get_optimal_device
        device = get_optimal_device()
        device_type, device_name, vram_gb = get_device_info()
        
        print(f"✓ Genesis device detection working")
        print(f"  Device: {device}")
        print(f"  Name: {device_name}")
        print(f"  VRAM: {vram_gb:.1f} GB")
    except Exception as e:
        print(f"⚠️  Warning: Genesis detection test failed: {e}")
    
    # Mark setup as complete
    SETUP_MARKER.touch()
    
    print_header("Setup Complete!")
    print("\n✅ Genesis is ready to use!")
    print("\nYou can now:")
    print("  - Generate videos from text prompts")
    print("  - Fine-tune models on your videos")
    print("  - Create captions for training data")
    print("\nTo start Genesis, run:")
    print("  python run_genesis.py")
    
    return True

def main():
    """Main entry point for setup."""
    try:
        success = run_first_time_setup()
        if success:
            print("\n" + "=" * 70)
            response = input("\nLaunch Genesis now? (y/n): ")
            if response.lower() == 'y':
                # Launch Genesis
                print("\nLaunching Genesis...\n")
                import subprocess
                subprocess.run([sys.executable, 'run_genesis.py'])
            return 0
        else:
            print("\n" + "=" * 70)
            print("Setup incomplete. Run this script again when ready.")
            return 1
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user")
        return 1
    except Exception as e:
        print(f"\n❌ Setup error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
