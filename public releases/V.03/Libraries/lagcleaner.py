import gc
import os
import sys
import subprocess
import psutil
import threading
import time
from termcolor import colored, cprint



def clear_vram_and_reset(devmode=False):
    """
    Comprehensive function to clear VRAM, system memory, and reset resources
    Call this at the end of your script to prevent lag buildup
    """
    
    try:
        # 1. Force Python garbage collection multiple times
        for i in range(3):
            gc.collect()
            time.sleep(0.1)
        
        # 2. Clear CUDA/GPU memory if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                if devmode == True:
                    cprint("ʕ•ᴥ•ʔ ✓ CUDA VRAM cleared", 'yellow', attrs=["bold"])
        except ImportError:
            pass
        
        # 3. Clear TensorFlow GPU memory if available
        try:
            import tensorflow as tf
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.reset_memory_growth(gpu)
                if devmode == True:
                    cprint("ʕ•ᴥ•ʔ ✓ TensorFlow GPU memory cleared", 'yellow', attrs=["bold"])
        except ImportError:
            pass
        
        # 4. Clear PyTorch-related modules from memory
        modules_to_remove = []
        for name, module in sys.modules.items():
            if any(keyword in name.lower() for keyword in [
                'torch', 'cuda', 'f5_tts', 'vocos', 'transformers', 
                'torchaudio', 'librosa', 'soundfile', 'numpy'
            ]):
                modules_to_remove.append(name)
        
        for module_name in modules_to_remove:
            if module_name in sys.modules:
                del sys.modules[module_name]
        
        # 5. Force garbage collection again after module cleanup
        gc.collect()
        
        # 6. Clear system page cache (Linux/Mac only)
        if os.name != 'nt':  # Not Windows
            try:
                # This requires sudo, so it might not work in all environments
                subprocess.run(['sync'], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print("✓ System cache synced")
            except:
                pass
        
        # 7. Force process memory cleanup
        try:
            process = psutil.Process(os.getpid())
            # Get current memory info
            memory_info = process.memory_info()
            if devmode == True:
                cprint(f"ʕ•ᴥ•ʔ ✓ Memory usage: {memory_info.rss / 1024 / 1024:.1f} MB", 'yellow', attrs=["bold"])
        except ImportError:
            pass
        
        # 8. Windows-specific VRAM clearing
        if os.name == 'nt':  # Windows
            try:
                # Use Windows API to clear graphics memory
                subprocess.run([
                    'powershell', '-Command',
                    'Get-Process | Where-Object {$_.ProcessName -like "*python*"} | ForEach-Object {[System.GC]::Collect()}'
                ], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if devmode == True:
                    cprint("ʕ•ᴥ•ʔ ✓ Windows memory cleanup attempted", 'yellow', attrs=["bold"])
            except:
                pass
        
        # 9. Final aggressive cleanup
        gc.disable()
        gc.enable()
        gc.collect()

        cprint("ʕ•ᴥ•ʔ Resource cleanup complete!", 'yellow', attrs=["bold"])
        
    except Exception as e:
        print(f"Warning: Some cleanup operations failed: {e}")
        # Still do basic cleanup even if advanced methods fail
        gc.collect()

def emergency_memory_cleanup():
    """
    Emergency function for when system is severely lagged
    More aggressive cleanup that might temporarily affect performance
    """
    print("Emergency memory cleanup...")
    
    try:
        # Kill any lingering F5-TTS or Python processes (be careful with this!)
        if os.name == 'nt':  # Windows
            subprocess.run([
                'taskkill', '/F', '/IM', 'python.exe', '/T'
            ], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:  # Linux/Mac
            # This is very aggressive - only use if really needed
            subprocess.run([
                'pkill', '-f', 'f5_tts'
            ], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Clear all caches
        clear_vram_and_reset()
        
        print("✓ Emergency cleanup complete!")
        
    except Exception as e:
        print(f"Emergency cleanup error: {e}")

def monitor_and_cleanup():
    """
    Background monitor that automatically cleans up if memory usage gets too high
    Call this at the start of your script to prevent lag buildup
    """
    def memory_monitor():
        try:
            process = psutil.Process(os.getpid())
            while True:
                memory_mb = process.memory_info().rss / 1024 / 1024
                
                # If memory usage exceeds 2GB, do cleanup
                if memory_mb > 2048:
                    print(f"High memory usage detected: {memory_mb:.1f} MB - cleaning up...")
                    clear_vram_and_reset()
                    time.sleep(5)  # Wait before next check
                
                time.sleep(10)  # Check every 10 seconds
                
        except ImportError:
            # psutil not available, skip monitoring
            pass
        except Exception:
            # Monitor failed, continue silently
            pass
    
    # Start monitor in background thread
    monitor_thread = threading.Thread(target=memory_monitor, daemon=True)
    monitor_thread.start()

# Usage examples:
def f5_with_cleanup(gentext="I apologize but the voice model program has failed in some way. Please try again and report this issue if it persists."):
    """
    Your F5 function with automatic cleanup
    """
    try:
        # Your original F5 function code here
        print("Starting F5-TTS generation...")
        # run_f5_tts_inference(gentext)
        print("Playback starting...")
        # play_wav()
        print("Complete!")
        
    finally:
        # Always clean up, even if something fails
        clear_vram_and_reset()

if __name__ == "__main__":
    # Test the cleanup function
    print("Testing VRAM cleanup...")
    clear_vram_and_reset()
    
    # Example: Start memory monitoring (optional)
    # monitor_and_cleanup()
    
    # Your regular code here...
    
    # Always clean up at the end
    clear_vram_and_reset()