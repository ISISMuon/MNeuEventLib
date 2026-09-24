import atexit

# re-export Rust in main namespace
from MNeuEventLib.core import *

# Cleanly release GPU device resources upon Python process termination to prevent driver shutdown crashes
try:
    from MNeuEventLib.core import cleanup_gpu
    atexit.register(cleanup_gpu)
except ImportError:
    pass
