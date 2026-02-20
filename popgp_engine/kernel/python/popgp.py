import ctypes
import os
import sys
import numpy as np
from ctypes import POINTER, c_int, c_float, c_double, c_void_p

# Load the Shared Library
def _load_library():
    # 1. Add DLL directories (Windows)
    if os.name == "nt":
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Add local directory
        try:
            os.add_dll_directory(base_dir)
            print(f"Added Local Dir: {base_dir}")
        except Exception as e:
            print(f"Warning: Could not add local dir: {e}")

        # Add CUDA bin directory
        cuda_path = os.environ.get("CUDA_PATH")
        if cuda_path:
            cuda_bin = os.path.join(cuda_path, "bin")
            if os.path.exists(cuda_bin):
                try:
                    os.add_dll_directory(cuda_bin)
                    print(f"Added CUDA bin: {cuda_bin}")
                except Exception as e:
                    print(f"Warning: Could not add CUDA bin: {e}")
        
        # Add VCPKG bin directory
        vcpkg_bin = os.path.abspath(os.path.join(base_dir, "../../build/vcpkg_installed/x64-windows/bin"))
        if os.path.exists(vcpkg_bin):
            try:
                os.add_dll_directory(vcpkg_bin)
                print(f"Added VCPKG bin: {vcpkg_bin}")
            except Exception as e:
                print(f"Warning: Could not add VCPKG bin: {e}")

    # Search paths: 
    lib_name = "phase_flow"
    if os.name == "nt":
        lib_name += ".dll"
    elif sys.platform == "darwin":
        lib_name = "lib" + lib_name + ".dylib"
    else:
        lib_name = "lib" + lib_name + ".so"
        
    search_dirs = [
        os.path.dirname(os.path.abspath(__file__)),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../build/kernel/Release"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../build/popgp_engine/kernel"),
    ]
    
    for d in search_dirs:
        path = os.path.join(d, lib_name)
        if os.path.exists(path):
            try:
                print(f"Attempting to load: {path}")
                return ctypes.CDLL(path)
            except Exception as e:
                print(f"Failed to load {path}: {e}")
                
    raise RuntimeError(f"Could not find POPGP Kernel library ({lib_name}). Build the project first.")

_lib = _load_library()

# Define C Structures (Complex)
class cuFloatComplex(ctypes.Structure):
    _fields_ = [("x", c_float), ("y", c_float)]

class cuDoubleComplex(ctypes.Structure):
    _fields_ = [("x", c_double), ("y", c_double)]

# Define Function Signatures
try:
    _lib.launch_phase_flow_float.argtypes = [
        c_void_p, c_void_p,
        c_void_p, c_void_p, c_void_p,
        c_int, c_float
    ]

    _lib.launch_phase_flow_double.argtypes = [
        c_void_p, c_void_p,
        c_void_p, c_void_p, c_void_p,
        c_int, c_double
    ]
except Exception as e:
    print(f"Warning: Could not bind function signatures: {e}")

class Engine:
    """
    Python Interface to the POPGP C++/CUDA Kernel.
    """
    def __init__(self, num_cells, precision="double"):
        self.num_cells = num_cells
        self.precision = precision

    def step(self, d_alphas, d_betas, d_src, d_dst, d_weights, dt):
        """
        Run one step of Phase-Ordered Flow.
        Inputs: PyTorch or CuPy tensors (must be on GPU).
        """
        ptr_alphas = self._get_ptr(d_alphas)
        ptr_betas  = self._get_ptr(d_betas)
        ptr_src    = self._get_ptr(d_src)
        ptr_dst    = self._get_ptr(d_dst)
        ptr_w      = self._get_ptr(d_weights)
        
        num_edges = d_src.numel() if hasattr(d_src, "numel") else d_src.size
        
        if self.precision == "float":
            _lib.launch_phase_flow_float(
                c_void_p(ptr_alphas), c_void_p(ptr_betas), 
                c_void_p(ptr_src), c_void_p(ptr_dst), c_void_p(ptr_w),
                num_edges, float(dt)
            )
        else:
            _lib.launch_phase_flow_double(
                c_void_p(ptr_alphas), c_void_p(ptr_betas), 
                c_void_p(ptr_src), c_void_p(ptr_dst), c_void_p(ptr_w),
                num_edges, float(dt)
            )

    def _get_ptr(self, tensor):
        if hasattr(tensor, "data_ptr"):
            return tensor.data_ptr()
        if hasattr(tensor, "data"):
            return tensor.data.ptr
        raise ValueError("Unknown tensor type. Use PyTorch or CuPy.")

