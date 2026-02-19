import ctypes
import os
import sys
import numpy as np
from ctypes import POINTER, c_int, c_float, c_double, c_void_p

# Load the Shared Library
def _load_library():
    # Search paths: 
    # 1. Local directory (if copied by CMake)
    # 2. build/bin/ (Standard CMake output)
    # 3. System paths
    
    lib_name = "phase_flow"
    if os.name == "nt":
        lib_name += ".dll"
    elif sys.platform == "darwin":
        lib_name = "lib" + lib_name + ".dylib"
    else:
        lib_name = "lib" + lib_name + ".so"
        
    search_dirs = [
        os.path.dirname(os.path.abspath(__file__)),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../build/bin"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../build/popgp_engine/kernel"),
    ]
    
    for d in search_dirs:
        path = os.path.join(d, lib_name)
        if os.path.exists(path):
            try:
                return ctypes.CDLL(path)
            except Exception as e:
                print(f"Found lib at {path} but failed to load: {e}")
                
    raise RuntimeError(f"Could not find POPGP Kernel library ({lib_name}). Build the project first.")

_lib = _load_library()

# Define C Structures (Complex)
class cuFloatComplex(ctypes.Structure):
    _fields_ = [("x", c_float), ("y", c_float)]

class cuDoubleComplex(ctypes.Structure):
    _fields_ = [("x", c_double), ("y", c_double)]

# Define Function Signatures
_lib.launch_phase_flow_float.argtypes = [
    POINTER(cuFloatComplex), POINTER(cuFloatComplex),
    POINTER(c_int), POINTER(c_int), POINTER(c_float),
    c_int, c_float
]

_lib.launch_phase_flow_double.argtypes = [
    POINTER(cuDoubleComplex), POINTER(cuDoubleComplex),
    POINTER(c_int), POINTER(c_int), POINTER(c_double),
    c_int, c_double
]

class Engine:
    """
    Python Interface to the POPGP C++/CUDA Kernel.
    Manages GPU memory and kernel launches via ctypes.
    """
    def __init__(self, num_cells, precision="double"):
        self.num_cells = num_cells
        self.precision = precision
        self.edges = None
        
        # We rely on an external library (like PyCuda or CuPy) for GPU memory management
        # OR we can just pass host pointers and let the C++ side handle upload/download (slower).
        #
        # For high performance, this binding expects *pointers to GPU memory*.
        # Assuming the user uses CuPy to manage data.
        
        # Check if cupy is available
        try:
            import cupy as cp
            self.cp = cp
        except ImportError:
            raise ImportError("CuPy is required to manage GPU memory for the POPGP Engine.")

    def step(self, d_alphas, d_betas, d_src, d_dst, d_weights, dt):
        """
        Run one step of Phase-Ordered Flow.
        All inputs must be CuPy arrays (on device).
        """
        num_edges = d_src.size
        
        # Get raw pointers
        ptr_alphas = d_alphas.data.ptr
        ptr_betas  = d_betas.data.ptr
        ptr_src    = d_src.data.ptr
        ptr_dst    = d_dst.data.ptr
        ptr_w      = d_weights.data.ptr
        
        if self.precision == "float":
            _lib.launch_phase_flow_float(
                ctypes.cast(ptr_alphas, POINTER(cuFloatComplex)),
                ctypes.cast(ptr_betas, POINTER(cuFloatComplex)),
                ctypes.cast(ptr_src, POINTER(c_int)),
                ctypes.cast(ptr_dst, POINTER(c_int)),
                ctypes.cast(ptr_w, POINTER(c_float)),
                num_edges, dt
            )
        else:
            _lib.launch_phase_flow_double(
                ctypes.cast(ptr_alphas, POINTER(cuDoubleComplex)),
                ctypes.cast(ptr_betas, POINTER(cuDoubleComplex)),
                ctypes.cast(ptr_src, POINTER(c_int)),
                ctypes.cast(ptr_dst, POINTER(c_int)),
                ctypes.cast(ptr_w, POINTER(c_double)),
                num_edges, dt
            )

    def prune(self, d_active_mask, d_cut_sizes):
        """ Run Area Law Pruning """
        # TODO: Expose prune kernel
        pass
