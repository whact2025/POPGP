"""
Python bindings for the POPGP C++/CUDA phase-flow kernel.

Loads the native shared library (phase_flow.dll/.so/.dylib) and exposes
it via the Engine class. See docs/framework.md Section 4.4.1 for the
underlying phase-flow mechanism.
"""

import ctypes
import logging
import os
import sys
from ctypes import c_double, c_float, c_int, c_void_p
from pathlib import Path

log = logging.getLogger(__name__)

_LIB_DIR = Path(__file__).parent / "_lib"

_ENGINE_ROOT = Path(__file__).resolve().parent.parent / "popgp_engine"


def _lib_filename() -> str:
    if os.name == "nt":
        return "phase_flow.dll"
    elif sys.platform == "darwin":
        return "libphase_flow.dylib"
    return "libphase_flow.so"


def _add_dll_directories() -> None:
    """Register directories that contain transitive DLL dependencies (Windows)."""
    if os.name != "nt":
        return

    dirs_to_add = [
        _LIB_DIR,
    ]

    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        dirs_to_add.append(Path(cuda_path) / "bin")

    vcpkg_bin = _ENGINE_ROOT / "build" / "vcpkg_installed" / "x64-windows" / "bin"
    dirs_to_add.append(vcpkg_bin)

    for d in dirs_to_add:
        if d.is_dir():
            try:
                os.add_dll_directory(str(d))
                log.debug("DLL search path: %s", d)
            except OSError:
                log.debug("Could not add DLL directory: %s", d)


def _load_library() -> ctypes.CDLL:
    _add_dll_directories()

    lib_name = _lib_filename()
    search_dirs = [
        _LIB_DIR,
        _ENGINE_ROOT / "build" / "kernel" / "Release",
        _ENGINE_ROOT / "build" / "kernel" / "Debug",
        _ENGINE_ROOT / "build" / "kernel",
    ]

    # winmode=0 is required on Windows Python 3.8+ so that directories
    # registered via os.add_dll_directory() are actually searched when
    # resolving transitive DLL dependencies (CUDA runtime, fmt, etc.).
    load_kwargs = {"winmode": 0} if os.name == "nt" else {}

    errors: list[str] = []
    for d in search_dirs:
        path = d / lib_name
        if path.exists():
            try:
                lib = ctypes.CDLL(str(path), **load_kwargs)
                log.info("Loaded kernel from %s", path)
                return lib
            except OSError as exc:
                errors.append(f"  {path}: {exc}")

    msg = (
        f"Could not find or load POPGP kernel library ({lib_name}).\n"
        "Build the C++/CUDA engine first (popgp_engine/build.bat --test).\n"
    )
    if errors:
        msg += "Attempted paths:\n" + "\n".join(errors)
    raise RuntimeError(msg)


_lib = _load_library()

_lib.launch_phase_flow_float.argtypes = [
    c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p,
    c_int, c_float,
]

_lib.launch_phase_flow_double.argtypes = [
    c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p,
    c_int, c_double,
]


class Engine:
    """
    Python interface to the POPGP C++/CUDA kernel.

    Wraps the phase-flow shared library via ctypes.  All tensor arguments
    passed to :meth:`step` must reside on the GPU (PyTorch CUDA tensors
    or CuPy arrays).
    """

    def __init__(self, precision: str = "double"):
        self.precision = precision

    def step(self, d_alphas, d_betas, d_src, d_dst, d_weights, dt: float):
        """Run one phase-order step of the kernel (Section 4.4.1)."""
        ptr_a = self._get_ptr(d_alphas)
        ptr_b = self._get_ptr(d_betas)
        ptr_s = self._get_ptr(d_src)
        ptr_d = self._get_ptr(d_dst)
        ptr_w = self._get_ptr(d_weights)

        num_edges = d_src.numel() if hasattr(d_src, "numel") else d_src.size

        if self.precision == "float":
            _lib.launch_phase_flow_float(
                c_void_p(ptr_a), c_void_p(ptr_b),
                c_void_p(ptr_s), c_void_p(ptr_d), c_void_p(ptr_w),
                c_int(num_edges), c_float(dt),
            )
        else:
            _lib.launch_phase_flow_double(
                c_void_p(ptr_a), c_void_p(ptr_b),
                c_void_p(ptr_s), c_void_p(ptr_d), c_void_p(ptr_w),
                c_int(num_edges), c_double(dt),
            )

    @staticmethod
    def _get_ptr(tensor) -> int:
        if hasattr(tensor, "data_ptr"):
            return tensor.data_ptr()
        if hasattr(tensor, "data") and hasattr(tensor.data, "ptr"):
            return tensor.data.ptr
        raise TypeError("Unknown tensor type. Use PyTorch or CuPy.")
