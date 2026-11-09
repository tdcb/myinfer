"""Host memory, GPU working-set and thermal state discovery."""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import platform
import subprocess
from dataclasses import asdict, dataclass
from typing import Optional

import mlx.core as mx

GB = 1024**3
THERMAL_STATES = ["nominal", "fair", "serious", "critical"]


def _sysctl(name: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ["sysctl", "-n", name], capture_output=True, text=True, timeout=2
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def total_ram_bytes() -> int:
    if platform.system() == "Darwin":
        v = _sysctl("hw.memsize")
        if v and v.isdigit():
            return int(v)
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")


def iogpu_wired_limit_bytes() -> int:
    """User override set with `sudo sysctl iogpu.wired_limit_mb=N` (0 = macOS default)."""
    v = _sysctl("iogpu.wired_limit_mb")
    if v and v.isdigit():
        return int(v) * 1024 * 1024
    return 0


def chip_name() -> str:
    if platform.system() == "Darwin":
        v = _sysctl("machdep.cpu.brand_string")
        if v:
            return v
    info = mx.device_info()
    return str(info.get("device_name") or platform.processor() or "unknown")


_thermal_fn = None


def thermal_state() -> str:
    """NSProcessInfo.thermalState via the ObjC runtime; 'unknown' off macOS."""
    global _thermal_fn
    if platform.system() != "Darwin":
        return "unknown"
    try:
        if _thermal_fn is None:
            objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
            ctypes.cdll.LoadLibrary(ctypes.util.find_library("Foundation"))
            objc.objc_getClass.restype = ctypes.c_void_p
            objc.objc_getClass.argtypes = [ctypes.c_char_p]
            objc.sel_registerName.restype = ctypes.c_void_p
            objc.sel_registerName.argtypes = [ctypes.c_char_p]
            send_ptr = ctypes.cast(objc.objc_msgSend, ctypes.c_void_p).value
            send_obj = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(send_ptr)
            send_long = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p)(send_ptr)
            info = send_obj(
                objc.objc_getClass(b"NSProcessInfo"), objc.sel_registerName(b"processInfo")
            )
            sel = objc.sel_registerName(b"thermalState")
            _thermal_fn = lambda: send_long(info, sel)  # noqa: E731
        idx = int(_thermal_fn())
        return THERMAL_STATES[idx] if 0 <= idx < len(THERMAL_STATES) else "unknown"
    except Exception:
        return "unknown"


@dataclass
class HardwareProfile:
    chip: str
    gpu: str
    metal: bool
    total_ram_bytes: int
    recommended_working_set_bytes: int
    iogpu_wired_limit_bytes: int

    @property
    def gpu_limit_bytes(self) -> int:
        """Largest amount the GPU can keep wired without paging."""
        if self.iogpu_wired_limit_bytes:
            return min(self.iogpu_wired_limit_bytes, self.total_ram_bytes)
        return self.recommended_working_set_bytes

    def to_dict(self) -> dict:
        d = asdict(self)
        d["gpu_limit_bytes"] = self.gpu_limit_bytes
        return d


def detect() -> HardwareProfile:
    info = mx.device_info()
    ram = total_ram_bytes()
    metal = mx.metal.is_available()
    rec = int(info.get("max_recommended_working_set_size") or 0)
    if not rec:
        rec = int(ram * 0.75)
    return HardwareProfile(
        chip=chip_name(),
        gpu=str(info.get("device_name") or ("Apple GPU" if metal else "CPU")),
        metal=metal,
        total_ram_bytes=ram,
        recommended_working_set_bytes=rec,
        iogpu_wired_limit_bytes=iogpu_wired_limit_bytes() if metal else 0,
    )


def memory_snapshot() -> dict:
    return {
        "active_bytes": mx.get_active_memory(),
        "cache_bytes": mx.get_cache_memory(),
        "peak_bytes": mx.get_peak_memory(),
    }
