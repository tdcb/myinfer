"""SSD expert streaming for MoE models larger than unified memory.

Everything except the routed experts (embeddings, attention, norms, routers,
shared experts) is loaded normally. Routed expert weights stay on disk and are
read on demand into a bounded LRU cache:

* MLX-format checkpoints already store each MoE layer's experts stacked as
  ``[num_experts, ...]`` tensors, so one expert is a contiguous byte range per
  tensor and can be read with a single ``pread`` (no repacking needed).
* Each MoE layer evaluates its router, reads back the chosen expert ids, loads
  any missing experts in parallel, and runs the stock ``gather_qmm`` path on a
  compact stack of just those experts.
* During decode the next layer's router is applied to the current hidden state
  to guess its experts and start reading them early.
* On macOS reads use ``F_NOCACHE`` so expert bytes are not cached twice (page
  cache + expert cache).
"""

from __future__ import annotations

import fcntl
import json
import os
import platform
import struct
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm.models.switch_layers import SwitchGLU, SwitchMLP

F_NOCACHE = 48

_NP_DTYPES = {
    "U8": (np.uint8, None),
    "I8": (np.int8, None),
    "U16": (np.uint16, None),
    "I16": (np.int16, None),
    "U32": (np.uint32, None),
    "I32": (np.int32, None),
    "F16": (np.float16, None),
    "F32": (np.float32, None),
    "BF16": (np.uint16, mx.bfloat16),
}


@dataclass
class TensorLocation:
    file: str
    dtype: str
    shape: Tuple[int, ...]
    start: int
    end: int



@dataclass
class ExpertTensor:
    """Where one expert parameter lives: a stacked [E, ...] tensor or one tensor per expert."""

    dtype: str
    shape: Tuple[int, ...]  # per-expert shape
    stacked: Optional[TensorLocation] = None
    per_expert: Optional[List[TensorLocation]] = None

    @property
    def per_expert_bytes(self) -> int:
        if self.stacked is not None:
            return (self.stacked.end - self.stacked.start) // self.stacked.shape[0]
        loc = self.per_expert[0]
        return loc.end - loc.start

    @property
    def total_bytes(self) -> int:
        if self.stacked is not None:
            return self.stacked.end - self.stacked.start
        return sum(l.end - l.start for l in self.per_expert)

    def byte_range(self, expert: int) -> Tuple[str, int, int]:
        if self.stacked is not None:
            n = self.per_expert_bytes
            return self.stacked.file, self.stacked.start + expert * n, n
        loc = self.per_expert[expert]
        return loc.file, loc.start, loc.end - loc.start


def locate_expert_tensor(
    index: Dict[str, TensorLocation], path: str, rel: str, shape: Tuple[int, ...]
) -> ExpertTensor:
    key = f"{path}.{rel}"
    loc = index.get(key)
    if loc is not None:
        if loc.shape != shape:
            raise ValueError(
                f"Expert tensor '{key}' is {loc.shape} on disk but the model expects {shape}; "
                "the checkpoint does not match its config."
            )
        return ExpertTensor(loc.dtype, shape[1:], stacked=loc)
    parent = path.rsplit(".", 1)[0]
    per = []
    for e in range(shape[0]):
        k = f"{parent}.experts.{e}.{rel}"
        l = index.get(k)
        if l is None or l.shape != shape[1:]:
            raise ValueError(
                f"Expert tensor '{key}' (or per-expert '{parent}.experts.N.{rel}') was not found "
                "with the expected shape. Convert the model to MLX format first "
                "(python -m mlx_lm convert ...)."
            )
        per.append(l)
    return ExpertTensor(per[0].dtype, shape[1:], per_expert=per)


def read_safetensors_index(model_path: Path) -> Dict[str, TensorLocation]:
    index: Dict[str, TensorLocation] = {}
    for f in sorted(Path(model_path).glob("model*.safetensors")):
        with open(f, "rb") as fh:
            (n,) = struct.unpack("<Q", fh.read(8))
            header = json.loads(fh.read(n))
        base = 8 + n
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            s, e = meta["data_offsets"]
            index[key] = TensorLocation(str(f), meta["dtype"], tuple(meta["shape"]), base + s, base + e)
    return index


class ExpertReader:
    """Reads one expert's slice of a stacked tensor from disk."""

    def __init__(self):
        self._fds: Dict[str, int] = {}
        self._lock = threading.Lock()

    def _fd(self, path: str) -> int:
        with self._lock:
            fd = self._fds.get(path)
            if fd is None:
                fd = os.open(path, os.O_RDONLY)
                if platform.system() == "Darwin":
                    try:
                        fcntl.fcntl(fd, F_NOCACHE, 1)
                    except OSError:
                        pass
                self._fds[path] = fd
            return fd

    def read(self, loc: ExpertTensor, expert: int) -> np.ndarray:
        np_dtype, _ = _NP_DTYPES[loc.dtype]
        path, offset, nbytes = loc.byte_range(expert)
        buf = np.empty(nbytes, dtype=np.uint8)
        fd = self._fd(path)
        view = memoryview(buf)
        got = 0
        while got < nbytes:
            n = os.preadv(fd, [view[got:]], offset + got) if hasattr(os, "preadv") else None
            if n is None:
                chunk = os.pread(fd, nbytes - got, offset + got)
                view[got : got + len(chunk)] = chunk
                n = len(chunk)
            if n <= 0:
                raise IOError(f"Short read from {path} at {offset + got}")
            got += n
        return buf.view(np_dtype).reshape(loc.shape)

    def close(self):
        with self._lock:
            for fd in self._fds.values():
                os.close(fd)
            self._fds.clear()


def to_mx(arr: np.ndarray, st_dtype: str) -> mx.array:
    _, view_as = _NP_DTYPES[st_dtype]
    a = mx.array(arr)
    return a.view(view_as) if view_as is not None else a


@dataclass
class StreamStats:
    hits: int = 0
    misses: int = 0
    prefetch_issued: int = 0
    prefetch_used: int = 0
    bytes_read: int = 0
    read_seconds: float = 0.0
    evictions: int = 0

    def to_dict(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
            "prefetch_issued": self.prefetch_issued,
            "prefetch_used": self.prefetch_used,
            "gb_read": round(self.bytes_read / 1024**3, 3),
            "read_gb_per_s": round(self.bytes_read / 1024**3 / self.read_seconds, 2)
            if self.read_seconds
            else 0.0,
            "evictions": self.evictions,
        }


class ExpertCache:
    """Byte-bounded LRU of experts keyed by (layer, expert)."""

    def __init__(self, capacity_bytes: int, io_threads: int = 8):
        self.capacity_bytes = int(capacity_bytes)
        self.reader = ExpertReader()
        self.pool = ThreadPoolExecutor(max_workers=io_threads, thread_name_prefix="m5-expert-io")
        self._entries: "OrderedDict[Tuple[int, int], Tuple[Dict[str, mx.array], int]]" = OrderedDict()
        self._inflight: Dict[Tuple[int, int], Future] = {}
        self._prefetched: set = set()
        self._bytes = 0
        self.stats = StreamStats()
        self._lock = threading.Lock()

    @property
    def used_bytes(self) -> int:
        return self._bytes

    def _read_expert(self, locs: Dict[str, ExpertTensor], expert: int):
        t0 = time.perf_counter()
        out = {rel: self.reader.read(loc, expert) for rel, loc in locs.items()}
        nbytes = sum(a.nbytes for a in out.values())
        with self._lock:
            self.stats.read_seconds += time.perf_counter() - t0
            self.stats.bytes_read += nbytes
        return out

    def _submit(self, key, locs):
        fut = self._inflight.get(key)
        if fut is None:
            fut = self.pool.submit(self._read_expert, locs, key[1])
            self._inflight[key] = fut
        return fut

    def prefetch(self, layer: int, experts: Sequence[int], locs: Dict[str, ExpertTensor]) -> None:
        for e in experts:
            key = (layer, int(e))
            if key in self._entries or key in self._inflight:
                continue
            self._submit(key, locs)
            self._prefetched.add(key)
            self.stats.prefetch_issued += 1

    def _insert(self, key, arrays: Dict[str, mx.array], protect: set) -> None:
        nbytes = sum(a.nbytes for a in arrays.values())
        self._entries[key] = (arrays, nbytes)
        self._bytes += nbytes
        while self._bytes > self.capacity_bytes and self._entries:
            old_key = next(iter(self._entries))
            if old_key in protect:
                if len(protect) >= len(self._entries):
                    break
                self._entries.move_to_end(old_key)
                continue
            _, old_bytes = self._entries.pop(old_key)
            self._bytes -= old_bytes
            self.stats.evictions += 1

    def fetch(
        self,
        layer: int,
        experts: Sequence[int],
        locs: Dict[str, ExpertTensor],
    ) -> List[Dict[str, mx.array]]:
        keys = [(layer, int(e)) for e in experts]
        protect = set(keys)
        found: Dict[Tuple[int, int], Dict[str, mx.array]] = {}
        for key in keys:
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
                found[key] = entry[0]
                self.stats.hits += 1
            else:
                self._submit(key, locs)
        for key in keys:
            if key in found:
                continue
            raw = self._inflight.pop(key).result()
            if key in self._prefetched:
                self._prefetched.discard(key)
                self.stats.prefetch_used += 1
            else:
                self.stats.misses += 1
            arrays = {rel: to_mx(a, locs[rel].dtype) for rel, a in raw.items()}
            self._insert(key, arrays, protect)
            found[key] = arrays
        self._settle_mispredicted(layer, protect, locs)
        return [found[k] for k in keys]

    def _settle_mispredicted(self, layer: int, protect: set, locs: Dict[str, ExpertTensor]) -> None:
        """Move unused prefetches for this layer into the LRU so they stay bounded."""
        stale = [k for k in self._prefetched if k[0] == layer and k not in protect]
        for key in stale:
            self._prefetched.discard(key)
            fut = self._inflight.pop(key, None)
            if fut is None:
                continue
            raw = fut.result()
            self._insert(key, {rel: to_mx(a, locs[rel].dtype) for rel, a in raw.items()}, protect)

    def clear(self) -> None:
        self._entries.clear()
        self._bytes = 0

    def close(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)
        self.reader.close()
        self.clear()


def _get_by_path(root: nn.Module, path: str):
    obj = root
    for part in path.split("."):
        obj = obj[int(part)] if isinstance(obj, list) else obj[part]
    return obj


def _set_attr_path(root: nn.Module, rel: str, value) -> None:
    *parents, leaf = rel.split(".")
    obj = root
    for p in parents:
        obj = obj[int(p)] if isinstance(obj, list) else obj[p]
    setattr(obj, leaf, value)


def _router_top_k(parent: nn.Module, default: int) -> int:
    for name in ("num_experts_per_tok", "top_k", "k", "n_activated_experts"):
        v = getattr(parent, name, None)
        if isinstance(v, int) and v > 0:
            return v
    return default


class StreamingExperts(nn.Module):
    """Drop-in replacement for SwitchGLU/SwitchMLP that pulls experts from disk."""

    def __init__(
        self,
        inner: nn.Module,
        layer_id: int,
        locs: Dict[str, ExpertTensor],
        cache: ExpertCache,
        num_experts: int,
    ):
        super().__init__()
        self.inner = inner
        # Plain attributes (not module children/parameters).
        _plain = object.__setattr__
        _plain(self, "_layer_id", layer_id)
        _plain(self, "_locs", locs)
        _plain(self, "_cache", cache)
        _plain(self, "_num_experts", num_experts)
        _plain(self, "_placeholders", {rel: v for rel, v in tree_flatten(inner.parameters())})
        _plain(self, "_next", None)  # (gate_module, StreamingExperts, top_k)

    def set_next(self, gate: nn.Module, nxt: "StreamingExperts", top_k: int) -> None:
        object.__setattr__(self, "_next", (gate, nxt, top_k))

    def _predict_next(self, x: mx.array):
        gate, nxt, top_k = self._next
        out = gate(x.reshape(-1, x.shape[-1]))
        if isinstance(out, tuple):
            return nxt, out[0]
        k = min(top_k, out.shape[-1])
        return nxt, mx.argpartition(-out, kth=k - 1, axis=-1)[..., :k]

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        pred = None
        n_tokens = indices.size // max(indices.shape[-1], 1)
        if self._next is not None and n_tokens <= 4:
            nxt, pred = self._predict_next(x)
            mx.eval(indices, pred)
        else:
            mx.eval(indices)
        inds = np.array(indices)
        uniq = np.unique(inds)
        if pred is not None:
            nxt._cache.prefetch(nxt._layer_id, np.unique(np.array(pred)).tolist(), nxt._locs)

        experts = self._cache.fetch(self._layer_id, uniq.tolist(), self._locs)
        for rel in self._placeholders:
            _set_attr_path(self.inner, rel, mx.stack([e[rel] for e in experts]))
        remap = np.zeros(self._num_experts, dtype=np.uint32)
        remap[uniq] = np.arange(len(uniq), dtype=np.uint32)
        try:
            return self.inner(x, mx.array(remap[inds]))
        finally:
            for rel, ph in self._placeholders.items():
                _set_attr_path(self.inner, rel, ph)


@dataclass
class StreamingInfo:
    layers: int
    num_experts: int
    expert_bytes: int
    total_expert_bytes: int
    cache: ExpertCache

    def to_dict(self) -> dict:
        return {
            "moe_layers": self.layers,
            "num_experts": self.num_experts,
            "expert_mb": round(self.expert_bytes / 1024**2, 2),
            "total_expert_gb": round(self.total_expert_bytes / 1024**3, 2),
            "cache_capacity_gb": round(self.cache.capacity_bytes / 1024**3, 2),
            "cache_used_gb": round(self.cache.used_bytes / 1024**3, 2),
            "cache_slots": int(self.cache.capacity_bytes // max(self.expert_bytes, 1)),
            **self.cache.stats.to_dict(),
        }


def _path_order(path: str):
    return [(0, int(p), "") if p.isdigit() else (1, 0, p) for p in path.split(".")]


def find_expert_modules(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Routed-expert modules in forward (layer) order."""
    found = [
        (path, m)
        for path, m in model.named_modules()
        if isinstance(m, (SwitchGLU, SwitchMLP))
    ]
    return sorted(found, key=lambda pm: _path_order(pm[0]))


def estimate_expert_bytes(model_path: Path, model: nn.Module) -> int:
    index = read_safetensors_index(model_path)
    total = 0
    for path, inner in find_expert_modules(model):
        for rel, arr in tree_flatten(inner.parameters()):
            total += locate_expert_tensor(index, path, rel, tuple(arr.shape)).total_bytes
    return total


def enable_expert_streaming(
    model: nn.Module,
    model_path: Path,
    cache_bytes: int,
    io_threads: int = 8,
    prefetch: bool = True,
) -> StreamingInfo:
    """Replace routed experts with disk-backed streaming modules.

    Must be called on a lazily loaded model *before* its parameters are
    evaluated, so the full expert tensors are never read into memory.
    """
    index = read_safetensors_index(model_path)
    modules = find_expert_modules(model)
    if not modules:
        raise ValueError("Expert streaming requested but the model has no routed-expert layers.")
    cache = ExpertCache(cache_bytes, io_threads=io_threads)
    wrappers: List[Tuple[nn.Module, StreamingExperts]] = []
    expert_bytes = 0
    total = 0
    num_experts = 0
    for layer_id, (path, inner) in enumerate(modules):
        locs: Dict[str, ExpertTensor] = {}
        params = tree_flatten(inner.parameters())
        for rel, arr in params:
            locs[rel] = locate_expert_tensor(index, path, rel, tuple(arr.shape))
            if locs[rel].dtype not in _NP_DTYPES:
                raise ValueError(f"Unsupported expert dtype {locs[rel].dtype} for {path}.{rel}")
        n_exp = params[0][1].shape[0]
        num_experts = max(num_experts, n_exp)
        expert_bytes = max(expert_bytes, sum(l.per_expert_bytes for l in locs.values()))
        total += sum(l.total_bytes for l in locs.values())
        for rel, arr in params:
            _set_attr_path(inner, rel, mx.zeros((1, *arr.shape[1:]), dtype=arr.dtype))
        parent_path, attr = path.rsplit(".", 1)
        parent = _get_by_path(model, parent_path)
        wrapper = StreamingExperts(inner, layer_id, locs, cache, n_exp)
        setattr(parent, attr, wrapper)
        wrappers.append((parent, wrapper))

    if prefetch:
        for (_, w), (nparent, nw) in zip(wrappers, wrappers[1:]):
            gate = nparent.get("gate") if isinstance(nparent, nn.Module) else None
            if gate is not None and callable(gate):
                w.set_next(gate, nw, _router_top_k(nparent, 8))

    return StreamingInfo(len(wrappers), num_experts, expert_bytes, total, cache)
