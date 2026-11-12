"""SSD expert streaming for MoE models larger than unified memory.

Everything except the routed experts (embeddings, attention, norms, routers,
shared experts) is loaded normally. Routed expert weights stay on disk and are
read on demand into a bounded expert cache:

* MLX-format checkpoints already store each MoE layer's experts stacked as
  ``[num_experts, ...]`` tensors, so one expert is a contiguous byte range per
  tensor and can be read with a single ``pread`` per tensor. ``repack_experts``
  optionally writes a sidecar where every expert's tensors sit together, so an
  expert is one read instead of one per tensor (quantized experts have 9).
* Each MoE layer evaluates its router, reads back the chosen expert ids, loads
  any missing experts in parallel, and runs the stock ``gather_qmm`` path on a
  compact stack of just those experts.
* During decode the next layer's router is applied to the current hidden state
  to guess its experts and start reading them early.
* An ``ExpertProfile`` keeps a decayed count of how often each expert is routed
  to. The cache evicts the coldest expert instead of the least recent one, and
  at load the hottest experts from the saved profile are read in the background
  so a fresh process starts warm (Strata's hot-expert profile, on unified memory).
* On macOS reads use ``F_NOCACHE`` so expert bytes are not cached twice (page
  cache + expert cache).
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import platform
import struct
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm.models.switch_layers import SwitchGLU, SwitchMLP

log = logging.getLogger(__name__)

F_NOCACHE = 48
SIDECAR_DIR = ".m5engine"
PROFILE_FILE = "expert-profile.npz"
PACK_FILE = "experts.pack"
_PACK_MAGIC = b"M5XP0001"

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


LayerLocs = Dict[str, ExpertTensor]


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


def _pread_into(fd: int, buf: np.ndarray, offset: int, path: str) -> None:
    nbytes = buf.nbytes
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


class _Files:
    """Read-only descriptors, opened once, with the page cache bypassed on macOS."""

    def __init__(self):
        self._fds: Dict[str, int] = {}
        self._lock = threading.Lock()

    def fd(self, path: str) -> int:
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

    def close(self):
        with self._lock:
            for fd in self._fds.values():
                os.close(fd)
            self._fds.clear()


class ExpertReader:
    """Reads one expert's slice of a stacked tensor from disk."""

    def __init__(self):
        self.files = _Files()

    def read(self, loc: ExpertTensor, expert: int) -> np.ndarray:
        np_dtype, _ = _NP_DTYPES[loc.dtype]
        path, offset, nbytes = loc.byte_range(expert)
        buf = np.empty(nbytes, dtype=np.uint8)
        _pread_into(self.files.fd(path), buf, offset, path)
        return buf.view(np_dtype).reshape(loc.shape)

    def read_expert(self, locs: LayerLocs, expert: int) -> Dict[str, np.ndarray]:
        return {rel: self.read(loc, expert) for rel, loc in locs.items()}

    def close(self):
        self.files.close()


def to_mx(arr: np.ndarray, st_dtype: str) -> mx.array:
    _, view_as = _NP_DTYPES[st_dtype]
    a = mx.array(arr)
    return a.view(view_as) if view_as is not None else a


# ------------------------------------------------------------------ pack


def _locs_fingerprint(layer_locs: Sequence[LayerLocs]) -> str:
    desc = [
        [
            (rel, Path(t.byte_range(0)[0]).name, t.dtype, list(t.shape), t.byte_range(0)[1], t.total_bytes)
            for rel, t in sorted(locs.items())
        ]
        for locs in layer_locs
    ]
    return hashlib.sha1(json.dumps(desc).encode()).hexdigest()


class ExpertPack:
    """Sidecar file with every expert's tensors contiguous: one read per expert.

    Layout: magic, u64 header length, JSON header, then for each layer, for each
    expert, its tensors in ``rels`` order. Built once by ``repack_experts``; the
    header records a fingerprint of the source tensors so a re-downloaded or
    re-quantized checkpoint invalidates it.
    """

    def __init__(self, path: Path, header: dict, data_start: int):
        self.path = str(path)
        self.header = header
        self.data_start = data_start  # layer offsets in the header are relative to this
        self.files = _Files()
        self._layers = header["layers"]

    @staticmethod
    def sidecar(model_path: Path) -> Path:
        return Path(model_path) / SIDECAR_DIR / PACK_FILE

    @classmethod
    def open(cls, model_path: Path, layer_locs: Sequence[LayerLocs]) -> Optional["ExpertPack"]:
        p = cls.sidecar(model_path)
        if not p.exists():
            return None
        try:
            with open(p, "rb") as fh:
                if fh.read(len(_PACK_MAGIC)) != _PACK_MAGIC:
                    return None
                (n,) = struct.unpack("<Q", fh.read(8))
                header = json.loads(fh.read(n))
        except (OSError, ValueError):
            return None
        if header.get("fingerprint") != _locs_fingerprint(layer_locs) or not header.get("complete"):
            log.warning("Expert pack %s does not match the checkpoint; ignoring it", p)
            return None
        return cls(p, header, len(_PACK_MAGIC) + 8 + n)

    def read(self, layer: int, expert: int) -> Dict[str, np.ndarray]:
        L = self._layers[layer]
        stride = L["stride"]
        buf = np.empty(stride, dtype=np.uint8)
        _pread_into(self.files.fd(self.path), buf, self.data_start + L["offset"] + expert * stride, self.path)
        out = {}
        pos = 0
        for rel, dtype, shape, size in zip(L["rels"], L["dtypes"], L["shapes"], L["sizes"]):
            np_dtype, _ = _NP_DTYPES[dtype]
            out[rel] = buf[pos : pos + size].view(np_dtype).reshape(shape)
            pos += size
        return out

    def close(self):
        self.files.close()


def write_expert_pack(
    model_path: Path,
    layer_locs: Sequence[LayerLocs],
    num_experts: Sequence[int],
    progress: Optional[Callable[[int, int], None]] = None,
) -> Path:
    """Write the contiguous expert sidecar; returns its path."""
    out = ExpertPack.sidecar(model_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    layers = []
    offset = 0
    for locs, n in zip(layer_locs, num_experts):
        rels = sorted(locs)
        sizes = [locs[r].per_expert_bytes for r in rels]
        stride = sum(sizes)
        layers.append({
            "rels": rels, "sizes": sizes, "dtypes": [locs[r].dtype for r in rels],
            "shapes": [list(locs[r].shape) for r in rels], "stride": stride, "experts": n, "offset": offset,
        })
        offset += stride * n
    header = {"layers": layers, "fingerprint": _locs_fingerprint(layer_locs), "complete": False, "bytes": offset}
    hdr = json.dumps(header).encode()
    reader = ExpertReader()
    tmp = out.with_suffix(".tmp")
    total = offset
    done = 0
    try:
        with open(tmp, "wb") as fh:
            fh.write(_PACK_MAGIC + struct.pack("<Q", len(hdr)) + hdr)
            for locs, L in zip(layer_locs, layers):
                for e in range(L["experts"]):
                    for rel in L["rels"]:
                        fh.write(reader.read(locs[rel], e).tobytes())
                    done += L["stride"]
                    if progress is not None:
                        progress(done, total)
            # Same byte length as the header written above ("false" -> "true ").
            fh.seek(len(_PACK_MAGIC) + 8)
            fh.write(hdr.replace(b'"complete": false', b'"complete": true '))
        os.replace(tmp, out)
    finally:
        reader.close()
        if tmp.exists():
            tmp.unlink()
    return out


# --------------------------------------------------------------- profile


class ExpertProfile:
    """Decayed routing frequency per (layer, expert).

    Each MoE layer call adds its routing counts normalised to one token, and the
    first layer's call decays everything first, so one step of decode and one
    chunk of prefill weigh the same. Scores order evictions and the warm fill.
    """

    def __init__(self, layers: int, num_experts: int, decay: float = 0.998):
        self.scores = np.zeros((layers, num_experts), dtype=np.float64)
        self.decay = decay
        self.steps = 0
        self.dirty = False
        self.paused = False  # e.g. while the engine measures the KV footprint with dummy tokens
        self._lock = threading.Lock()

    @staticmethod
    def sidecar(model_path: Path) -> Path:
        return Path(model_path) / SIDECAR_DIR / PROFILE_FILE

    @classmethod
    def load(cls, model_path: Path, layers: int, num_experts: int) -> "ExpertProfile":
        prof = cls(layers, num_experts)
        p = cls.sidecar(model_path)
        if p.exists():
            try:
                with np.load(p) as z:
                    scores = z["scores"]
                    if scores.shape == (layers, num_experts):
                        prof.scores = scores.astype(np.float64)
                        prof.steps = int(z["steps"])
                    else:
                        log.warning("Expert profile %s has shape %s, expected %s; starting fresh",
                                    p, scores.shape, (layers, num_experts))
            except (OSError, ValueError, KeyError) as e:
                log.warning("Could not read expert profile %s: %s", p, e)
        return prof

    def save(self, model_path: Path) -> Optional[Path]:
        if not self.dirty:
            return None
        p = self.sidecar(model_path)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp.npz")
            with self._lock:
                np.savez(tmp, scores=self.scores.astype(np.float32), steps=np.int64(self.steps))
                self.dirty = False
            os.replace(tmp, p)
            return p
        except OSError as e:
            log.warning("Could not save expert profile to %s: %s", p, e)
            return None

    def record(self, layer: int, inds: np.ndarray) -> None:
        if self.paused:
            return
        counts = np.bincount(inds.ravel().astype(np.int64), minlength=self.scores.shape[1])
        n_tokens = inds.size // max(inds.shape[-1], 1)
        with self._lock:
            if layer == 0:
                self.scores *= self.decay
                self.steps += 1
            self.scores[layer] += counts[: self.scores.shape[1]] / max(n_tokens, 1)
            self.dirty = True

    @property
    def observed(self) -> int:
        return int(np.count_nonzero(self.scores))

    def ranked(self) -> np.ndarray:
        """(layer, expert) pairs with a non-zero score, hottest first."""
        with self._lock:
            flat = self.scores.ravel()
            order = np.argsort(-flat, kind="stable")
            order = order[flat[order] > 0]
        return np.stack(np.unravel_index(order, self.scores.shape), axis=1)

    def to_dict(self) -> dict:
        return {"steps": self.steps, "observed_experts": self.observed, "size": int(self.scores.size)}


# ----------------------------------------------------------------- cache


@dataclass
class StreamStats:
    hits: int = 0
    misses: int = 0
    prefetch_issued: int = 0
    prefetch_used: int = 0
    bytes_read: int = 0
    read_seconds: float = 0.0
    evictions: int = 0
    warmed: int = 0

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
            "warmed": self.warmed,
        }


class ExpertCache:
    """Byte-bounded cache of experts keyed by (layer, expert).

    Eviction drops the expert with the lowest profile score (oldest first on
    ties); without a profile it is a plain LRU. ``set_capacity`` resizes it at
    run time so unused KV-cache budget can hold experts between long requests.
    """

    def __init__(
        self,
        capacity_bytes: int,
        io_threads: int = 8,
        profile: Optional[ExpertProfile] = None,
        pack: Optional[ExpertPack] = None,
    ):
        self.capacity_bytes = int(capacity_bytes)
        self.reader = ExpertReader()
        self.pack = pack
        self.profile = profile
        self.pool = ThreadPoolExecutor(max_workers=io_threads, thread_name_prefix="m5-expert-io")
        self._warm_pool: Optional[ThreadPoolExecutor] = None
        self._entries: "OrderedDict[Tuple[int, int], Tuple[Dict[str, mx.array], int]]" = OrderedDict()
        self._inflight: Dict[Tuple[int, int], Future] = {}
        self._prefetched: set = set()
        self._bytes = 0
        self.stats = StreamStats()
        self._lock = threading.Lock()  # stats
        self._elock = threading.RLock()  # entries
        self._closed = False

    @property
    def used_bytes(self) -> int:
        return self._bytes

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key) -> bool:
        return key in self._entries

    def _read_expert(self, locs: LayerLocs, layer: int, expert: int) -> Dict[str, np.ndarray]:
        t0 = time.perf_counter()
        out = self.pack.read(layer, expert) if self.pack is not None else self.reader.read_expert(locs, expert)
        nbytes = sum(a.nbytes for a in out.values())
        with self._lock:
            self.stats.read_seconds += time.perf_counter() - t0
            self.stats.bytes_read += nbytes
        return out

    def _submit(self, key, locs):
        fut = self._inflight.get(key)
        if fut is None:
            fut = self.pool.submit(self._read_expert, locs, key[0], key[1])
            self._inflight[key] = fut
        return fut

    def prefetch(self, layer: int, experts: Sequence[int], locs: LayerLocs) -> None:
        for e in experts:
            key = (layer, int(e))
            if key in self._entries or key in self._inflight:
                continue
            self._submit(key, locs)
            self._prefetched.add(key)
            self.stats.prefetch_issued += 1

    def _evict_to(self, target: int, protect: set) -> None:
        with self._elock:
            if self._bytes <= target:
                return
            cands = [k for k in self._entries if k not in protect]
            if not cands:
                return
            if self.profile is not None:
                arr = np.array(cands)
                order = np.argsort(self.profile.scores[arr[:, 0], arr[:, 1]], kind="stable")
            else:
                order = range(len(cands))  # insertion (recency) order
            for i in order:
                if self._bytes <= target:
                    break
                _, old_bytes = self._entries.pop(cands[i])
                self._bytes -= old_bytes
                self.stats.evictions += 1

    def _insert(self, key, arrays: Dict[str, mx.array], protect: set) -> None:
        nbytes = sum(a.nbytes for a in arrays.values())
        with self._elock:
            self._entries[key] = (arrays, nbytes)
            self._bytes += nbytes
            if self._bytes > self.capacity_bytes:
                # Evict a little past the line so the scan is not repeated for every insert.
                slack = min(8 * nbytes, self.capacity_bytes // 50)
                self._evict_to(self.capacity_bytes - slack, protect | {key})

    def set_capacity(self, capacity_bytes: int) -> None:
        capacity_bytes = max(0, int(capacity_bytes))
        if capacity_bytes == self.capacity_bytes:
            return
        self.capacity_bytes = capacity_bytes
        self._evict_to(capacity_bytes, set())

    def fetch(self, layer: int, experts: Sequence[int], locs: LayerLocs) -> List[Dict[str, mx.array]]:
        keys = [(layer, int(e)) for e in experts]
        protect = set(keys)
        found: Dict[Tuple[int, int], Dict[str, mx.array]] = {}
        with self._elock:
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
            with self._elock:
                if key in self._entries:  # the warm fill got there first
                    arrays = self._entries[key][0]
                else:
                    self._insert(key, arrays, protect)
            found[key] = arrays
        self._settle_mispredicted(layer, protect, locs)
        return [found[k] for k in keys]

    def _settle_mispredicted(self, layer: int, protect: set, locs: LayerLocs) -> None:
        """Move unused prefetches for this layer into the cache so they stay bounded."""
        stale = [k for k in self._prefetched if k[0] == layer and k not in protect]
        for key in stale:
            self._prefetched.discard(key)
            fut = self._inflight.pop(key, None)
            if fut is None:
                continue
            raw = fut.result()
            with self._elock:
                if key not in self._entries:
                    self._insert(key, {rel: to_mx(a, locs[rel].dtype) for rel, a in raw.items()}, protect)

    # ------------------------------------------------------------ warm fill
    def warm(self, layer_locs: Sequence[LayerLocs], limit_bytes: Optional[int] = None, threads: int = 2) -> int:
        """Read the profile's hottest experts in the background until the cache is ~full.

        Returns how many experts were scheduled. Reads use their own small pool
        so a request arriving mid-fill still gets the main I/O threads.
        """
        if self.profile is None or self._closed:
            return 0
        limit = self.capacity_bytes if limit_bytes is None else min(limit_bytes, self.capacity_bytes)
        sizes = [sum(t.per_expert_bytes for t in locs.values()) for locs in layer_locs]
        planned: List[Tuple[int, int]] = []
        total = self._bytes
        for layer, expert in self.profile.ranked():
            key = (int(layer), int(expert))
            if key in self._entries or total + sizes[key[0]] > limit:
                continue
            planned.append(key)
            total += sizes[key[0]]
        if not planned:
            return 0
        if self._warm_pool is None:
            self._warm_pool = ThreadPoolExecutor(max_workers=threads, thread_name_prefix="m5-expert-warm")

        def job(keys: List[Tuple[int, int]]):
            for key in keys:
                if self._closed:
                    return
                if key in self._entries or key in self._inflight:
                    continue
                locs = layer_locs[key[0]]
                raw = self._read_expert(locs, key[0], key[1])
                arrays = {rel: to_mx(a, locs[rel].dtype) for rel, a in raw.items()}
                nbytes = sum(a.nbytes for a in arrays.values())
                with self._elock:
                    if key in self._entries or self._bytes + nbytes > self.capacity_bytes:
                        continue
                    self._entries[key] = (arrays, nbytes)
                    self._entries.move_to_end(key, last=False)  # colder than anything used since
                    self._bytes += nbytes
                    self.stats.warmed += 1

        for i in range(threads):
            self._warm_pool.submit(job, planned[i::threads])
        return len(planned)

    def wait_warm(self, timeout: Optional[float] = None) -> None:
        if self._warm_pool is not None:
            self._warm_pool.shutdown(wait=True)
            self._warm_pool = None

    def reset_stats(self) -> None:
        self.stats = StreamStats()

    def clear(self) -> None:
        with self._elock:
            self._entries.clear()
            self._bytes = 0

    def close(self) -> None:
        self._closed = True
        if self._warm_pool is not None:
            self._warm_pool.shutdown(wait=True, cancel_futures=True)
        self.pool.shutdown(wait=False, cancel_futures=True)
        self.reader.close()
        if self.pack is not None:
            self.pack.close()
        self.clear()


# ------------------------------------------------------------- modules


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


_TOP_K_NAMES = ("num_experts_per_tok", "top_k", "n_activated_experts")


def _router_top_k(parent: nn.Module, default: int) -> int:
    for obj in (parent, parent.get("gate") if isinstance(parent, nn.Module) else None):
        if obj is None:
            continue
        for name in _TOP_K_NAMES:
            v = getattr(obj, name, None)
            if isinstance(v, int) and v > 0:
                return v
    return default


def set_active_experts(model: nn.Module, k: int) -> Tuple[int, int]:
    """Route each token to ``k`` experts instead of the model's default (Strata's "Speed" setting).

    Patches the top-k attribute on every MoE block and its router. Returns
    (default_k, layers_patched). Fewer experts read less from the SSD per
    token at a small quality cost; more than the default is refused.
    """
    modules = find_expert_modules(model)
    if not modules:
        raise ValueError("active_experts only applies to Mixture-of-Experts models.")

    def block_of(path: str) -> nn.Module:
        if path.endswith(".inner"):  # already wrapped by StreamingExperts
            path = path[: -len(".inner")]
        return _get_by_path(model, path.rsplit(".", 1)[0])

    default = _router_top_k(block_of(modules[0][0]), 0)
    if default <= 0:
        raise ValueError("Could not find this model's experts-per-token setting to override.")
    if not 1 <= k <= default:
        raise ValueError(f"active_experts must be between 1 and the model's default {default}.")
    patched = 0
    for path, _ in modules:
        parent = block_of(path)
        hit = False
        for obj in (parent, parent.get("gate")):
            if obj is None:
                continue
            for name in _TOP_K_NAMES:
                if isinstance(getattr(obj, name, None), int):
                    setattr(obj, name, k)
                    hit = True
        patched += hit
    return default, patched


class StreamingExperts(nn.Module):
    """Drop-in replacement for SwitchGLU/SwitchMLP that pulls experts from disk."""

    def __init__(
        self,
        inner: nn.Module,
        layer_id: int,
        locs: LayerLocs,
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
        if self._cache.profile is not None:
            self._cache.profile.record(self._layer_id, inds)
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
    layer_locs: List[LayerLocs]
    base_capacity_bytes: int = 0
    warm_scheduled: int = 0

    def to_dict(self) -> dict:
        return {
            "moe_layers": self.layers,
            "num_experts": self.num_experts,
            "expert_mb": round(self.expert_bytes / 1024**2, 2),
            "total_expert_gb": round(self.total_expert_bytes / 1024**3, 2),
            "cache_capacity_gb": round(self.cache.capacity_bytes / 1024**3, 2),
            "cache_base_gb": round(self.base_capacity_bytes / 1024**3, 2),
            "cache_used_gb": round(self.cache.used_bytes / 1024**3, 2),
            "cache_slots": int(self.cache.capacity_bytes // max(self.expert_bytes, 1)),
            "cached_experts": len(self.cache),
            "resident_fraction": round(len(self.cache) / max(self.layers * self.num_experts, 1), 4),
            "packed": self.cache.pack is not None,
            "profile": self.cache.profile.to_dict() if self.cache.profile else None,
            "warm_scheduled": self.warm_scheduled,
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


def locate_all_experts(model_path: Path, model: nn.Module) -> Tuple[List[LayerLocs], List[int]]:
    """Per MoE layer: where each expert parameter lives, and the layer's expert count."""
    index = read_safetensors_index(model_path)
    layer_locs: List[LayerLocs] = []
    counts: List[int] = []
    for path, inner in find_expert_modules(model):
        params = tree_flatten(inner.parameters())
        locs: LayerLocs = {}
        for rel, arr in params:
            locs[rel] = locate_expert_tensor(index, path, rel, tuple(arr.shape))
            if locs[rel].dtype not in _NP_DTYPES:
                raise ValueError(f"Unsupported expert dtype {locs[rel].dtype} for {path}.{rel}")
        layer_locs.append(locs)
        counts.append(params[0][1].shape[0])
    return layer_locs, counts


def estimate_expert_bytes(model_path: Path, model: nn.Module) -> int:
    layer_locs, _ = locate_all_experts(model_path, model)
    return sum(t.total_bytes for locs in layer_locs for t in locs.values())


def repack_experts(model: nn.Module, model_path: Path,
                   progress: Optional[Callable[[int, int], None]] = None) -> Path:
    """Write the contiguous expert sidecar for a lazily loaded MoE model."""
    layer_locs, counts = locate_all_experts(model_path, model)
    if not layer_locs:
        raise ValueError("This model has no routed-expert layers to repack.")
    return write_expert_pack(model_path, layer_locs, counts, progress)


def enable_expert_streaming(
    model: nn.Module,
    model_path: Path,
    cache_bytes: int,
    io_threads: int = 8,
    prefetch: bool = True,
    profile: bool = True,
    use_pack: bool = True,
    warm: bool = True,
) -> StreamingInfo:
    """Replace routed experts with disk-backed streaming modules.

    Must be called on a lazily loaded model *before* its parameters are
    evaluated, so the full expert tensors are never read into memory.
    """
    modules = find_expert_modules(model)
    if not modules:
        raise ValueError("Expert streaming requested but the model has no routed-expert layers.")
    layer_locs, counts = locate_all_experts(model_path, model)
    num_experts = max(counts)
    prof = ExpertProfile.load(model_path, len(modules), num_experts) if profile else None
    pack = ExpertPack.open(model_path, layer_locs) if use_pack else None
    cache = ExpertCache(cache_bytes, io_threads=io_threads, profile=prof, pack=pack)
    wrappers: List[Tuple[nn.Module, StreamingExperts]] = []
    expert_bytes = 0
    total = 0
    for layer_id, ((path, inner), locs, n_exp) in enumerate(zip(modules, layer_locs, counts)):
        expert_bytes = max(expert_bytes, sum(l.per_expert_bytes for l in locs.values()))
        total += sum(l.total_bytes for l in locs.values())
        for rel, arr in tree_flatten(inner.parameters()):
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

    info = StreamingInfo(len(wrappers), num_experts, expert_bytes, total, cache, layer_locs, cache_bytes)
    if warm and prof is not None and prof.steps:
        info.warm_scheduled = cache.warm(layer_locs)
        if info.warm_scheduled:
            log.info("Warming the expert cache with %d experts from the saved profile (%d steps)",
                     info.warm_scheduled, prof.steps)
    return info
