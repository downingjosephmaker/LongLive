#!/usr/bin/env python3
"""
LongLive 2.0 — FastAPI Server (Async Task + Multi-shot Anchor Edition)

Unified API: each shot can choose its conditioning mode (text-only or image),
and multi-shot generation supports per-shot first-frame anchoring via the
pipeline-level shot_anchors mechanism.

Base model: Wan2.2-TI2V-5B (no last-frame conditioning support — only
first-frame i2v / multi-shot first-frame anchors are honoured).

API Endpoints
-------------
  POST /api/generate                   — Unified generation (sync by default,
                                          async via ?async=true)
  POST /api/generate/upload-v2         — Same but with file uploads (sync/async)
  GET  /api/task/{task_id}             — Poll an async task
  DELETE /api/task/{task_id}           — Cancel a queued task
  GET  /api/tasks                      — List recent tasks (debug)
  GET  /api/status                     — Server & GPU status
  GET  /api/videos                     — List generated videos
  GET  /output/{filename}              — Download generated video
  POST /api/unload                     — Unload model to free GPU

Async Mode
----------
When `?async=true` is appended to the generate endpoints, the server returns
immediately with `{ "status": "queued", "task_id": "..." }` and the caller
polls `GET /api/task/{task_id}` until status is `succeeded` or `failed`.

Task statuses:
  queued  — waiting for GPU lock
  running — generation in progress (progress 0..100)
  succeeded — video ready
  failed  — error occurred
  cancelled — cancelled before execution

Form Upload
-----------
  POST /api/generate/upload-v2
    shots_json  (required) — JSON array of {prompt, first_frame?, blocks?}
    num_frames  (optional) — int, default 128
    seed        (optional) — int, default 0
    fps         (optional) — int, default 24
    async       (optional) — "true" for async mode
    img_<ref>   (optional) — uploaded images, ref matches first_frame value
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import time
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

# cuDNN 9.20.0 supports Blackwell (sm_120, RTX 5090) up to sm_121
# Enable cuDNN for conv/VAE acceleration
if torch.cuda.is_available():
    _gpu_name = torch.cuda.get_device_name(0)
    _cudnn_ver = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    print(f"[API] GPU: {_gpu_name}, cuDNN: {_cudnn_ver}", flush=True)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import UploadFile as StarletteUploadFile
from pydantic import BaseModel, Field

import uvicorn
from omegaconf import OmegaConf

# ---------------------------------------------------------------------------
# Resolution & Duration presets
# ---------------------------------------------------------------------------
RESOLUTION_PRESETS = {
    "480p":            {"width": 832,  "height": 480,  "label": "480p 横屏 (832×480)"},
    "480p_portrait":   {"width": 480,  "height": 832,  "label": "480p 竖屏 (480×832)"},
    "540p":            {"width": 960,  "height": 544,  "label": "540p 横屏 (960×544)"},
    "540p_portrait":   {"width": 544,  "height": 960,  "label": "540p 竖屏 (544×960)"},
    "720p":            {"width": 1280, "height": 704,  "label": "720p 横屏 (1280×704)"},
    "720p_portrait":   {"width": 704,  "height": 1280, "label": "720p 竖屏 (704×1280)"},
}

# 单镜头时长（秒），多镜头时 total_frames = per_shot_frames × shot_count
DURATION_PRESETS = {
    "1s":  1.0,
    "2s":  2.0,
    "3s":  3.0,
    "5s":  5.0,
}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(os.environ.get("LONGLIVE_OUTPUT", "/output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

UPLOAD_DIR = Path("/tmp/longlive_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

PUBLIC_BASE = os.environ.get("LONGLIVE_PUBLIC_BASE", "").rstrip("/")

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_pipe = None
_config = None
_device = None

_TASKS: Dict[str, "TaskState"] = {}
_TASK_LOCK = asyncio.Lock()


_LOADED_CONFIG_KIND: Optional[str] = None  # "t2v" | "i2v"

def _ensure_loaded(kind: str = "t2v"):
    """Load model on first call. kind selects T2V vs I2V config (different
    independent_first_frame setting). Reload pipeline if kind changes."""
    global _pipe, _config, _device, _LOADED_CONFIG_KIND
    if _pipe is not None and _LOADED_CONFIG_KIND == kind:
        return

    sys.path.insert(0, os.getcwd())

    from pipeline import CausalDiffusionInferencePipeline
    from utils.config import normalize_config
    from utils.inference_utils import load_generator_checkpoint, place_vae_for_streaming

    if kind == "i2v":
        default_cfg = "configs/inference_i2v.yaml"
    else:
        default_cfg = "configs/inference.yaml"
    config_path = os.environ.get("LONGLIVE_CONFIG_" + kind.upper(), None) \
                  or os.environ.get("LONGLIVE_CONFIG", None) \
                  or default_cfg
    print(f"[API] Building pipeline (kind={kind}, config={config_path})...", flush=True)

    # 旧 pipeline 释放
    if _pipe is not None:
        del _pipe
        _pipe = None
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    _config = normalize_config(OmegaConf.load(config_path))
    _device = torch.device("cuda")

    torch.set_grad_enabled(False)
    _pipe = CausalDiffusionInferencePipeline(_config, device=_device)

    ckpt_path = getattr(_config, "generator_ckpt", None)
    if ckpt_path:
        print(f"[API] Loading generator checkpoint: {ckpt_path}", flush=True)
        load_generator_checkpoint(_pipe.generator, ckpt_path)

    # Load everything to CPU in bf16 first — avoid OOM from loading ALL to GPU at once
    _pipe = _pipe.to(device='cpu', dtype=torch.bfloat16)
    place_vae_for_streaming(_pipe, _config)
    _pipe.generator.model.eval().requires_grad_(False)

    # Selective GPU loading: DiT on GPU, T5+VAE on CPU (moved on-demand during inference)
    if torch.cuda.get_device_properties(_device).total_memory < 30 * 1024**3:
        print(f"[API] GPU < 30GB, loading DiT to GPU only (T5+VAE stay on CPU)...", flush=True)
        _pipe.generator.to(device=_device, dtype=torch.bfloat16)
        import gc; gc.collect()
        torch.cuda.empty_cache()
        free_gb = torch.cuda.mem_get_info()[0] / 1024**3
        print(f"[API] DiT on GPU, VRAM free: {free_gb:.1f}GB", flush=True)
    else:
        _pipe = _pipe.to(device=_device, dtype=torch.bfloat16)

    _LOADED_CONFIG_KIND = kind
    print(f"[API] Pipeline ready (kind={kind}).", flush=True)


def _unload_model():
    """Unload model and free GPU memory."""
    global _pipe, _config, _device
    if _pipe is None:
        return False
    import gc
    print("[API] Unloading pipeline, freeing GPU memory...")
    _pipe = None
    _config = None
    gc.collect()
    torch.cuda.empty_cache()
    print("[API] Pipeline unloaded.")
    return True


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class ShotInput(BaseModel):
    """One shot in the generation sequence."""
    prompt: str = Field(..., description="Text prompt for this shot")
    first_frame: Optional[str] = Field(None, description="First-frame image path or upload filename (I2V / anchor)")
    blocks: Optional[int] = Field(None, description="Number of blocks for this shot (default: auto)")


class GenerateRequest(BaseModel):
    """Unified generation request."""
    shots: List[ShotInput] = Field(..., min_length=1, description="Shots to generate")
    resolution: str = Field("720p", description="分辨率预设 key")
    duration: str = Field("2s", description="时长预设 key")
    seed: int = Field(0, ge=0, le=100, description="随机种子 (0=随机, 1-100=指定)")
    fps: int = Field(24, description="Output video fps")


def _validate_and_resolve(resolution: str, duration: str, fps: int, num_shots: int = 1):
    """Validate preset keys and resolve to (width, height, num_frames, latent_h, latent_w).

    num_frames = per_shot_frames × num_shots, where per_shot_frames = duration_seconds × fps (aligned to 8).
    """
    if resolution not in RESOLUTION_PRESETS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的分辨率: '{resolution}'，可选: {list(RESOLUTION_PRESETS.keys())}",
        )
    if duration not in DURATION_PRESETS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的时长: '{duration}'，可选: {list(DURATION_PRESETS.keys())}",
        )

    preset = RESOLUTION_PRESETS[resolution]
    width, height = preset["width"], preset["height"]
    per_shot_seconds = DURATION_PRESETS[duration]

    latent_h = height // 16
    latent_w = width // 16
    # DiT patch_size=(1,2,2) requires latent H/W to be even
    if latent_h % 2 != 0 or latent_w % 2 != 0:
        raise HTTPException(
            status_code=400,
            detail=f"分辨率 {width}×{height} 的 latent 尺寸 ({latent_h}×{latent_w}) 必须均为偶数，"
                   f"请调整为 16 的偶数倍（如 {latent_h//2*2*16}×{latent_w//2*2*16}）",
        )

    # per_shot_frames aligned to fpb(8)
    fpb = 8
    per_shot_frames = max(fpb, (int(per_shot_seconds * fps) // fpb) * fpb)
    num_frames = per_shot_frames * num_shots
    total_seconds = num_frames / fps

    print(f"[API] Resolution: {width}×{height} (latent {latent_h}×{latent_w}), "
          f"Per-shot: {per_shot_frames} frames ({per_shot_seconds:.1f}s), "
          f"Shots: {num_shots}, Total: {num_frames} frames ({total_seconds:.1f}s @ {fps}fps)", flush=True)

    return width, height, num_frames, latent_h, latent_w


# ---------------------------------------------------------------------------
# Async task management
# ---------------------------------------------------------------------------
class TaskStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskState:
    """In-memory state for one async generation task."""

    def __init__(self, task_id: str):
        self.task_id: str = task_id
        self.status: TaskStatus = TaskStatus.QUEUED
        self.progress: int = 0  # 0..100 coarse-grained (0/50/100)
        self.seed: int = 0
        self.video_path: Optional[str] = None
        self.shots_info: List[Dict[str, Any]] = []
        self.error: Optional[str] = None
        self.created_at: float = time.time()
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        self.cancel_flag: bool = False
        self._tmp_files: List[Path] = []

    def to_dict(self, queue_position: Optional[int] = None) -> Dict[str, Any]:
        resp: Dict[str, Any] = {
            "task_id": self.task_id,
            "status": self.status.value,
            "progress": self.progress,
            "seed": self.seed,
            "shots_info": self.shots_info,
            "error": self.error,
            "video_url": _abs_url(self.video_path) if self.video_path else None,
            "video_path": self.video_path,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }
        if queue_position is not None:
            resp["queue_position"] = queue_position
        return resp


def _register_task(state: TaskState) -> None:
    _TASKS[state.task_id] = state


def _get_task(task_id: str) -> Optional[TaskState]:
    return _TASKS.get(task_id)


def _list_pending_before(task_id: str) -> int:
    """Count queued tasks ahead of this one."""
    count = 0
    for tid, s in _TASKS.items():
        if s.status == TaskStatus.QUEUED and s.created_at < _get_task(task_id).created_at:
            count += 1
    return count


# ---------------------------------------------------------------------------
# URL helper
# ---------------------------------------------------------------------------
def _abs_url(local_path: Optional[str]) -> Optional[str]:
    if not local_path:
        return None
    if PUBLIC_BASE:
        return f"{PUBLIC_BASE}/output/{Path(local_path).name}"
    return f"/output/{Path(local_path).name}"


# ---------------------------------------------------------------------------
# Mode detection
# ---------------------------------------------------------------------------
def _mode_of(shot: ShotInput) -> str:
    return "i2v" if shot.first_frame else "t2v"


def _build_shots_info(shots: List[ShotInput]) -> List[Dict[str, Any]]:
    return [{"prompt": s.prompt, "mode": _mode_of(s), "blocks": s.blocks} for s in shots]


# ---------------------------------------------------------------------------
# Block-count resolver
# ---------------------------------------------------------------------------
def _resolve_shot_block_counts(shots: List[ShotInput], total_blocks: int) -> List[int]:
    if len(shots) == 1:
        if shots[0].blocks is not None:
            return [shots[0].blocks]
        return [total_blocks]
    explicit = [s.blocks for s in shots if s.blocks is not None]
    if explicit:
        if any(s.blocks is None for s in shots):
            raise ValueError(
                "When setting blocks, all shots must have explicit blocks in multi-shot mode"
            )
        return [s.blocks for s in shots]
    equal = total_blocks // len(shots)
    remainder = total_blocks % len(shots)
    result = [equal + (1 if i < remainder else 0) for i in range(len(shots))]
    return result


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------
def _encode_frame(image_path: str) -> torch.Tensor:
    """Load image, preprocess, encode to VAE latent."""
    from utils.inference_utils import load_and_preprocess_image, image_to_tensor, encode_image_to_latent
    img = load_and_preprocess_image(image_path)
    tensor = image_to_tensor(img, _device)
    latent = encode_image_to_latent(_pipe.vae, tensor)
    return latent


def _resolve_img(value: str, image_map: Optional[Dict[str, str]]) -> str:
    """Resolve an image path: check upload map first, then treat as filesystem path."""
    if image_map and value in image_map:
        return image_map[value]
    return value


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------
def _run_generate(
    shots: List[ShotInput],
    num_frames: int,
    seed: int,
    fps: int,
    image_map: Optional[Dict[str, str]] = None,
    latent_h: int = 44,
    latent_w: int = 80,
) -> str:
    """Synchronous generation — called from asyncio.to_thread.

    Follows the official LongLive 2.0 inference contract documented in README.md:
        noise, prompts = prepare_single_prompt_inputs(config, prompt, device)
        video = pipe.inference(noise=noise, text_prompts=prompts)
        save_video(video[0], path, fps=24)

    Multi-shot extensions (shot_anchors + per-shot prompt list) are wired into
    pipe.inference()'s `shot_anchors` / `text_prompts` kwargs.
    """
    from utils.inference_utils import prepare_single_prompt_inputs, save_video

    # 根据是否提供首帧选 yaml：T2V 用 inference.yaml（无 indep_ff），I2V 用 inference_i2v.yaml
    has_first_frame = bool(shots and shots[0].first_frame)
    kind = "i2v" if has_first_frame else "t2v"
    _ensure_loaded(kind)

    # ── 合法化 num_frames ──────────────────────────────────────────
    # prepare_single_prompt_inputs 要求 num_frames % num_frame_per_block == 0
    fpb = int(getattr(_config, "num_frame_per_block", 8))
    legalised = max(fpb, (num_frames // fpb) * fpb)
    if legalised != num_frames:
        print(f"[API] num_frames {num_frames} → {legalised} (kind={kind}, fpb={fpb})", flush=True)
    num_frames = legalised
    # 写回 config 让 prepare_single_prompt_inputs / pipe.inference 都看到一致值
    try:
        _config.num_output_frames = num_frames
    except Exception:
        pass

    # ── 动态设置分辨率 ──────────────────────────────────────────────
    # image_or_video_shape = [batch, frames, C, H_latent, W_latent]
    orig_shape = list(_config.image_or_video_shape)
    new_shape = [orig_shape[0], num_frames, orig_shape[2], latent_h, latent_w]
    _config.image_or_video_shape = new_shape
    if orig_shape[3] != latent_h or orig_shape[4] != latent_w:
        print(f"[API] Latent shape changed: [{orig_shape[3]},{orig_shape[4]}] → [{latent_h},{latent_w}] "
              f"(pixel {latent_h*16}×{latent_w*16})", flush=True)
        # Resolution changed — must fully clear KV cache (size depends on frame_seq_length)
        import math as _math
        _pipe.frame_seq_length = _math.prod(new_shape[-2:]) // 4
        _pipe.clear_cache()
        torch.cuda.empty_cache()
        if hasattr(_pipe.generator, 'model') and hasattr(_pipe.generator.model, 'block_mask'):
            _pipe.generator.model.block_mask = None
        try:
            from wan_5b.modules.causal_model import _FREQS_I_CACHE
            _FREQS_I_CACHE.clear()
        except Exception:
            pass

    image_map = image_map or {}
    block_counts = _resolve_shot_block_counts(shots, num_frames)
    shot_anchors = []
    current_chunk = 0
    for i, (shot, blocks) in enumerate(zip(shots, block_counts)):
        if i > 0 and shot.first_frame:  # shot 0 用 initial_latent 路径，多镜头用 anchor
            path = _resolve_img(shot.first_frame, image_map)
            latent = _encode_frame(path)
            shot_anchors.append({"chunk_index": current_chunk, "latent": latent})
        current_chunk += blocks

    # 第一镜头 prompt 当 base prompt 给 prepare_single_prompt_inputs，按 num_blocks 平铺
    base_prompt = shots[0].prompt if shots else ""
    # 用确定性 generator 保证 seed 可复现（prepare_single_prompt_inputs 不接 seed kwarg，
    # 须通过 torch.Generator 传入）
    generator = torch.Generator(device=_device).manual_seed(int(seed) & 0x7FFFFFFF)
    noise, prompts = prepare_single_prompt_inputs(
        _config, base_prompt, _device, dtype=torch.bfloat16, generator=generator
    )

    # 多镜头：用每个 shot 的 prompt 覆盖对应 block 段（prompts 是 List[List[str]]）
    if len(shots) > 1:
        # prompts[batch=0] 是按 block 平铺的 base_prompt list；按 block_counts 切片覆盖
        seq = prompts[0]
        idx = 0
        for shot, blocks in zip(shots, block_counts):
            for b in range(blocks):
                if idx >= len(seq):
                    break
                seq[idx] = shot.prompt
                idx += 1
        prompts = [seq]

    # 首帧锚定（shot 0 走标准 I2V 路径）
    initial_latent = None
    if shots and shots[0].first_frame:
        first_frame_path = _resolve_img(shots[0].first_frame, image_map)
        initial_latent = _encode_frame(first_frame_path)

    # 推理：pipe.inference 是命名方法，非 __call__
    with torch.no_grad():
        print(f"[API] Starting inference (num_frames={len(noise[0])}, blocks={len(noise[0])//8})...", flush=True)

        need_offload = next(_pipe.text_encoder.parameters()).device.type != 'cuda'

        if need_offload:
            # Step 1: 全部在 CPU，搬 T5 到 GPU 编码
            print("[API] Moving T5 to GPU for encoding...", flush=True)
            _pipe.text_encoder.to(device='cuda')
            torch.cuda.empty_cache()

            # Monkey-patch: T5 encode 完后立即放回 CPU，DiT 搬到 GPU
            _orig_forward = _pipe.text_encoder.forward
            _offload_done = [False]
            def _patched_forward(*args, **kwargs):
                result = _orig_forward(*args, **kwargs)
                if not _offload_done[0]:
                    print("[API] T5 done, offload T5→CPU, DiT→GPU...", flush=True)
                    _pipe.text_encoder.to(device='cpu')
                    _pipe.generator.to(device='cuda', dtype=torch.bfloat16)
                    _pipe.generator.model.eval().requires_grad_(False)
                    torch.cuda.empty_cache()
                    free_gb = torch.cuda.mem_get_info()[0] / 1024**3
                    print(f"[API] Ready for DiT, VRAM free: {free_gb:.1f}GB", flush=True)
                    _offload_done[0] = True
                return result
            _pipe.text_encoder.forward = _patched_forward

            # Monkey-patch VAE decode: DiT→CPU, VAE→GPU, chunk-decode to avoid OOM
            _vae_moved = [False]
            def _patched_vae_decode(x):
                if not _vae_moved[0]:
                    print("[API] DiT sampling done, offload DiT→CPU, VAE→GPU...", flush=True)
                    # Move ALL generator submodules to CPU
                    _pipe.generator.to(device='cpu')
                    # Force release of any CUDA tensors still referenced
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    gc.collect()
                    torch.cuda.empty_cache()
                    # Verify DiT actually left GPU
                    free_gb = torch.cuda.mem_get_info()[0] / 1024**3
                    total_gb = torch.cuda.mem_get_info()[1] / 1024**3
                    print(f"[API] DiT offloaded, VRAM free: {free_gb:.1f}GB / {total_gb:.1f}GB", flush=True)
                    _pipe.vae.to(device='cuda')
                    torch.cuda.empty_cache()
                    free_gb = torch.cuda.mem_get_info()[0] / 1024**3
                    print(f"[API] VAE ready on GPU, VRAM free: {free_gb:.1f}GB", flush=True)
                    _vae_moved[0] = True
                # Use chunked decode to limit VRAM usage
                return _pipe.vae.decode_to_pixel_chunk(x, chunk_size=1)
            _pipe.vae.decode_to_pixel = _patched_vae_decode

        video_tensor = _pipe.inference(
            noise=noise,
            text_prompts=prompts,
            initial_latent=initial_latent,
            return_latents=False,
        )
        print(f"[API] Inference done, video_tensor shape={video_tensor.shape}", flush=True)

        if need_offload:
            _pipe.text_encoder.forward = _orig_forward
            _pipe.vae.to(device='cpu')
            torch.cuda.empty_cache()
            print("[API] VAE offloaded to CPU", flush=True)

        filename = f"longlive_{int(time.time())}_{seed}.mp4"
        out_path = OUTPUT_DIR / filename
        print(f"[API] Saving video to {out_path}...", flush=True)
        save_video(video_tensor[0], out_path, fps=fps)
        print(f"[API] Video saved: {out_path}", flush=True)

    return str(out_path)


# ---------------------------------------------------------------------------
# Background task runner
# ---------------------------------------------------------------------------
async def _run_task_in_background(
    state: TaskState,
    shots: List[ShotInput],
    num_frames: int,
    seed: int,
    fps: int,
    image_map: Optional[Dict[str, str]] = None,
    latent_h: int = 44,
    latent_w: int = 80,
):
    """Background coroutine: wait for GPU lock then run generation."""
    state.status = TaskStatus.RUNNING
    state.started_at = time.time()
    state.seed = seed
    try:
        def _progress_cb(pct: int) -> None:
            state.progress = pct

        path = await asyncio.to_thread(
            _run_generate, shots, num_frames, seed, fps, image_map, latent_h, latent_w
        )
        state.video_path = path
        state.status = TaskStatus.SUCCEEDED
        state.progress = 100
    except asyncio.CancelledError:
        state.status = TaskStatus.CANCELLED
    except BaseException as e:
        import traceback as _tb
        tb_str = _tb.format_exc()
        # 同时打到 stderr 方便 docker logs，并存进 state.error
        print(f"[API] Task {state.task_id} failed:\n{tb_str}", flush=True)
        state.status = TaskStatus.FAILED
        state.error = f"{type(e).__name__}: {e}\n{tb_str}"
    finally:
        state.finished_at = time.time()
        for p in state._tmp_files:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass


def _spawn_task(
    shots, num_frames, seed, fps, image_map=None, tmp_files=None, latent_h=44, latent_w=80
) -> TaskState:
    state = TaskState(task_id=uuid.uuid4().hex[:12])
    state._tmp_files = tmp_files or []
    _register_task(state)
    asyncio.create_task(_run_task_in_background(state, shots, num_frames, seed, fps, image_map, latent_h, latent_w))
    return state


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="LongLive 2.0")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    import traceback as _tb
    print(f"[API] Unhandled exception:\n{_tb.format_exc()}", flush=True)
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=500, detail=str(exc))

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/status")
async def status():
    """Return server and GPU status."""
    info: Dict[str, Any] = {
        "service": "LongLive 2.0 API",
        "version": "2.0.2",
        "model_loaded": _pipe is not None,
        "output_dir": str(OUTPUT_DIR),
        "public_base": PUBLIC_BASE or None,
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["vram_free_gb"] = round(torch.cuda.mem_get_info()[0] / 1024 ** 3, 1)
        info["vram_total_gb"] = round(torch.cuda.mem_get_info()[1] / 1024 ** 3, 1)
    async with _TASK_LOCK:
        info["tasks_total"] = len(_TASKS)
        info["tasks_running"] = sum(1 for s in _TASKS.values() if s.status == TaskStatus.RUNNING)
        info["tasks_queued"] = sum(1 for s in _TASKS.values() if s.status == TaskStatus.QUEUED)
    return info


def _pick_seed(seed_field: int) -> int:
    return seed_field if seed_field != 0 else int(time.time()) % (2 ** 31)


@app.post("/api/generate")
async def generate(
    req: GenerateRequest,
    async_: bool = Query(False, alias="async", description="Return immediately with task_id"),
):
    """Unified generation endpoint."""
    width, height, num_frames, latent_h, latent_w = _validate_and_resolve(
        req.resolution, req.duration, req.fps, len(req.shots)
    )
    s = _pick_seed(req.seed)

    if async_:
        state = _spawn_task(req.shots, num_frames, s, req.fps, image_map=None,
                            latent_h=latent_h, latent_w=latent_w)
        return {"status": "queued", "task_id": state.task_id}

    try:
        path = await asyncio.to_thread(
            _run_generate, req.shots, num_frames, s, req.fps, None, latent_h, latent_w
        )
        return {
            "status": "ok",
            "video": path,
            "seed": s,
            "shots_info": _build_shots_info(req.shots),
        }
    except Exception as e:
        import traceback as _tb
        print(f"[API] _run_generate FAILED:\n{_tb.format_exc()}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate/upload-v2")
async def generate_upload_v2(
    request: Request,
    shots_json: str = Form(...),
    resolution: str = Form("720p"),
    duration: str = Form("2s"),
    seed: str = Form("0"),
    fps: str = Form("24"),
    async_: bool = Query(False, alias="async", description="Return immediately with task_id"),
):
    """Generate with uploaded images.

    In shots_json, use first_frame values as filenames.
    Upload each image with the field name 'img_{filename}'.
    """
    try:
        shots_data = json.loads(shots_json)
        shots = [ShotInput(**s) for s in shots_data]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid shots JSON: {e}")

    f = int(fps)
    width, height, num_frames, latent_h, latent_w = _validate_and_resolve(resolution, duration, f, len(shots))
    s = _pick_seed(int(seed) if seed.lstrip("-").isdigit() else 0)

    form = await request.form()
    image_map: Dict[str, str] = {}
    saved_files: List[Path] = []

    for key, value in form.items():
        if isinstance(value, (UploadFile, StarletteUploadFile)) and value.filename:
            suffix = Path(value.filename or "img.png").suffix
            dest = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
            with open(dest, "wb") as fout:
                shutil.copyfileobj(value.file, fout)
            saved_files.append(dest)
            field_ref = key
            if field_ref.startswith("img_"):
                field_ref = field_ref[4:]
            image_map[field_ref] = str(dest)
            image_map[value.filename] = str(dest)

    if async_:
        state = _spawn_task(shots, num_frames, s, f, image_map, tmp_files=saved_files,
                            latent_h=latent_h, latent_w=latent_w)
        return {"status": "queued", "task_id": state.task_id}

    try:
        path = await asyncio.to_thread(
            _run_generate, shots, num_frames, s, f, image_map, latent_h, latent_w
        )
        return {
            "status": "ok",
            "video": path,
            "seed": s,
            "shots_info": _build_shots_info(shots),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if not async_:
            for p in saved_files:
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass


@app.get("/api/task/{task_id}")
async def get_task(task_id: str):
    """Poll an async task by id."""
    state = _get_task(task_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    queue_position = _list_pending_before(task_id) if state.status == TaskStatus.QUEUED else None
    return state.to_dict(queue_position=queue_position)


@app.delete("/api/task/{task_id}")
async def cancel_task(task_id: str):
    """Cancel a QUEUED task. RUNNING tasks cannot be safely interrupted."""
    state = _get_task(task_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    if state.status != TaskStatus.QUEUED:
        return {
            "task_id": task_id,
            "status": state.status.value,
            "cancelled": False,
            "reason": "Only QUEUED tasks can be cancelled",
        }
    state.cancel_flag = True
    return {"task_id": task_id, "status": "cancelled", "cancelled": True}


@app.get("/api/tasks")
async def list_tasks(limit: int = Query(50, ge=1, le=200)):
    """List recent tasks (newest first)."""
    async with _TASK_LOCK:
        items = list(_TASKS.values())
    items = list(reversed(items))[:limit]
    return {"total": len(items), "tasks": [s.to_dict() for s in items]}


@app.get("/api/videos")
async def list_videos():
    """List all generated videos."""
    videos = []
    for p in sorted(OUTPUT_DIR.glob("*.mp4")):
        videos.append({
            "name": p.name,
            "size_mb": round(p.stat().st_size / 1024 / 1024, 2),
            "url": _abs_url(str(p)),
        })
    return {"videos": videos}


@app.get("/output/{filename}")
async def download_video(filename: str):
    """Download a generated video file."""
    if "/" in filename or "\\" in filename or filename.startswith(".."):
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(str(path), media_type="video/mp4", filename=filename)


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------

@app.post("/api/unload")
async def unload_model():
    """Unload model to free GPU memory for other tasks."""
    async with _TASK_LOCK:
        running = any(s.status == TaskStatus.RUNNING for s in _TASKS.values())
    if running:
        raise HTTPException(status_code=409, detail="Cannot unload while tasks are running")
    freed = _unload_model()
    if not freed:
        return {"detail": "Model not loaded"}
    return {"detail": "Model unloaded, GPU memory freed"}


# ---------------------------------------------------------------------------
# Web UI
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def web_ui():
    return HTML_PAGE


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LongLive 2.0</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{--bg:#0a0a0b;--card:#141416;--border:#2a2a2e;--text:#e4e4e7;--dim:#71717a;--accent:#a78bfa;--accent2:#818cf8;--green:#34d399;--red:#f87171;--yellow:#fbbf24;--blue:#60a5fa;--radius:10px}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;line-height:1.6}
a{color:var(--accent);text-decoration:none}
a:hover{text-decoration:underline}

.header{border-bottom:1px solid var(--border);padding:16px 24px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px}
.header h1{font-size:20px;font-weight:600;background:linear-gradient(135deg,var(--accent),var(--blue));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.header-actions{display:flex;gap:8px;align-items:center}

.container{max-width:1200px;margin:0 auto;padding:24px}

/* Status bar */
.status-bar{display:flex;gap:12px;margin-bottom:24px;flex-wrap:wrap}
.stat{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);padding:12px 16px;flex:1;min-width:140px}
.stat-label{font-size:12px;color:var(--dim);margin-bottom:2px}
.stat-value{font-size:18px;font-weight:600}
.stat-value.green{color:var(--green)}
.stat-value.red{color:var(--red)}
.stat-value.yellow{color:var(--yellow)}

/* Cards */
.card{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);padding:20px;margin-bottom:20px}
.card h2{font-size:16px;font-weight:600;margin-bottom:16px;display:flex;align-items:center;gap:8px}
.card h2 .badge{font-size:11px;padding:2px 8px;border-radius:99px;font-weight:500}
.badge-green{background:rgba(52,211,153,.15);color:var(--green)}
.badge-red{background:rgba(248,113,113,.15);color:var(--red)}
.badge-blue{background:rgba(96,165,250,.15);color:var(--blue)}
.badge-yellow{background:rgba(251,191,36,.15);color:var(--yellow)}

/* Form */
.form-row{margin-bottom:14px}
.form-row label{display:block;font-size:13px;color:var(--dim);margin-bottom:4px}
textarea,input[type="number"],input[type="text"],select{
  width:100%;background:#1c1c1f;border:1px solid var(--border);border-radius:8px;
  padding:10px 12px;color:var(--text);font-size:14px;font-family:inherit;outline:none;transition:border .2s
}
textarea:focus,input:focus,select:focus{border-color:var(--accent)}
textarea{resize:vertical;min-height:60px}
.shots-list{display:flex;flex-direction:column;gap:10px}
.shot-item{background:#1c1c1f;border:1px solid var(--border);border-radius:8px;padding:12px;position:relative}
.shot-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;font-size:13px;font-weight:600;color:var(--dim)}
.shot-header .mode-tag{font-size:11px;padding:1px 6px;border-radius:4px;background:rgba(167,139,250,.15);color:var(--accent)}
.btn-remove{background:none;border:none;color:var(--red);cursor:pointer;font-size:18px;padding:0 4px;opacity:.7;transition:opacity .2s}
.btn-remove:hover{opacity:1}
.form-inline{display:flex;gap:8px;align-items:end}
.form-inline .form-row{flex:1;margin-bottom:0}
.file-label{display:inline-flex;align-items:center;gap:6px;padding:6px 12px;background:#1c1c1f;border:1px dashed var(--border);border-radius:6px;cursor:pointer;font-size:13px;color:var(--dim);transition:border .2s}
.file-label:hover{border-color:var(--accent);color:var(--text)}
.file-label input{display:none}

/* Buttons */
.btn{display:inline-flex;align-items:center;gap:6px;padding:8px 16px;border-radius:8px;font-size:14px;font-weight:500;cursor:pointer;border:none;transition:all .2s;font-family:inherit}
.btn-primary{background:linear-gradient(135deg,var(--accent),var(--accent2));color:#fff}
.btn-primary:hover{opacity:.9;transform:translateY(-1px)}
.btn-primary:disabled{opacity:.5;cursor:not-allowed;transform:none}
.btn-sm{padding:5px 12px;font-size:13px}
.btn-ghost{background:transparent;border:1px solid var(--border);color:var(--text)}
.btn-ghost:hover{border-color:var(--accent);color:var(--accent)}
.btn-danger{background:rgba(248,113,113,.15);color:var(--red);border:1px solid rgba(248,113,113,.3)}
.btn-danger:hover{background:rgba(248,113,113,.25)}

/* Shot toggle */
.shot-toggle{display:flex;gap:6px;margin-bottom:14px}
.shot-toggle button{padding:6px 12px;border-radius:6px;border:1px solid var(--border);background:transparent;color:var(--dim);cursor:pointer;font-size:13px;font-family:inherit}
.shot-toggle button.active{background:rgba(167,139,250,.15);border-color:var(--accent);color:var(--accent)}
.shot-toggle button:hover{border-color:var(--accent)}

/* Params row */
.params-row{display:flex;gap:12px;flex-wrap:wrap}
.params-row .form-row{flex:1;min-width:100px}
.params-row input{width:100%}

/* Tasks */
.task-item{display:flex;align-items:center;gap:12px;padding:10px 0;border-bottom:1px solid var(--border)}
.task-item:last-child{border-bottom:none}
.task-status{font-size:12px;padding:2px 8px;border-radius:4px;font-weight:500;white-space:nowrap}
.status-queued{background:rgba(251,191,36,.15);color:var(--yellow)}
.status-running{background:rgba(96,165,250,.15);color:var(--blue)}
.status-succeeded{background:rgba(52,211,153,.15);color:var(--green)}
.status-failed{background:rgba(248,113,113,.15);color:var(--red)}
.status-cancelled{background:rgba(113,113,122,.15);color:var(--dim)}
.task-info{flex:1;font-size:13px;overflow:hidden}
.task-info .prompt{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:400px;color:var(--dim)}
.progress-bar{width:120px;height:4px;background:#2a2a2e;border-radius:2px;overflow:hidden;flex-shrink:0}
.progress-bar .fill{height:100%;background:var(--accent);border-radius:2px;transition:width .3s}
.task-actions{flex-shrink:0}

/* Videos */
.video-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px}
.video-card{background:#1c1c1f;border:1px solid var(--border);border-radius:8px;overflow:hidden;transition:border .2s}
.video-card:hover{border-color:var(--accent)}
.video-card video{width:100%;display:block;max-height:200px;object-fit:contain;background:#000}
.video-meta{padding:10px 12px;font-size:12px;color:var(--dim);display:flex;justify-content:space-between;align-items:center}
.video-meta .name{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:180px}

/* VRAM bar */
.vram-bar{width:100%;height:8px;background:#1c1c1f;border-radius:4px;overflow:hidden;margin-top:6px}
.vram-bar .used{height:100%;border-radius:4px;transition:width .5s,background .5s}

/* Toast */
.toast{position:fixed;top:20px;right:20px;padding:12px 20px;border-radius:8px;font-size:14px;z-index:999;animation:slideIn .3s;pointer-events:none}
@keyframes slideIn{from{transform:translateX(100%);opacity:0}to{transform:translateX(0);opacity:1}}

.empty{text-align:center;padding:30px;color:var(--dim);font-size:14px}

/* Responsive */
@media(max-width:640px){
  .container{padding:16px}
  .params-row{flex-direction:column}
  .status-bar{flex-direction:column}
}
</style>
</head>
<body>

<div class="header">
  <h1>🐉 LongLive 2.0</h1>
  <div class="header-actions">
    <button class="btn btn-sm btn-ghost" onclick="loadStatus()">🔄 刷新</button>
    <button class="btn btn-sm btn-danger" id="btn-unload" onclick="unloadModel()" style="display:none">⏏ 卸载模型</button>
  </div>
</div>

<div class="container">
  <!-- Status -->
  <div class="status-bar" id="status-bar">
    <div class="stat"><div class="stat-label">GPU</div><div class="stat-value" id="s-gpu">-</div></div>
    <div class="stat"><div class="stat-label">模型</div><div class="stat-value" id="s-model">-</div></div>
    <div class="stat"><div class="stat-label">显存</div><div class="stat-value" id="s-vram">-</div>
      <div class="vram-bar"><div class="used" id="s-vram-bar"></div></div>
    </div>
    <div class="stat"><div class="stat-label">任务</div><div class="stat-value" id="s-tasks">-</div></div>
  </div>

  <!-- Generate -->
  <div class="card">
    <h2>🎬 生成视频</h2>
    <div class="shot-toggle">
      <button class="active" onclick="setShotMode(1,this)">单镜头</button>
      <button onclick="setShotMode(2,this)">双镜头</button>
      <button onclick="setShotMode(3,this)">三镜头</button>
      <button onclick="setShotMode(0,this)">自定义</button>
    </div>
    <div class="shots-list" id="shots-list"></div>
    <div class="params-row" style="margin-top:14px">
      <div class="form-row"><label>分辨率</label><select id="resolution">
        <option value="480p">480p 横屏 (832×480)</option>
        <option value="480p_portrait">480p 竖屏 (480×832)</option>
        <option value="540p">540p 横屏 (960×544)</option>
        <option value="540p_portrait">540p 竖屏 (544×960)</option>
        <option value="720p" selected>720p 横屏 (1280×704)</option>
        <option value="720p_portrait">720p 竖屏 (704×1280)</option>
      </select></div>
      <div class="form-row"><label>单镜头时长</label><select id="duration">
        <option value="1s">1 秒</option>
        <option value="2s" selected>2 秒</option>
        <option value="3s">3 秒</option>
        <option value="5s">5 秒</option>
      </select></div>
      <div class="form-row"><label>随机种子 (0=随机)</label><input type="number" id="seed" value="0" min="0" max="100" title="相同 prompt + 相同种子 = 相同视频"></div>
    </div>
    <div style="margin-top:16px;display:flex;gap:8px">
      <button class="btn btn-primary" id="btn-gen" onclick="generate()">🚀 开始生成</button>
      <button class="btn btn-ghost" onclick="loadTasks()">📋 任务列表</button>
      <button class="btn btn-ghost" onclick="loadVideos()">📁 视频列表</button>
    </div>
  </div>

  <!-- Tasks -->
  <div class="card" id="tasks-card" style="display:none">
    <h2>📋 任务队列</h2>
    <div id="tasks-list"></div>
  </div>

  <!-- Videos -->
  <div class="card" id="videos-card" style="display:none">
    <h2>📁 已生成视频</h2>
    <div class="video-grid" id="video-grid"></div>
  </div>
</div>

<script>
let shots = [{prompt:'',first_frame:null,blocks:null}];
let shotCount = 1;
let pollingTaskId = null;
let pollingTimer = null;

function $(id){return document.getElementById(id)}

function toast(msg, type='info'){
  const t = document.createElement('div');
  t.className = 'toast';
  t.style.background = type==='error'?'rgba(248,113,113,.9)':type==='success'?'rgba(52,211,153,.9)':'rgba(167,139,250,.9)';
  t.style.color = '#fff';
  t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(()=>t.remove(), 3000);
}

// --- Status ---
async function loadStatus(){
  try{
    const r = await fetch('/api/status');
    const d = await r.json();
    $('s-gpu').textContent = d.gpu || 'N/A';
    const ml = d.model_loaded;
    $('s-model').textContent = ml ? '已加载' : '未加载';
    $('s-model').className = 'stat-value ' + (ml?'green':'red');
    $('btn-unload').style.display = ml ? '' : 'none';
    if(d.vram_free_gb != null){
      const used = d.vram_total_gb - d.vram_free_gb;
      const pct = Math.round(used/d.vram_total_gb*100);
      $('s-vram').textContent = used.toFixed(1)+'/'+d.vram_total_gb+' GB';
      const bar = $('s-vram-bar');
      bar.style.width = pct+'%';
      bar.style.background = pct>90?'var(--red)':pct>70?'var(--yellow)':'var(--green)';
    }
    const parts = [];
    if(d.tasks_running) parts.push(d.tasks_running+' 运行');
    if(d.tasks_queued) parts.push(d.tasks_queued+' 排队');
    $('s-tasks').textContent = parts.length?parts.join(' / '):'空闲';
  }catch(e){console.error(e)}
}

async function unloadModel(){
  if(!confirm('确定卸载模型？这会释放 GPU 显存。')) return;
  try{
    const r = await fetch('/api/unload',{method:'POST'});
    const d = await r.json();
    toast(d.detail, 'success');
    loadStatus();
  }catch(e){toast(e.message,'error')}
}

// --- Shots ---
function setShotMode(n, btn){
  shotCount = n || 1;
  // Update active button style
  document.querySelectorAll('.shot-toggle button').forEach(b=>{
    b.classList.remove('active');
  });
  btn.classList.add('active');
  if(n===0){
    shots.push({prompt:'',first_frame:null,blocks:null});
    shotCount = shots.length;
  }
  shots = shots.slice(0, shotCount);
  while(shots.length < shotCount) shots.push({prompt:'',first_frame:null,blocks:null});
  renderShots();
}

function removeShot(i){
  shots.splice(i,1);
  shotCount = shots.length;
  renderShots();
}

function renderShots(){
  const c = $('shots-list');
  c.innerHTML = shots.map((s,i)=>{
    const mode = s.first_frame ? 'I2V' : 'T2V';
    return `<div class="shot-item">
      <div class="shot-header">
        <span>镜头 ${i+1} <span class="mode-tag" id="mode-${i}">${mode}</span></span>
        ${shots.length>1?`<button class="btn-remove" onclick="removeShot(${i})">✕</button>`:''}
      </div>
      <div class="form-row"><label>提示词</label><textarea id="prompt-${i}" placeholder="描述画面内容..." oninput="shots[${i}].prompt=this.value">${s.prompt}</textarea></div>
      <div class="form-inline">
        <div class="form-row"><label>首帧图片</label>
          <label class="file-label" id="flabel-${i}">📷 选择图片
            <input type="file" accept="image/*" onchange="handleFile(${i},this)">
          </label>
          <span id="fname-${i}" style="font-size:12px;color:var(--dim);margin-left:6px">${s.first_frame?'✓ '+s.first_frame:''}</span>
        </div>
      </div>
    </div>`;
  }).join('');
}

function handleFile(i, input){
  const file = input.files[0];
  if(!file) return;
  shots[i].first_frame = file.name;
  shots[i]._file = file;
  $('fname-'+i).textContent = '✓ '+file.name;
  $('mode-'+i).textContent = 'I2V';
}

// --- Generate ---
async function generate(){
  const btn = $('btn-gen');
  btn.disabled = true;
  btn.textContent = '⏳ 提交中...';

  const resolution = $('resolution').value;
  const duration = $('duration').value;
  const seed = parseInt($('seed').value)||0;
  const fps = 24;

  // Check if any shot has uploaded files → use upload-v2
  const hasFiles = shots.some(s=>s._file);
  if(hasFiles){
    const fd = new FormData();
    const shotsData = shots.map(s=>({prompt:s.prompt, first_frame:s.first_frame||null, blocks:s.blocks}));
    fd.append('shots_json', JSON.stringify(shotsData));
    fd.append('resolution', resolution);
    fd.append('duration', duration);
    fd.append('seed', seed);
    fd.append('fps', fps);

    shots.forEach((s,i)=>{
      if(s._file){
        fd.append('img_'+s.first_frame, s._file);
      }
    });

    try{
      const r = await fetch('/api/generate/upload-v2?async=true',{
        method:'POST', body:fd
      });
      if(!r.ok){const e=await r.json();throw new Error(e.detail||'Error');}
      const d = await r.json();
      toast('任务已提交: '+d.task_id,'success');
      startPolling(d.task_id);
    }catch(e){toast(e.message,'error')}
  } else {
    // JSON generate
    const body = {
      shots: shots.map(s=>({prompt:s.prompt, first_frame:s.first_frame||null, blocks:s.blocks})),
      resolution, duration, seed, fps
    };
    try{
      const r = await fetch('/api/generate?async=true',{
        method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)
      });
      if(!r.ok){const e=await r.json();throw new Error(e.detail||'Error');}
      const d = await r.json();
      toast('任务已提交: '+d.task_id,'success');
      startPolling(d.task_id);
    }catch(e){toast(e.message,'error')}
  }
  btn.disabled = false;
  btn.textContent = '🚀 开始生成';
}

function startPolling(taskId){
  pollingTaskId = taskId;
  $('tasks-card').style.display = '';
  if(pollingTimer) clearInterval(pollingTimer);
  pollingTimer = setInterval(()=>{pollTask(taskId);loadStatus();}, 3000);
  pollTask(taskId);
  loadTasks();
}

async function pollTask(taskId){
  try{
    const r = await fetch('/api/task/'+taskId);
    if(r.status===404){clearInterval(pollingTimer);return}
    const d = await r.json();
    loadTasks();
    if(d.status==='succeeded'){
      clearInterval(pollingTimer);
      toast('✅ 生成完成！','success');
      loadVideos();
    }else if(d.status==='failed'){
      clearInterval(pollingTimer);
      toast('❌ 失败: '+(d.error||'未知'),'error');
    }
  }catch(e){console.error(e)}
}

// --- Tasks ---
async function loadTasks(){
  try{
    const r = await fetch('/api/tasks?limit=20');
    const d = await r.json();
    $('tasks-card').style.display = '';
    if(!d.tasks.length){
      $('tasks-list').innerHTML = '<div class="empty">暂无任务</div>';
      return;
    }
    $('tasks-list').innerHTML = d.tasks.map(t=>{
      const pct = t.progress||0;
      return `<div class="task-item">
        <span class="task-status status-${t.status}">${t.status}</span>
        <div class="task-info">
          <div style="font-weight:500">${t.shots_info?(()=>{const p=t.shots_info.map(s=>s.prompt).join(' → ');return p.length>60?p.slice(0,60)+'…':p})():''}</div>
          <div class="prompt">seed: ${t.seed} | ${new Date(t.created_at*1000).toLocaleTimeString()}</div>
        </div>
        ${t.status==='running'?`<div class="progress-bar"><div class="fill" style="width:${pct}%"></div></div>`:''}
        ${t.video_url?`<div class="task-actions"><a href="${t.video_url}" class="btn btn-sm btn-ghost" target="_blank">▶</a></div>`:''}
      </div>`;
    }).join('');
  }catch(e){console.error(e)}
}

// --- Videos ---
async function loadVideos(){
  try{
    const r = await fetch('/api/videos');
    const d = await r.json();
    $('videos-card').style.display = '';
    if(!d.videos.length){
      $('video-grid').innerHTML = '<div class="empty">暂无视频</div>';
      return;
    }
    $('video-grid').innerHTML = d.videos.slice().reverse().map(v=>`
      <div class="video-card">
        <video controls preload="metadata" src="/output/${v.name}"></video>
        <div class="video-meta">
          <span class="name" title="${v.name}">${v.name}</span>
          <span>${v.size_mb} MB</span>
        </div>
      </div>`).join('');
  }catch(e){console.error(e)}
}

// --- Init ---
renderShots();
loadStatus();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Static files
# ---------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory=str(OUTPUT_DIR)), name="output")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    host = os.environ.get("LONGLIVE_HOST", "0.0.0.0")
    port = int(os.environ.get("LONGLIVE_PORT", "19521"))
    uvicorn.run(app, host=host, port=port)
