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

Async Mode
----------
When `?async=true` is appended to the generate endpoints, the server returns
immediately with `{ "status": "queued", "task_id": "..." }` and the caller
polls `GET /api/task/{task_id}` until status is `succeeded` or `failed`.

Task statuses:
  queued     -> waiting for GPU lock (queue_position included)
  running    -> generation in progress (progress: 0/50/100)
  succeeded  -> video ready (video_url contains absolute URL when
                LONGLIVE_PUBLIC_BASE is set)
  failed     -> error in `error` field
  cancelled  -> task was cancelled while queued

Multi-shot Anchoring
--------------------
- shots[0].first_frame  -> classic I2V (encoded as `initial_latent`)
- shots[i>0].first_frame -> encoded and injected as `shot_anchors[]` at the
  chunk index that starts that shot. Pipeline replaces the noise tensor's
  first frame at that chunk with the reference latent, while the existing
  scene-cut prefix triggers `multi_shot_sink` KV recache and RoPE phase
  offset for clean shot transitions.

The api auto-prefixes shot[i>0] prompts with the scene-cut sentinel so the
pipeline's `_is_shot_boundary` activates without callers needing to know
the convention.

Examples
--------
  # T2V single-shot
  {"shots": [{"prompt": "A robot walks through a lab"}]}

  # I2V single-shot
  {"shots": [{"prompt": "A robot walks", "first_frame": "/path/to/img.jpg"}]}

  # Multi-shot with per-shot first-frame anchoring
  {
    "shots": [
      {"prompt": "Robot stands at the lab entrance",
       "first_frame": "/prompts/shot0.jpg", "blocks": 8},
      {"prompt": "Robot walks to the workbench",
       "first_frame": "/prompts/shot1.jpg", "blocks": 8},
      {"prompt": "Robot examines a component",
       "first_frame": "/prompts/shot2.jpg", "blocks": 8}
    ],
    "num_frames": 192
  }
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import threading
import time
import uuid
from collections import OrderedDict
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from omegaconf import OmegaConf
from pydantic import BaseModel, Field
from starlette.datastructures import UploadFile as StarletteUploadFile

# Scene-cut sentinel — must match utils.dataset.DEFAULT_SCENE_CUT_PREFIX.
# Auto-prepended to prompts of shots after the first so the pipeline's
# `_is_shot_boundary` / `_is_scene_cut` heuristics fire and `multi_shot_sink`
# performs the proper KV recache + RoPE offset.
SCENE_CUT_PREFIX = "The scene transitions. "

# ---------------------------------------------------------------------------
# Lazy-initialised globals
# ---------------------------------------------------------------------------
_pipe = None
_config = None
_device = None


def _ensure_loaded():
    """Load model on first call."""
    global _pipe, _config, _device
    if _pipe is not None:
        return

    sys.path.insert(0, os.getcwd())

    from pipeline import CausalDiffusionInferencePipeline
    from utils.config import normalize_config, section_get
    from utils.inference_utils import load_generator_checkpoint, place_vae_for_streaming

    config_path = os.environ.get(
        "LONGLIVE_CONFIG",
        "configs/inference_i2v.yaml",
    )
    _config = normalize_config(OmegaConf.load(config_path))
    _device = torch.device("cuda")

    torch.set_grad_enabled(False)
    print("[API] Building pipeline...")
    _pipe = CausalDiffusionInferencePipeline(_config, device=_device)

    ckpt_path = getattr(_config, "generator_ckpt", None)
    if ckpt_path:
        print(f"[API] Loading generator checkpoint: {ckpt_path}")
        load_generator_checkpoint(_pipe.generator, ckpt_path)

    _pipe = _pipe.to(device=_device, dtype=torch.bfloat16)
    place_vae_for_streaming(_pipe, _config)
    _pipe.generator.model.eval().requires_grad_(False)
    print("[API] Pipeline ready.")


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
    num_frames: int = Field(128, description="Total latent frames (must be divisible by num_frame_per_block)")
    seed: int = Field(0, description="Random seed (0 = pick a fresh one)")
    fps: int = Field(24, description="Output video fps")


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
        }
        if self.status == TaskStatus.QUEUED and queue_position is not None:
            resp["queue_position"] = queue_position
        if self.started_at:
            resp["elapsed_seconds"] = int(
                (self.finished_at or time.time()) - self.started_at
            )
        if self.status == TaskStatus.SUCCEEDED:
            resp["video"] = self.video_path
            resp["video_url"] = _abs_url(self.video_path)
            resp["shots_info"] = self.shots_info
        if self.status == TaskStatus.FAILED:
            resp["error"] = self.error or "unknown error"
        if self.status == TaskStatus.CANCELLED:
            resp["error"] = "Task cancelled by caller"
        return resp


_TASKS: "OrderedDict[str, TaskState]" = OrderedDict()
_TASK_LOCK = threading.Lock()
_MAX_TASKS = 200

# Only ONE generation may touch the GPU at a time (VRAM constraint).
_GEN_LOCK = asyncio.Lock()


def _register_task(state: TaskState) -> None:
    with _TASK_LOCK:
        _TASKS[state.task_id] = state
        while len(_TASKS) > _MAX_TASKS:
            old_id, old_state = _TASKS.popitem(last=False)
            if old_state.status not in (TaskStatus.QUEUED, TaskStatus.RUNNING):
                continue
            _TASKS[old_id] = old_state
            _TASKS.move_to_end(old_id, last=False)
            break


def _get_task(task_id: str) -> Optional[TaskState]:
    with _TASK_LOCK:
        return _TASKS.get(task_id)


def _list_pending_before(task_id: str) -> int:
    """Return number of queued tasks ahead of the given task_id."""
    with _TASK_LOCK:
        count = 0
        for tid, st in _TASKS.items():
            if tid == task_id:
                return count
            if st.status == TaskStatus.QUEUED:
                count += 1
        return count


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="LongLive 2.0 API",
    version="2.0.2",
    description="T2V / I2V / Multi-shot anchor video generation API with async task model",
)

OUTPUT_DIR = Path(os.environ.get("LONGLIVE_OUTPUT", "/output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

UPLOAD_DIR = Path("/tmp/longlive_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

PUBLIC_BASE = os.environ.get("LONGLIVE_PUBLIC_BASE", "").rstrip("/")


def _abs_url(local_path: Optional[str]) -> Optional[str]:
    """Map an output file path to an externally reachable URL."""
    if not local_path:
        return None
    fname = Path(local_path).name
    if PUBLIC_BASE:
        return f"{PUBLIC_BASE}/output/{fname}"
    return f"/output/{fname}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _mode_of(shot: ShotInput) -> str:
    return "i2v" if shot.first_frame else "t2v"


def _build_shots_info(shots: List[ShotInput]) -> List[Dict[str, Any]]:
    return [
        {"index": i, "mode": _mode_of(sh), "blocks": sh.blocks}
        for i, sh in enumerate(shots)
    ]


def _resolve_shot_block_counts(shots: List[ShotInput], total_blocks: int) -> List[int]:
    """Distribute blocks across shots, respecting explicit block counts."""
    explicit_blocks = []
    auto_indices = []

    for i, shot in enumerate(shots):
        if shot.blocks is not None:
            explicit_blocks.append((i, shot.blocks))
        else:
            auto_indices.append(i)

    used = sum(b for _, b in explicit_blocks)
    if used > total_blocks:
        raise ValueError(
            f"Explicit block counts ({used}) exceed total blocks ({total_blocks})"
        )

    remaining = total_blocks - used
    num_auto = len(auto_indices)
    if num_auto == 0 and used != total_blocks:
        raise ValueError(
            f"Explicit block counts ({used}) don't equal total blocks ({total_blocks})"
        )

    result = [0] * len(shots)
    for i, b in explicit_blocks:
        result[i] = b

    if num_auto > 0:
        per_shot = remaining // num_auto
        remainder = remaining - per_shot * num_auto
        for i in auto_indices:
            result[i] = per_shot
        result[auto_indices[-1]] += remainder

    return result


def _encode_frame(image_path: str) -> torch.Tensor:
    """Load and encode an image to VAE latent. Returns (1, 1, C, h, w)."""
    from utils.inference_utils import (
        load_and_preprocess_image,
        image_to_tensor,
        encode_image_to_latent,
    )

    config = _config
    latent_shape = list(config.image_or_video_shape[2:])  # [C, h, w]
    pixel_h = latent_shape[1] * 8
    pixel_w = latent_shape[2] * 8

    img = load_and_preprocess_image(image_path, target_width=pixel_w, target_height=pixel_h)
    tensor = image_to_tensor(img, _device)
    latent = encode_image_to_latent(_pipe.vae, tensor)  # (1, 1, C, h, w)
    return latent


def _resolve_img(value: str, image_map: Optional[Dict[str, str]]) -> str:
    """Resolve an upload filename or absolute path to a real path on disk."""
    if image_map and value in image_map:
        return image_map[value]
    return value


# ---------------------------------------------------------------------------
# Core generation logic (runs in thread pool)
# ---------------------------------------------------------------------------
def _run_generate(
    shots: List[ShotInput],
    num_frames: int,
    seed: int,
    fps: int,
    image_map: Optional[Dict[str, str]] = None,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> str:
    """Unified generation — call via asyncio.to_thread."""
    _ensure_loaded()
    from utils.config import section_get
    from utils.inference_utils import save_video
    from utils.misc import set_seed

    if progress_cb:
        progress_cb(0)

    config = _config
    config.num_output_frames = num_frames
    frames_per_block = int(getattr(config, "num_frame_per_block", 1))

    if num_frames % frames_per_block != 0:
        raise ValueError(
            f"num_frames={num_frames} must be divisible by "
            f"num_frame_per_block={frames_per_block}"
        )

    num_blocks = num_frames // frames_per_block

    # Enable independent_first_frame if any shot has a first_frame
    has_first_frame = any(s.first_frame for s in shots)
    if has_first_frame:
        if not section_get(config, "inference", "independent_first_frame", False):
            config.inference.independent_first_frame = True

    set_seed(seed)
    seed_g = torch.Generator(device=_device)
    seed_g.manual_seed(seed)

    latent_shape = list(config.image_or_video_shape[2:])

    # Resolve block counts per shot
    block_counts = _resolve_shot_block_counts(shots, num_blocks)

    # Whether the chunk schedule has a leading t=0 frame chunk inserted by
    # the pipeline's independent_first_frame mode. The pipeline schedules
    # `[1] + [frames_per_block] * num_blocks` in that case (see
    # _inference_inner: `if self.independent_first_frame and initial_latent
    # is None: all_num_frames = [1] + all_num_frames`). When initial_latent
    # IS provided, the t=0 frame is consumed by the warm-up loop instead
    # and does NOT add an extra chunk to the main schedule.
    use_independent_first = bool(getattr(_config.inference, "independent_first_frame", False))
    initial_latent = None
    if shots[0].first_frame:
        img_path = _resolve_img(shots[0].first_frame, image_map)
        print(f"[API] Encoding first frame for shot 0: {img_path}")
        initial_latent = _encode_frame(img_path)

    # Compute the chunk index where each shot starts.
    # Chunk indexing (when initial_latent is provided, no leading [1] chunk):
    #   shot 0 occupies chunks [0, block_counts[0])
    #   shot i starts at sum(block_counts[:i])
    # When initial_latent is None and independent_first_frame is true, the
    # pipeline prepends one leading single-frame chunk → shot starts shift +1.
    chunk_offset = 0
    if initial_latent is None and use_independent_first:
        chunk_offset = 1

    shot_chunk_starts: List[int] = []
    running = chunk_offset
    for bc in block_counts:
        shot_chunk_starts.append(running)
        running += bc

    # Build per-block prompt list.
    # Prepend SCENE_CUT_PREFIX to the FIRST block of every shot[i>0] so that
    # `_is_shot_boundary` / `_is_scene_cut` activate inside the pipeline.
    all_prompts: List[str] = []
    shots_info: List[Dict[str, Any]] = []
    for i, (shot, bc) in enumerate(zip(shots, block_counts)):
        for blk in range(bc):
            if i > 0 and blk == 0:
                all_prompts.append(SCENE_CUT_PREFIX + shot.prompt)
            else:
                all_prompts.append(shot.prompt)
        shots_info.append({
            "index": i,
            "mode": _mode_of(shot),
            "prompt": shot.prompt[:80],
            "blocks": bc,
        })

    # Build shot_anchors for shots[i>0].first_frame so each shot starts
    # from its own visual reference. Shot 0's first_frame is already wired
    # through `initial_latent` above and must NOT also appear in anchors.
    shot_anchors: List[Dict[str, Any]] = []
    for i in range(1, len(shots)):
        shot = shots[i]
        if not shot.first_frame:
            continue
        img_path = _resolve_img(shot.first_frame, image_map)
        try:
            anchor = _encode_frame(img_path)
        except Exception as e:
            raise ValueError(f"shot[{i}].first_frame encode failed: {e}")
        shot_anchors.append({
            "chunk_index": shot_chunk_starts[i],
            "latent": anchor,
        })
        print(f"[API] Shot anchor prepared: shot={i} chunk_index={shot_chunk_starts[i]} src={img_path}")

    # Generate noise
    noise = torch.randn(
        [1, num_frames, *latent_shape],
        device=_device,
        dtype=torch.bfloat16,
        generator=seed_g,
    )

    if progress_cb:
        progress_cb(50)

    # Run inference
    print(f"[API] Generating video: {len(shots)} shots, {num_blocks} blocks, "
          f"mode(s)={[s['mode'] for s in shots_info]}, "
          f"anchors={len(shot_anchors)}")
    t0 = time.time()
    video = _pipe.inference(
        noise=noise,
        text_prompts=all_prompts,
        initial_latent=initial_latent,
        start_frame_index=0,
        shot_anchors=shot_anchors if shot_anchors else None,
    )
    elapsed = time.time() - t0
    print(f"[API] Generation completed in {elapsed:.1f}s")

    # Save
    out_name = f"gen_seed{seed}_{int(time.time())}.mp4"
    out_path = OUTPUT_DIR / out_name
    save_video(video[0], str(out_path), fps=fps)
    print(f"[API] Video saved: {out_path}")

    if progress_cb:
        progress_cb(100)
    return str(out_path)


# ---------------------------------------------------------------------------
# Async runner
# ---------------------------------------------------------------------------
async def _run_task_in_background(
    state: TaskState,
    shots: List[ShotInput],
    num_frames: int,
    seed: int,
    fps: int,
    image_map: Optional[Dict[str, str]],
) -> None:
    """Background coroutine: wait for GPU lock then run generation."""
    async with _GEN_LOCK:
        if state.cancel_flag:
            state.status = TaskStatus.CANCELLED
            state.finished_at = time.time()
            return

        state.status = TaskStatus.RUNNING
        state.started_at = time.time()

        def _cb(pct: int) -> None:
            state.progress = pct

        try:
            path = await asyncio.to_thread(
                _run_generate,
                shots,
                num_frames,
                seed,
                fps,
                image_map,
                _cb,
            )
            state.video_path = path
            state.shots_info = _build_shots_info(shots)
            state.progress = 100
            state.status = TaskStatus.SUCCEEDED
        except Exception as e:
            state.error = str(e)
            state.status = TaskStatus.FAILED
        finally:
            state.finished_at = time.time()
            for p in state._tmp_files:
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass


def _spawn_task(
    shots: List[ShotInput],
    num_frames: int,
    seed: int,
    fps: int,
    image_map: Optional[Dict[str, str]],
    tmp_files: Optional[List[Path]] = None,
) -> TaskState:
    task_id = uuid.uuid4().hex
    state = TaskState(task_id)
    state.seed = seed
    if tmp_files:
        state._tmp_files = list(tmp_files)
    _register_task(state)
    asyncio.create_task(
        _run_task_in_background(state, shots, num_frames, seed, fps, image_map)
    )
    return state


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
    with _TASK_LOCK:
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
    s = _pick_seed(req.seed)

    if async_:
        state = _spawn_task(req.shots, req.num_frames, s, req.fps, image_map=None)
        return {"status": "queued", "task_id": state.task_id}

    try:
        path = await asyncio.to_thread(
            _run_generate, req.shots, req.num_frames, s, req.fps
        )
        return {
            "status": "ok",
            "video": path,
            "seed": s,
            "shots_info": _build_shots_info(req.shots),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate/upload-v2")
async def generate_upload_v2(
    request: Request,
    shots_json: str = Form(...),
    num_frames: str = Form("128"),
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

    nf = int(num_frames)
    s = _pick_seed(int(seed) if seed.lstrip("-").isdigit() else 0)
    f = int(fps)

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
        state = _spawn_task(shots, nf, s, f, image_map, tmp_files=saved_files)
        return {"status": "queued", "task_id": state.task_id}

    try:
        path = await asyncio.to_thread(
            _run_generate, shots, nf, s, f, image_map
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
    with _TASK_LOCK:
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
