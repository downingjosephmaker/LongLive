# LongLive 2.0 API 调用文档

> 基础地址：`http://localhost:19521`
> 基座模型：Wan2.2-TI2V-5B（仅支持首帧 i2v，不支持尾帧）

---

## 概述

LongLive 2.0 提供统一的视频生成 API，能力如下：

| 能力 | 说明 |
|------|------|
| **T2V** 文本生视频 | 纯文本 prompt 生成视频 |
| **I2V** 图生视频 | 一张首帧 + 文本 prompt 生成视频 |
| **Multi-shot** 多镜头长视频 | 多个镜头连续生成，**每个镜头可独立指定 first_frame 做视觉锚定** |

**核心设计**：每个镜头（shot）通过 prompt + 可选 first_frame 独立控制。多镜头通过 shots 数组组合，每个 shot 的 first_frame 都会作为该镜头起点注入 diffusion（shot anchor），shot 边界由 `multi_shot_sink` 机制管理 KV cache 与 RoPE 相位。

**异步模式**：所有生成接口支持 `?async=true` 切换为任务模式，立即返回 task_id，客户端通过 `GET /api/task/{id}` 轮询，避免同步连接挂死。

---

## 1. 服务状态

```
GET /api/status
```

**响应示例：**

```json
{
  "service": "LongLive 2.0 API",
  "version": "2.0.2",
  "model_loaded": true,
  "output_dir": "/output",
  "public_base": "http://127.0.0.1:19521",
  "gpu": "NVIDIA GeForce RTX 5090",
  "vram_free_gb": 22.5,
  "vram_total_gb": 24.5,
  "tasks_total": 12,
  "tasks_running": 1,
  "tasks_queued": 0
}
```

---

## 2. 统一生成接口

### 2.1 JSON 模式（文件已在服务器上）

```
POST /api/generate              # 同步
POST /api/generate?async=true   # 异步（推荐生产用）
Content-Type: application/json
```

**请求体：**

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `shots` | Shot[] | ✅ | — | 镜头数组，至少 1 个 |
| `num_frames` | int | ❌ | 128 | 总 latent 帧数（必须能被 `num_frame_per_block=8` 整除） |
| `seed` | int | ❌ | 0 | 随机种子（0 = 自动生成） |
| `fps` | int | ❌ | 24 | 输出视频帧率 |

**Shot 对象：**

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `prompt` | string | ✅ | 文本描述 |
| `first_frame` | string | ❌ | 首帧图片路径（容器内路径）。**shot[0]** 用作 i2v 起点；**shot[i>0]** 作为该镜头的 shot anchor 注入 diffusion |
| `blocks` | int | ❌ | 该镜头占用的 block 数（默认自动均分） |

---

### 2.2 上传模式（图片通过接口上传）

```
POST /api/generate/upload-v2              # 同步
POST /api/generate/upload-v2?async=true   # 异步
Content-Type: multipart/form-data
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `shots_json` | string | ✅ | shots 数组的 JSON 字符串 |
| `num_frames` | string | ❌ | 总帧数，默认 "128" |
| `seed` | string | ❌ | 随机种子，默认 "0" |
| `fps` | string | ❌ | 帧率，默认 "24" |
| `img_{name}` | file | ❌ | 图片文件，name 对应 shot 中 `first_frame` 的值 |

> 上传文件时，shot 中的 `first_frame` 填写文件名（不含 `img_` 前缀），上传时用 `img_{文件名}` 作为字段名。

---

## 3. 调用示例

### 3.1 T2V — 文本生视频（单镜头）

```bash
curl -X POST http://localhost:19521/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "shots": [
      {"prompt": "A compact silver robot walks through a clean robotics lab."}
    ],
    "num_frames": 128,
    "seed": 42
  }'
```

**响应：**

```json
{
  "status": "ok",
  "video": "/output/gen_seed42_1747800000.mp4",
  "seed": 42,
  "shots_info": [
    {"index": 0, "mode": "t2v", "blocks": null}
  ]
}
```

---

### 3.2 I2V — 单图生视频

**JSON 模式（图片已在服务器上）：**

```bash
curl -X POST http://localhost:19521/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "shots": [
      {
        "prompt": "A silver robot walks forward, examining equipment on the workbench.",
        "first_frame": "/prompts/robot_standing.jpg"
      }
    ],
    "num_frames": 128,
    "seed": 42
  }'
```

**上传模式（图片通过接口上传）：**

```bash
curl -X POST http://localhost:19521/api/generate/upload-v2 \
  -F 'shots_json=[{"prompt":"A silver robot walks forward, examining equipment.","first_frame":"robot.png"}]' \
  -F "img_robot.png=@/path/to/robot_standing.jpg" \
  -F "num_frames=128" \
  -F "seed=42"
```

> 上传字段名 `img_robot.png` 对应 shot 中的 `first_frame="robot.png"`。

---

### 3.3 Multi-shot — 多镜头长视频（视觉锚定）

#### 场景 A：纯文本多镜头

```bash
curl -X POST http://localhost:19521/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "shots": [
      {"prompt": "A silver robot stands at the entrance of a clean robotics lab."},
      {"prompt": "The robot walks forward, examining equipment on the workbench."},
      {"prompt": "The robot picks up a small component and inspects it closely."},
      {"prompt": "The robot places the component down and turns toward the camera."}
    ],
    "num_frames": 256,
    "seed": 42
  }'
```

> 4 个镜头共 256 帧，自动均分每镜头 8 个 block（256 ÷ 8 ÷ 4）。
> 服务端会自动在 shot[i>0] 的第一个 block prompt 前加 `"The scene transitions. "` 前缀，触发 `multi_shot_sink` 的 KV recache 与 RoPE 相位偏移。

#### 场景 B：每个镜头都用独立首帧锚定（推荐）

```bash
curl -X POST http://localhost:19521/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "shots": [
      {
        "prompt": "Robot stands at the lab entrance",
        "first_frame": "/prompts/shot0.jpg",
        "blocks": 8
      },
      {
        "prompt": "Robot walks along the workbench",
        "first_frame": "/prompts/shot1.jpg",
        "blocks": 8
      },
      {
        "prompt": "Robot interacts with a terminal",
        "first_frame": "/prompts/shot2.jpg",
        "blocks": 8
      }
    ],
    "num_frames": 192,
    "seed": 42
  }'
```

每个 shot 的 `first_frame` 都会被编码成 latent 并在对应 chunk 起点注入 diffusion —— 这样不同镜头视觉一致性大幅提升，shot 边界由 pipeline 的 multi_shot_sink + RoPE offset 接管。

#### 场景 C：上传模式 + 多镜头锚定

```bash
curl -X POST http://localhost:19521/api/generate/upload-v2 \
  -F 'shots_json=[
    {"prompt":"Robot enters lab","first_frame":"s0.jpg","blocks":8},
    {"prompt":"Robot walks to workbench","first_frame":"s1.jpg","blocks":8},
    {"prompt":"Robot picks up component","first_frame":"s2.jpg","blocks":8}
  ]' \
  -F "img_s0.jpg=@/path/to/shot0_enter.jpg" \
  -F "img_s1.jpg=@/path/to/shot1_walk.jpg" \
  -F "img_s2.jpg=@/path/to/shot2_pickup.jpg" \
  -F "num_frames=192" \
  -F "seed=42"
```

---

## 4. 辅助接口

### 4.1 查看已生成视频

```
GET /api/videos
```

```json
{
  "videos": [
    {"name": "gen_seed42_1747800000.mp4", "size_mb": 35.2,
     "url": "http://127.0.0.1:19521/output/gen_seed42_1747800000.mp4"},
    {"name": "gen_seed42_1747800100.mp4", "size_mb": 52.8,
     "url": "http://127.0.0.1:19521/output/gen_seed42_1747800100.mp4"}
  ]
}
```

### 4.2 下载视频

```
GET /output/{filename}
```

```bash
curl -O http://localhost:19521/output/gen_seed42_1747800000.mp4
```

---

## 5. 参数说明

### num_frames（总帧数）

- 必须能被 `num_frame_per_block`（默认 8）整除
- 推荐值：128（约 5 秒）、256（约 10 秒）、512（约 20 秒）
- 帧数越多，生成时间越长，显存占用越高
- 128 latent frames ≈ 5 秒 @ 24fps

### blocks（每镜头的 block 数）

- 1 block = 8 latent frames ≈ 2.7 秒视频
- 不指定则自动均分
- 可手动指定实现不等长镜头：
  ```json
  {"shots": [
    {"prompt": "快节奏动作", "blocks": 4},
    {"prompt": "缓慢推进", "blocks": 12},
    {"prompt": "结尾定格", "blocks": 4}
  ]}
  ```
  以上示例：4 + 12 + 4 = 20 blocks → num_frames = 160

### seed

- 固定 seed 可复现结果
- 0 表示自动生成随机 seed
- 响应中会返回实际使用的 seed

### 图片要求

- 格式：JPG / PNG / WEBP
- 分辨率会自动 resize 到模型需要的尺寸（768×704）
- 建议提供接近 16:9 比例的图片，避免严重裁切

---

## 6. 错误处理

**请求格式错误（400）：**
```json
{"detail": "Invalid shots JSON: ..."}
```

**生成失败（500）：**
```json
{"detail": "CUDA out of memory. Tried to allocate 2.5 GiB"}
```

**文件不存在（404）：**
```json
{"detail": "Video not found"}
```

**建议：** 遇到 OOM（显存不足），减小 `num_frames` 或减少镜头数量。

---

## 7. Docker 管理

```bash
cd /home/local_video/LongLive

# 构建镜像（首次或代码更新后）
docker compose build

# 启动服务（后台）
docker compose up -d

# 查看日志
docker compose logs -f longlive

# 停止服务
docker compose down

# 全部（停止 + 构建 + 启动）
docker compose down && docker compose up -d --build
```

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LONGLIVE_CONFIG` | `configs/inference_i2v.yaml` | 配置文件路径 |
| `LONGLIVE_OUTPUT` | `/output` | 视频输出目录 |
| `LONGLIVE_HOST` | `0.0.0.0` | 监听地址 |
| `LONGLIVE_PORT` | `19521` | 监听端口 |
| `LONGLIVE_PUBLIC_BASE` | （无） | 异步任务 `video_url` 返回的绝对 URL 前缀。同机部署（Windows + WSL2）建议设为 `http://127.0.0.1:19521`；跨主机部署改为对应可达 IP |

### Volume 挂载

| 容器路径 | 宿主机 | 说明 |
|----------|--------|------|
| `/models` | `/home/local_video/models` | 模型文件（只读） |
| `/prompts` | `/home/local_video/LongLive/inference_prompts` | prompt 和图片资源（只读） |
| `/output` | `/home/local_video/LongLive/videos/output` | 生成的视频 |

> 图片放在 `/prompts/` 目录下即可在 JSON 请求中引用，例如 `first_frame: "/prompts/robot.jpg"`。

---

## 8. 异步任务模式

### 8.1 为什么需要异步模式

LongLive 单次生成动辄数分钟，同步接口会让调用方挂着 HTTP 连接等待，中间任何代理 / 反向代理 / 网关都可能在 60s ~ 180s 切断空闲连接。异步模式把"提交"与"等待"拆开：

- HTTP 连接秒级返回，无超时风险
- 调用方可以做进度条 UI
- 多客户端可同时排队（GPU 仍是单任务串行，超出自动排队）

### 8.2 切换开关

任何 `POST /api/generate*` 端点追加 `?async=true` 即变成异步。

### 8.3 异步流程

```
1. POST /api/generate?async=true        → 立即返回 { status: queued, task_id }
2. GET  /api/task/{task_id}             → 周期轮询直到 status = succeeded
3. GET  {video_url}                     → 下载成片字节流
```

### 8.4 接口详情

#### GET /api/task/{task_id}

| 字段 | 类型 | 出现时机 | 说明 |
|------|------|----------|------|
| `task_id` | string | 总是 | 任务 ID |
| `status` | string | 总是 | `queued` / `running` / `succeeded` / `failed` / `cancelled` |
| `progress` | int | 总是 | 0..100 粗粒度（0/50/100） |
| `seed` | int | 总是 | 实际使用的 seed |
| `queue_position` | int | status=queued | 队列前还有几个任务 |
| `elapsed_seconds` | int | status≥running | 从 running 开始计的秒数 |
| `video` | string | status=succeeded | 容器内绝对路径 |
| `video_url` | string | status=succeeded | 外部可访问 URL（依赖 `LONGLIVE_PUBLIC_BASE`） |
| `shots_info` | array | status=succeeded | 每个镜头的模式回报 |
| `error` | string | status=failed/cancelled | 错误描述 |

**示例：**

```json
// running
{
  "task_id": "9c1e4a8e7b2d4f5c",
  "status": "running",
  "progress": 50,
  "seed": 42,
  "elapsed_seconds": 78
}

// succeeded
{
  "task_id": "9c1e4a8e7b2d4f5c",
  "status": "succeeded",
  "progress": 100,
  "seed": 42,
  "elapsed_seconds": 305,
  "video": "/output/gen_seed42_1747800000.mp4",
  "video_url": "http://127.0.0.1:19521/output/gen_seed42_1747800000.mp4",
  "shots_info": [
    { "index": 0, "mode": "i2v", "blocks": 16 }
  ]
}

// failed
{
  "task_id": "9c1e4a8e7b2d4f5c",
  "status": "failed",
  "progress": 50,
  "seed": 42,
  "error": "CUDA out of memory. Tried to allocate 2.5 GiB"
}
```

#### DELETE /api/task/{task_id}

取消一个**尚在排队**的任务。已经 `running` 的任务无法安全中断。

```json
// 排队中 → 成功取消
{ "task_id": "...", "status": "cancelled", "cancelled": true }

// 已运行 → 拒绝取消
{ "task_id": "...", "status": "running", "cancelled": false,
  "reason": "Only QUEUED tasks can be cancelled" }
```

#### GET /api/tasks?limit=50

调试用：列出最近的任务（最新优先）。

### 8.5 LONGLIVE_PUBLIC_BASE

异步任务在 `succeeded` 状态会返回 `video_url`：

- 设了 `LONGLIVE_PUBLIC_BASE` → 返回**完整绝对 URL**，调用方直接 GET 即可
- 未设 → 返回相对路径 `/output/xxx.mp4`，调用方自行拼接 base URL

**部署建议：**

| 场景 | 推荐值 |
|------|--------|
| Windows + WSL2 同机（已通过 localhost 桥接） | `http://127.0.0.1:19521` |
| 同一台 Linux 宿主机 | `http://127.0.0.1:19521` |
| 跨主机（局域网） | `http://<LongLive 主机 LAN IP>:19521` |
| 公网部署 | `https://<域名>` 并在反向代理处理 TLS |

强烈建议设置该变量，否则调用方需要自己拼 base URL。

### 8.6 AiNet 集成参考

AiNet 通过 `LongLiveClient : IVideoClient` 接入：

| AiNet 调用 | LongLive 端点 |
|---|---|
| `Text2VideoAsync` | `POST /api/generate?async=true` |
| `Image2VideoAsync`（首帧） | `POST /api/generate/upload-v2?async=true` |
| `QueryAsync(token)` | `GET /api/task/{token}` |

AiNet `VideoGenResult` 字段映射：

| LongLive 响应 | AiNet `VideoGenResult` |
|---|---|
| `task_id` | `ExternalToken` |
| `status=queued` | `Status="in_queue"` |
| `status=running` | `Status="in_progress"` + `Progress` |
| `status=succeeded` + `video_url` | `Status="succeeded"` + `VideoBytes`（下载后填充） |
| `status=failed` + `error` | `Status="failed"` + `ErrorMessage` |

---

## 9. 真实能力声明

| 字段 / 模式 | API 行为 | Pipeline 真实行为 |
|------------|---------|--------------------|
| `shots[i].prompt` | ✅ | ✅ 每个镜头 prompt 按 block_counts 重复并按序拼接 |
| `shots[0].first_frame` | ✅ | ✅ 编码为 `initial_latent` 注入 diffusion，t=0 锚定 |
| `shots[i>0].first_frame` | ✅ | ✅ 编码为 `shot_anchors[i].latent`，在对应 chunk 起点注入 noise，**真正生效** |
| `shots[i].blocks` | ✅ | ✅ 控制每个镜头占用的 block 数（决定时长） |
| `last_frame` | ❌ | ❌ **wan_5b 基座不支持尾帧条件，已移除该字段** |

**多镜头工作机理**：

1. api_server 编码每个 shot 的 `first_frame` → shot_anchors 列表
2. 自动在 shot[i>0] 第一个 block 的 prompt 前加 `"The scene transitions. "` 触发 shot boundary
3. pipeline `_inference_inner` 进入新 chunk 时，把对应 shot anchor 替换 noise 第一帧
4. `_is_scene_cut` 触发 `_zero_kv_data` + `_pin_current_chunk`，KV cache 干净重启
5. `multi_shot_rope_offset` 按 shot 索引偏移 RoPE 相位，时间一致性强化

---

## 10. Docker Compose 配置参考

```yaml
services:
  longlive:
    build: .
    image: longlive2:latest
    container_name: longlive2
    ports:
      - "19521:19521"
    volumes:
      - /home/local_video/models:/models:ro
      - /home/local_video/LongLive/inference_prompts:/prompts:ro
      - /home/local_video/LongLive/videos/output:/output
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - LONGLIVE_PUBLIC_BASE=http://127.0.0.1:19521
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    shm_size: "8g"
    restart: unless-stopped
```
