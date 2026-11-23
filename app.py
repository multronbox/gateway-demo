"""
多模型 API Gateway — MVP 起步骨架（Gemini 先行，Claude 預留）
Traditional Chinese comments. No real provider key is in this file.

設計原則（沿用 api-gateway-mvp.md）：
- 真 API key 永遠只留喺 VM 嘅環境變數 / secrets，唔會經 API 回傳、
  唔會寫入 log，亦唔會貼入 chat。
- 公司 / agent 只用 gateway token 呼叫本服務。
- Claude 路由目前停用，只留插槽，返 503。

未完成、動工時先填：
- Gemini 生圖嗰段要對返 Google 官方文件同當時嘅 model ID 再接，
  唔好憑記憶寫死（model 名同定價經常變，見 README 嘅價錢表）。

--- Codespaces 示範模式（DEMO_MODE）---
- 環境變數 DEMO_MODE=1 時進入示範模式（devcontainer 預設開啟）：
  - POST /v1/run 同 /v1/image 唔使用任何真 provider key，會回傳
    明確標示 simulated 嘅模擬回應，同時展示 routing 決定（揀邊個
    provider／model、點解咁揀）。
  - gateway token 驗證會跳過（訪客唔會被 401 擋住），rate limit 仍然生效。
- DEMO_MODE 唔開（或唔設定）時，行為同原本骨架完全一樣：
  token 驗證照舊、未設定 Gemini 返 501、Claude 插槽返 503。
- 包入面冇任何 key 或 token；要行真 provider，請喺部署環境用
  環境變數供應 key，並關閉 DEMO_MODE。
"""
import json
import os
import time
from datetime import datetime, timezone

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

app = FastAPI(title="Taiwan client API gateway (MVP starter)")

# --- 設定：全部由環境變數讀入，唔好寫死喺 code 入面 ---
GATEWAY_TOKENS = {
    t.strip() for t in os.environ.get("GATEWAY_TOKENS", "").split(",") if t.strip()
}
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")  # 只供 VM 內部使用
GEMINI_IMAGE_MODEL = os.environ.get("GEMINI_IMAGE_MODEL", "")  # 動工時填官方 model ID
CLAUDE_ENABLED = os.environ.get("CLAUDE_ENABLED", "false").lower() == "true"
USAGE_LOG = os.environ.get("USAGE_LOG", "usage.jsonl")
# 簡單 rate limit：每 token 每分鐘最多 N 次，MVP 夠用就得
RATE_LIMIT_PER_MINUTE = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "30"))
# 示範模式：1/true/yes 先算開；devcontainer 會預設 DEMO_MODE=1
DEMO_MODE = os.environ.get("DEMO_MODE", "0").strip().lower() in ("1", "true", "yes")

_hits: dict[str, list[float]] = {}


class RunRequest(BaseModel):
    task: str = Field(..., description="image | code | chat")
    prompt: str = Field(..., min_length=1)
    size: str = Field(default="1K", description="生圖尺寸：1K / 2K / 4K（以官方為準）")


class ImageRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    size: str = Field(default="1K")


def _log_usage(token_label: str, task: str, model: str, ok: bool, note: str = "") -> None:
    """只記用量／任務／模型，絕對唔記 key 同 prompt 全文。"""
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "token": token_label[:8] + "…",  # 只留頭幾隻字識別，唔記全 token
        "task": task,
        "model": model,
        "ok": ok,
        "note": note,
    }
    with open(USAGE_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _check_token(authorization: str | None) -> str:
    # 示範模式：跳過 token 驗證，訪客唔會被 401 擋住；
    # rate limit 照樣生效（用固定標籤 "demo" 計數）。
    if DEMO_MODE:
        now = time.time()
        window = [t for t in _hits.get("demo", []) if now - t < 60]
        if len(window) >= RATE_LIMIT_PER_MINUTE:
            raise HTTPException(status_code=429, detail="rate limit exceeded")
        window.append(now)
        _hits["demo"] = window
        return "demo"
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing gateway token")
    token = authorization[7:].strip()
    if not GATEWAY_TOKENS or token not in GATEWAY_TOKENS:
        raise HTTPException(status_code=401, detail="invalid gateway token")
    now = time.time()
    window = [t for t in _hits.get(token, []) if now - t < 60]
    if len(window) >= RATE_LIMIT_PER_MINUTE:
        raise HTTPException(status_code=429, detail="rate limit exceeded")
    window.append(now)
    _hits[token] = window
    return token


@app.get("/health")
def health():
    return {"ok": True, "claude_enabled": CLAUDE_ENABLED,
            "gemini_configured": bool(GEMINI_API_KEY and GEMINI_IMAGE_MODEL),
            "demo_mode": DEMO_MODE}


@app.post("/v1/run")
def run(req: RunRequest, authorization: str | None = Header(default=None)):
    token = _check_token(authorization)
    task = req.task.lower()
    if DEMO_MODE:
        return _demo_route(token, task, req.prompt, req.size)
    if task == "image":
        return _image_route(token, req.prompt, req.size)
    if task == "code":
        # Claude 預留插槽：客戶而家唔訂閱 Claude，所以停用。
        _log_usage(token, "code", "claude (disabled)", False, "claude_paused")
        raise HTTPException(status_code=503, detail="code route disabled: Claude paused")
    if task == "chat":
        _log_usage(token, "chat", "unassigned", False, "no_model_assigned_yet")
        raise HTTPException(status_code=501, detail="chat route not assigned to a model yet")
    raise HTTPException(status_code=400, detail="unknown task; use image | code | chat")


@app.post("/v1/image")
def image(req: ImageRequest, authorization: str | None = Header(default=None)):
    token = _check_token(authorization)
    if DEMO_MODE:
        return _demo_route(token, "image", req.prompt, req.size)
    return _image_route(token, req.prompt, req.size)


# --- 示範模式：模擬 provider 回應，重點係展示 routing 決定 ---
# 每個回應都帶 simulated=true，唔會扮真結果。
_DEMO_ROUTES = {
    "chat": {
        "provider": "Google Gemini",
        "model": "Gemini（文字模型，正式版動工時對官方文件填 model ID）",
        "reason": "一般文字對話／摘要／改寫，交俾 Gemini 文字模型：速度快、成本低，"
                  "而且同生圖共用同一個 provider 帳戶，管理最簡單。",
    },
    "image": {
        "provider": "Google Gemini",
        "model": "Gemini 生圖模型（正式版動工時對官方文件填 model ID）",
        "reason": "生圖任務交俾 Gemini 生圖模型：呢個 gateway 嘅設計係生圖行 "
                  "Gemini 專線，唔會同文字任務爭同一條路。",
    },
    "code": {
        "provider": "Anthropic Claude（預留插槽）",
        "model": "Claude（未啟用）",
        "reason": "寫程式／改程式嘅任務，設計上會交俾 Claude。不過貴公司目前"
                  "未訂閱 Claude，所以正式版呢條路係暫停嘅（會回 503）；"
                  "示範模式照樣展示呢個 routing 決定，等您睇到成個分流邏輯。",
    },
}


def _demo_route(token: str, task: str, prompt: str, size: str):
    if task not in _DEMO_ROUTES:
        raise HTTPException(status_code=400, detail="unknown task; use image | code | chat")
    route = _DEMO_ROUTES[task]
    preview = prompt if len(prompt) <= 60 else prompt[:60] + "…"
    if task == "chat":
        simulated_reply = (
            f"（模擬回覆）我收到您嘅要求：「{preview}」。"
            "正式版呢度會係 Gemini 文字模型嘅真實回答內容。"
        )
    elif task == "image":
        simulated_reply = (
            f"（模擬回覆）正式版呢度會回傳一張 {size} 圖片，內容係按呢句 prompt "
            f"生成：「{preview}」。示範模式唔會真係生圖，所以唔會產生任何費用。"
        )
    else:  # code
        simulated_reply = (
            f"（模擬回覆）呢個程式任務（「{preview}」）設計上會交俾 Claude 處理；"
            "因為 Claude 路由目前暫停，正式版會回 503，唔會扮做到。"
        )
    _log_usage(token, task, route["model"], True, "demo_simulated")
    return {
        "ok": True,
        "demo_mode": True,
        "simulated": True,
        "task": task,
        "routing": {
            "provider": route["provider"],
            "model": route["model"],
            "reason": route["reason"],
        },
        "response": simulated_reply,
        "note": "呢個係示範模式嘅模擬回應，冇呼叫任何真模型、冇產生費用。"
                "正式版會用同一個 routing 決定，改為呼叫真 provider。",
    }


def _image_route(token: str, prompt: str, size: str):
    if not GEMINI_API_KEY or not GEMINI_IMAGE_MODEL:
        _log_usage(token, "image", "gemini (not configured)", False, "gemini_not_configured")
        # 骨架階段：未接真 Gemini 之前，明確話俾呼叫者知，唔好扮成功。
        return JSONResponse(
            status_code=501,
            content={"ok": False,
                     "error": "Gemini not configured yet — set GEMINI_API_KEY and "
                              "GEMINI_IMAGE_MODEL on the VM, then wire the provider call "
                              "to Google's current SDK/docs."},
        )
    # TODO（動工嗰步）：喺呢度用 GEMINI_API_KEY + GEMINI_IMAGE_MODEL 呼叫
    # Google 官方 Gemini API 生圖，只回傳圖片，唔回傳 key。
    # 未接好之前，唔好扮成功：就算 key 同 model 已設定，都要明確返 501，
    # 用量 log 亦要記 ok=false，否則測試嗰陣會誤以為已經生到圖。
    _log_usage(token, "image", GEMINI_IMAGE_MODEL, False, "gemini_provider_not_wired")
    return JSONResponse(
        status_code=501,
        content={"ok": False,
                 "error": "Gemini provider call not wired yet — key and model are set, "
                          "but the image call to Google's API still needs to be "
                          "implemented at build time (see TODO in app.py)."},
    )
