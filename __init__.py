"""
ComfyUI node: Seedream 4.X Edit Sequential (Atlas Cloud)
1 reference image + prompt in  ->  X generated images out (IMAGE batch)

Install: drop this folder in ComfyUI/custom_nodes/ and restart.
API key: set ATLASCLOUD_API_KEY env var, or paste it in the node's api_key field.
"""

import os
import io
import time
import base64

import numpy as np
import requests
import torch
from PIL import Image

try:
    import comfy.model_management as mm
except ImportError:  # running outside ComfyUI
    mm = None


API_BASE = "https://api.atlascloud.ai/api/v1/model"
SUBMIT_URL = f"{API_BASE}/generateImage"
POLL_URL = API_BASE + "/prediction/{}"

VERSIONS = ["4.7", "4.5", "4"]

SIZES = [
    # 2K
    "2048*2048", "2304*1728", "1728*2304", "2848*1600", "1600*2848",
    "2496*1664", "1664*2496", "3136*1344",
    # 4K
    "4096*4096", "4704*3520", "3520*4704", "5504*3040", "3040*5504",
    "4992*3328", "3328*4992", "6240*2656",
    # 1K
    "1024*1024", "1280*720", "720*1280", "1248*832", "832*1248", "1568*672",
    # manual
    "custom",
]

MIN_PIXELS = 921_600
MAX_PIXELS = 16_777_216
DONE_STATES = {"completed", "succeeded", "success"}
FAIL_STATES = {"failed", "error", "canceled", "cancelled"}


# ---------- helpers ----------

def _check_interrupt():
    if mm is not None:
        mm.throw_exception_if_processing_interrupted()


def _tensor_to_data_uri(img: torch.Tensor, fmt: str) -> str:
    """img: [H, W, C] float 0..1 -> data URI"""
    arr = (img.detach().cpu().numpy().clip(0, 1) * 255).round().astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    pil = Image.fromarray(arr[..., :3], "RGB")
    buf = io.BytesIO()
    if fmt == "jpeg":
        pil.save(buf, format="JPEG", quality=95)
    else:
        pil.save(buf, format="PNG", compress_level=4)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{fmt};base64,{b64}"


def _pil_to_tensor(pil: Image.Image) -> torch.Tensor:
    arr = np.asarray(pil.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr)  # [H, W, 3]


def _load_output(item: str, session: requests.Session) -> Image.Image:
    if item.startswith("http://") or item.startswith("https://"):
        r = session.get(item, timeout=120)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content))
    # base64 (with or without data: prefix)
    if item.startswith("data:"):
        item = item.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(item)))


def _unwrap(j: dict) -> dict:
    """Atlas wraps responses as {code, message, data}. Handle both wrapped and flat."""
    if isinstance(j, dict) and "code" in j and j.get("code") not in (200, 0, None):
        raise RuntimeError(f"Atlas API error {j.get('code')}: {j.get('message') or j}")
    if isinstance(j, dict) and isinstance(j.get("data"), dict):
        return j["data"]
    return j


def _resolve_size(size, w, h):
    if size != "custom":
        return size
    px = w * h
    if not (MIN_PIXELS <= px <= MAX_PIXELS):
        raise ValueError(
            f"Custom size {w}x{h} = {px} px, must be between {MIN_PIXELS} and {MAX_PIXELS}"
        )
    ratio = w / h
    if not (1 / 16 <= ratio <= 16):
        raise ValueError(f"Custom aspect ratio {w}:{h} out of range (1:16 to 16:1)")
    return f"{w}*{h}"


# ---------- node ----------

class SeedreamEditSequentialAtlas:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "image": ("IMAGE",),
                "model_version": (VERSIONS, {"default": "4.7"}),
                "size": (SIZES, {"default": "2048*2048"}),
                "num_images": ("INT", {"default": 4, "min": 1, "max": 14}),
                "prompt_expansion_mode": (["standard", "fast"], {"default": "standard"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "custom_width": ("INT", {"default": 2048, "min": 256, "max": 8192, "step": 8}),
                "custom_height": ("INT", {"default": 2048, "min": 256, "max": 8192, "step": 8}),
                "base64_output": ("BOOLEAN", {"default": False}),
                "ref_format": (["jpeg", "png"], {"default": "jpeg"}),
                "poll_interval": ("FLOAT", {"default": 3.0, "min": 0.5, "max": 30.0, "step": 0.5}),
                "timeout_sec": ("INT", {"default": 600, "min": 30, "max": 3600}),
                "model_override": ("STRING", {"default": ""}),
                # retried once with this version if the main one errors
                "fallback_version": (["none"] + VERSIONS, {"default": "4.5"}),
                # not sent to the API, only here so you can force a re-run
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("images", "urls", "count")
    FUNCTION = "generate"
    CATEGORY = "ACG/Seedream"

    def generate(
        self,
        prompt,
        image,
        model_version,
        size,
        num_images,
        prompt_expansion_mode,
        api_key="",
        custom_width=2048,
        custom_height=2048,
        base64_output=False,
        ref_format="jpeg",
        poll_interval=3.0,
        timeout_sec=600,
        model_override="",
        fallback_version="4.5",
        seed=0,
    ):
        key = (api_key or "").strip() or os.environ.get("ATLASCLOUD_API_KEY", "")
        if not key:
            raise ValueError("No API key. Set ATLASCLOUD_API_KEY or fill in api_key.")
        if not prompt.strip():
            raise ValueError("Prompt is empty.")

        override = model_override.strip()
        models = [override or f"bytedance/seedream-v{model_version}/edit-sequential"]
        if not override and fallback_version not in ("none", model_version):
            models.append(f"bytedance/seedream-v{fallback_version}/edit-sequential")

        if image.shape[0] > 1:
            print(f"[Seedream] Got a batch of {image.shape[0]}, using only the first as reference.")
        ref_uri = _tensor_to_data_uri(image[0], ref_format)

        payload = {
            "prompt": prompt,
            "images": [ref_uri],
            "size": _resolve_size(size, custom_width, custom_height),
            "num_images": int(min(num_images, 14)),  # 1 ref + 14 = 15 cap
            "prompt_expansion_mode": prompt_expansion_mode,
            "enable_base64_output": bool(base64_output),
        }

        session = requests.Session()
        session.headers.update({
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        })

        for i, model in enumerate(models):
            try:
                outputs = self._run(session, dict(payload, model=model), poll_interval, timeout_sec)
                break
            except (RuntimeError, requests.RequestException) as e:
                if i == len(models) - 1:
                    raise
                print(f"[Seedream] {model} failed ({e}), falling back to {models[i + 1]}")

        # download + convert
        pils = [_load_output(o, session) for o in outputs]
        tw, th = pils[0].size
        tensors = []
        for p in pils:
            if p.size != (tw, th):
                print(f"[Seedream] Resizing {p.size} -> {(tw, th)} to fit batch")
                p = p.convert("RGB").resize((tw, th), Image.LANCZOS)
            tensors.append(_pil_to_tensor(p))

        batch = torch.stack(tensors, dim=0)  # [N, H, W, 3]
        url_str = "\n".join(o for o in outputs if o.startswith("http"))
        return (batch, url_str, len(tensors))

    def _run(self, session, payload, poll_interval, timeout_sec):
        """Submit one prediction and poll until it finishes. Returns the outputs list."""
        model = payload["model"]
        num_images = payload["num_images"]

        # submit
        print(f"[Seedream] Submitting {model} | {payload['size']} | x{payload['num_images']}")
        r = session.post(SUBMIT_URL, json=payload, timeout=120)
        if r.status_code >= 400:
            raise RuntimeError(f"Submit failed ({r.status_code}): {r.text[:500]}")
        data = _unwrap(r.json())
        pred_id = data.get("id")
        if not pred_id:
            raise RuntimeError(f"No prediction id in response: {data}")
        print(f"[Seedream] Prediction id: {pred_id}")

        # poll
        start = time.time()
        outputs = []
        while True:
            _check_interrupt()
            if time.time() - start > timeout_sec:
                raise TimeoutError(f"Seedream prediction {pred_id} timed out after {timeout_sec}s")

            pr = session.get(POLL_URL.format(pred_id), timeout=60)
            if pr.status_code >= 500:
                # transient, keep trying
                print(f"[Seedream] Poll {pr.status_code}, retrying")
            else:
                if pr.status_code >= 400:
                    raise RuntimeError(f"Poll failed ({pr.status_code}): {pr.text[:500]}")
                pdata = _unwrap(pr.json())
                status = str(pdata.get("status", "")).lower()

                if status in DONE_STATES:
                    outputs = pdata.get("outputs") or []
                    break
                if status in FAIL_STATES:
                    err = pdata.get("error") or pdata.get("message") or pdata
                    raise RuntimeError(f"Seedream prediction failed: {err}")

            # sleep in small steps so Cancel in ComfyUI works fast
            end = time.time() + poll_interval
            while time.time() < end:
                _check_interrupt()
                time.sleep(0.25)

        if not outputs:
            raise RuntimeError("Prediction completed but returned no images (moderation?).")

        print(f"[Seedream] Done in {time.time() - start:.1f}s, got {len(outputs)}/{num_images} images")
        return outputs


NODE_CLASS_MAPPINGS = {
    "SeedreamEditSequentialAtlas": SeedreamEditSequentialAtlas,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeedreamEditSequentialAtlas": "Seedream 4.X Edit Sequential (Atlas)",
}
