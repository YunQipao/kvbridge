"""Shared model loading and runtime diagnostics.

This module centralizes device and dtype policy so integration scripts do not
silently move sharded models with ``model.to(...)``.  It deliberately does not
perform automatic CPU fallback when CUDA was requested.
"""

import os
import platform
import sys
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from huggingface_hub.utils import LocalEntryNotFoundError
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from .geometry import text_config


def unwrap_text_model(model):
    """Reuse the official composite's decoder and LM head without copying weights.

    Transformers Mistral3ForConditionalGeneration owns model.language_model
    (a bare decoder) and lm_head; it is not itself a causal language model.
    """
    if getattr(model.config, "model_type", None) != "mistral3":
        return model
    from transformers import AutoModelForCausalLM
    decoder = getattr(getattr(model, "model", None), "language_model", None)
    head = model.get_output_embeddings()
    if decoder is None or not hasattr(decoder, "layers") or head is None:
        raise ValueError("unsupported Mistral3 composite text model structure")
    # Meta construction allocates no real parameter storage; all text weights
    # are replaced with the original module objects.
    with torch.device("meta"):
        causal = AutoModelForCausalLM.from_config(text_config(model.config))
    causal.model = decoder
    causal.set_output_embeddings(head)
    return causal.eval()


def ministral_load_kwargs(model_id, config):
    """Load only text tensors from the verified local HF state-dict layout."""
    if getattr(config, "model_type", None) != "mistral3":
        return {}
    path = Path(model_id).expanduser()
    index = path / "model.safetensors.index.json"
    if index.exists():
        keys = list(json.loads(index.read_text())["weight_map"])
    elif (path / "model.safetensors").exists():
        from safetensors import safe_open
        with safe_open(str(path / "model.safetensors"), framework="pt", device="cpu") as f:
            keys = list(f.keys())
    else:
        raise FileNotFoundError("Ministral text-only loading requires local HF safetensors weights/index")
    if "model.language_model.embed_tokens.weight" in keys:
        mapping = {r"^model\.language_model\.": "model."}
    elif "model.embed_tokens.weight" in keys:
        mapping = {}
    else:
        raise ValueError("unrecognized Ministral checkpoint text state_dict namespace")
    return {"config": text_config(config), "key_mapping": mapping,
            "output_loading_info": True}


def print_runtime_diagnostics():
    try:
        import transformers
        transformers_version = transformers.__version__
    except Exception:
        transformers_version = "unavailable"
    try:
        import safetensors
        safetensors_version = safetensors.__version__
    except Exception:
        safetensors_version = "unavailable"
    values = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "transformers": transformers_version,
        "safetensors": safetensors_version,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    for key, value in values.items():
        print(f"{key}={value}")
    return values


def resolve_dtype(name, device=None):
    if isinstance(name, torch.dtype):
        return name
    name = (name or "auto").lower()
    if name == "auto":
        if device and str(device).startswith("cuda") and torch.cuda.is_available():
            return torch.bfloat16
        return torch.float32
    values = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in values:
        raise ValueError(f"unsupported dtype: {name}")
    dtype = values[name]
    if device and str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA dtype requested but torch.cuda.is_available() is False")
    if dtype == torch.bfloat16 and device and str(device).startswith("cuda"):
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("bfloat16 requested but CUDA device does not support it")
    return dtype


def require_device(device):
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")


def load_config(model_id, revision=None, local_files_only=False):
    path = Path(model_id).expanduser()
    if path.is_dir():
        model_id = str(path)
    return AutoConfig.from_pretrained(
        model_id, revision=revision, local_files_only=local_files_only)


def local_tokenizer_reference(model_id, revision=None, cache_dir=None):
    """Resolve a local snapshot without allowing a Hub request.

    Transformers 4.57.6 may call model_info from its Mistral regex check
    even with local_files_only=True. Passing a real directory avoids that
    remote-model branch without disabling or altering tokenizer behavior.
    """
    path = Path(model_id).expanduser()
    if path.is_dir():
        return str(path)
    if path.is_absolute() or str(model_id).startswith(("./", "../", "~")):
        raise FileNotFoundError(f"local tokenizer directory does not exist: {path}")
    try:
        return snapshot_download(str(model_id), revision=revision,
                                 cache_dir=cache_dir, local_files_only=True)
    except LocalEntryNotFoundError as exc:
        raise FileNotFoundError(
            f"no cached tokenizer snapshot for {model_id!r}, revision={revision!r}; "
            "download it on a network-enabled machine or use a local model directory"
        ) from exc


def load_tokenizer(model_id, revision=None, local_files_only=False, **kwargs):
    from huggingface_hub.constants import HF_HUB_OFFLINE

    offline = (local_files_only or HF_HUB_OFFLINE
               or os.environ.get("HF_HUB_OFFLINE", "").upper() in ("1", "ON", "YES", "TRUE"))
    if offline:
        model_id = local_tokenizer_reference(model_id, revision,
                                             cache_dir=kwargs.get("cache_dir"))
    try:
        return AutoTokenizer.from_pretrained(
            model_id, revision=revision, local_files_only=offline, **kwargs)
    except (ValueError, ImportError):
        # New official Tekken checkpoints may require the native backend.
        # This backend already implements the HF tokenizer API used here.
        path = Path(model_id).expanduser()
        if not (path / "tekken.json").is_file():
            raise
        try:
            from transformers import MistralCommonBackend
            return MistralCommonBackend.from_pretrained(
                str(path), local_files_only=True, **kwargs)
        except ImportError as backend_exc:
            raise ImportError("local Ministral tokenizer requires an existing mistral-common installation") from backend_exc


def assert_text_tokenizer_alignment(source_tokenizer, target_tokenizer, text):
    source = source_tokenizer(text, add_special_tokens=True).input_ids
    target = target_tokenizer(text, add_special_tokens=True).input_ids
    if list(source) != list(target):
        raise ValueError("source and target tokenizer IDs differ for evaluation prefix")


def load_causal_lm(model_id, *, revision=None, device="cpu",
                   device_map=None, dtype="auto", local_files_only=False,
                   **kwargs):
    # When device_map is supplied, ``device`` is only a diagnostic/default
    # value; Accelerate owns placement and model.to(device) is skipped.
    if isinstance(device_map, str) and device_map.lower() == "none":
        device_map = None
    if device_map is None:
        require_device(device)
    # In sharded auto mode let HF preserve the checkpoint dtype. Resolving
    # auto without a device would silently expand BF16 weights to FP32.
    resolved = ("auto" if device_map is not None and dtype == "auto" else
                resolve_dtype(dtype, device if device_map is None else None))
    model_kwargs = dict(
        revision=revision,
        local_files_only=local_files_only,
        dtype=resolved,
        **kwargs,
    )
    if device_map is not None:
        model_kwargs["device_map"] = device_map
    # Inspect only local config here; keep the established Qwen loading path.
    local_path = Path(model_id).expanduser()
    config_file = local_path / "config.json"
    if config_file.is_file() and json.loads(config_file.read_text()).get("model_type") in ("mistral3", "ministral3"):
        config = load_config(str(local_path), revision=revision, local_files_only=True)
        model_kwargs.update(ministral_load_kwargs(str(local_path), config))
        model_id = str(local_path)
        model_kwargs["local_files_only"] = True
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    except TypeError as exc:
        # Compatibility with Transformers versions predating ``dtype=``.
        if "dtype" not in str(exc):
            raise
        model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    if isinstance(model, tuple):
        model, info = model
        if info.get("missing_keys") or info.get("mismatched_keys"):
            raise RuntimeError(f"incomplete Ministral text weight load: {info}")
    model = unwrap_text_model(model)
    if device_map is None:
        model = model.to(device)
    return model.eval()


def _execution_device(module):
    """Account for Accelerate CPU/disk-offload hooks as well as resident weights."""
    hook = getattr(module, "_hf_hook", None)
    value = getattr(hook, "execution_device", None)
    if value is None:
        for child in getattr(hook, "hooks", ()):
            value = getattr(child, "execution_device", None)
            if value is not None:
                break
    if value is not None:
        return torch.device(f"cuda:{value}" if isinstance(value, int) else value)
    weight = getattr(module, "weight", None)
    device = weight.device if weight is not None else next(module.parameters()).device
    if device.type == "meta":
        raise RuntimeError("cannot resolve execution device for an offloaded meta module")
    return device


def get_model_input_device(model):
    """Place tokens beside input embeddings, not the first hf_device_map entry."""
    getter = getattr(model, "get_input_embeddings", None)
    if getter is not None:
        return _execution_device(getter())
    # Keep lightweight legacy toy models (which lack the HF API) working.
    return _execution_device(model)


def model_input_device(model):
    """Backward-compatible name used by fitting and existing callers."""
    return get_model_input_device(model)


def get_model_layer_devices(model):
    """Return attention execution devices in cache layer order (Qwen/Ministral)."""
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise ValueError("cannot resolve target decoder layers for KV cache placement")
    return [_execution_device(layer.self_attn.q_proj) for layer in layers]
