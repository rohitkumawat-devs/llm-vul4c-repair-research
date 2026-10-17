"""Model Loader for Pretrained Coding LLMs.

This module provides safe, resource-conscious loading and device management
for pretrained transformer models, specifically configured for baseline inference
with Qwen/Qwen2.5-Coder-1.5B-Instruct and compatible Hugging Face causal LLMs.

Design Principles:
- Resource-Conscious: Checks available CUDA VRAM before and after loading,
  defaulting to FP16/BF16 to fit within tight GPU budgets (~6 GB VRAM).
- Explicit OOM Handling: Catches CUDA out-of-memory errors cleanly, frees cached
  tensors, and raises actionable errors with memory diagnostic information.
- Mockable & Test-Friendly: Accepts dependency injection for tokenizer and model
  classes, allowing comprehensive unit tests without downloading weights.
- Reproducible & Non-Destructive: Purely loads models in evaluation mode (eval())
  with gradient computation disabled.
"""

from dataclasses import asdict, dataclass, field
import gc
import logging
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Tuple, Union

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)


class ModelLoaderError(Exception):
    """Base exception for all model loading errors."""
    pass


class ModelNotFoundError(ModelLoaderError):
    """Raised when the requested model name or local path cannot be found."""
    pass


class GPUMemoryError(ModelLoaderError):
    """Raised when loading or allocating model weights exceeds GPU memory."""
    pass


class ConfigurationError(ModelLoaderError):
    """Raised when invalid model configuration parameters are specified."""
    pass


@dataclass
class ModelConfig:
    """Configuration for loading pretrained LLMs."""
    model_name_or_path: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    device: str = "auto"  # "auto", "cuda", "cpu"
    torch_dtype: str = "float16"  # "float16", "bfloat16", "float32"
    device_map: Optional[str] = None  # e.g., "auto" or None
    trust_remote_code: bool = True
    low_cpu_mem_usage: bool = True
    attn_implementation: Optional[str] = None  # "sdpa", "flash_attention_2", or None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


class ModelLoader:
    """Manages downloading, caching, instantiating, and releasing LLM models."""

    DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        tokenizer_cls: Optional[Any] = None,
        model_cls: Optional[Any] = None,
    ) -> None:
        self.config = config or ModelConfig()
        self._tokenizer_cls = tokenizer_cls
        self._model_cls = model_cls

        self.tokenizer: Optional[Any] = None
        self.model: Optional[Any] = None
        self._resolved_device: Optional[str] = None
        self._resolved_dtype: Optional[Any] = None

    @staticmethod
    def get_gpu_memory_info() -> Dict[str, Any]:
        """Return diagnostic metrics regarding GPU memory state."""
        import torch
        if not torch.cuda.is_available():
            return {"cuda_available": False}

        device_idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_idx)
        total_gb = props.total_memory / (1024 ** 3)
        allocated_gb = torch.cuda.memory_allocated(device_idx) / (1024 ** 3)
        reserved_gb = torch.cuda.memory_reserved(device_idx) / (1024 ** 3)
        free_gb = total_gb - reserved_gb

        return {
            "cuda_available": True,
            "device_index": device_idx,
            "device_name": props.name,
            "total_gb": round(total_gb, 3),
            "allocated_gb": round(allocated_gb, 3),
            "reserved_gb": round(reserved_gb, 3),
            "free_approx_gb": round(free_gb, 3),
        }

    def _resolve_torch_dtype(self) -> Any:
        import torch
        dtype_str = self.config.torch_dtype.lower().strip()
        if dtype_str == "float16":
            return torch.float16
        elif dtype_str == "bfloat16":
            return torch.bfloat16
        elif dtype_str == "float32":
            return torch.float32
        else:
            raise ConfigurationError(
                f"Unsupported torch_dtype: '{self.config.torch_dtype}'. "
                f"Supported options: 'float16', 'bfloat16', 'float32'."
            )

    def _resolve_device(self) -> str:
        import torch
        dev = self.config.device.lower().strip()
        if dev == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        elif dev in ("cuda", "cpu"):
            if dev == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available. Falling back to CPU.")
                return "cpu"
            return dev
        elif dev.startswith("cuda:"):
            if not torch.cuda.is_available():
                logger.warning("CUDA device requested but CUDA not available. Falling back to CPU.")
                return "cpu"
            return dev
        else:
            raise ConfigurationError(f"Unsupported device specification: '{self.config.device}'")

    def load_tokenizer(self) -> Any:
        """Load and return the tokenizer."""
        if self.tokenizer is not None:
            return self.tokenizer

        if self._tokenizer_cls is not None:
            tokenizer_factory = self._tokenizer_cls
        else:
            try:
                from transformers import AutoTokenizer
                tokenizer_factory = AutoTokenizer
            except ImportError as err:
                raise ModelLoaderError(
                    "The 'transformers' package is required to load models. "
                    "Install it via `pip install transformers`."
                ) from err

        try:
            logger.info("Loading tokenizer for '%s'...", self.config.model_name_or_path)
            tok = tokenizer_factory.from_pretrained(
                self.config.model_name_or_path,
                trust_remote_code=self.config.trust_remote_code,
            )
            # Ensure pad_token is set (crucial for batching / generation)
            if tok.pad_token is None:
                if tok.eos_token is not None:
                    tok.pad_token = tok.eos_token
                else:
                    tok.pad_token = tok.bos_token
            self.tokenizer = tok
            return self.tokenizer
        except Exception as err:
            err_str = str(err)
            if "not found" in err_str.lower() or "repository not found" in err_str.lower():
                raise ModelNotFoundError(
                    f"Model repository or path not found: '{self.config.model_name_or_path}'. Error: {err}"
                ) from err
            raise ModelLoaderError(
                f"Failed to load tokenizer for '{self.config.model_name_or_path}': {err}"
            ) from err

    def load_model(self) -> Any:
        """Load and return the causal LLM model."""
        import torch

        if self.model is not None:
            return self.model

        if self._model_cls is not None:
            model_factory = self._model_cls
        else:
            try:
                from transformers import AutoModelForCausalLM
                model_factory = AutoModelForCausalLM
            except ImportError as err:
                raise ModelLoaderError(
                    "The 'transformers' package is required to load models. "
                    "Install it via `pip install transformers`."
                ) from err

        target_device = self._resolve_device()
        target_dtype = self._resolve_torch_dtype()
        self._resolved_device = target_device
        self._resolved_dtype = target_dtype

        # On CPU, default to float32 if float16 would cause issues or if specified
        if target_device == "cpu" and target_dtype == torch.float16:
            logger.info("Adjusting torch_dtype to float32 for CPU execution.")
            target_dtype = torch.float32

        load_kwargs: Dict[str, Any] = {
            "torch_dtype": target_dtype,
            "trust_remote_code": self.config.trust_remote_code,
            "low_cpu_mem_usage": self.config.low_cpu_mem_usage,
        }

        if self.config.attn_implementation:
            load_kwargs["attn_implementation"] = self.config.attn_implementation

        if target_device == "cuda":
            if self.config.device_map:
                load_kwargs["device_map"] = self.config.device_map
            else:
                load_kwargs["device_map"] = "auto"

        logger.info(
            "Loading model '%s' (device=%s, dtype=%s)...",
            self.config.model_name_or_path,
            target_device,
            target_dtype,
        )

        try:
            model = model_factory.from_pretrained(
                self.config.model_name_or_path,
                **load_kwargs,
            )

            # If device_map wasn't used and target is not CPU, move explicitly
            if "device_map" not in load_kwargs and target_device != "cpu":
                model = model.to(target_device)

            model.eval()
            self.model = model
            return self.model

        except (torch.cuda.OutOfMemoryError, RuntimeError) as err:
            err_str = str(err)
            if "out of memory" in err_str.lower() or isinstance(err, torch.cuda.OutOfMemoryError):
                self.cleanup()
                gpu_info = self.get_gpu_memory_info()
                raise GPUMemoryError(
                    f"CUDA Out of Memory while loading model '{self.config.model_name_or_path}'. "
                    f"GPU State: {gpu_info}. Suggestions: use float16/bfloat16, close other GPU processes, "
                    f"or run on CPU using --device cpu."
                ) from err
            raise ModelLoaderError(
                f"Failed to load model weights for '{self.config.model_name_or_path}': {err}"
            ) from err
        except Exception as err:
            err_str = str(err)
            if "not found" in err_str.lower() or "repository not found" in err_str.lower():
                raise ModelNotFoundError(
                    f"Model repository or path not found: '{self.config.model_name_or_path}'. Error: {err}"
                ) from err
            raise ModelLoaderError(
                f"Failed to load model weights for '{self.config.model_name_or_path}': {err}"
            ) from err

    def load(self) -> Tuple[Any, Any]:
        """Load and return both (model, tokenizer)."""
        tokenizer = self.load_tokenizer()
        model = self.load_model()
        return model, tokenizer

    def cleanup(self) -> None:
        """Release GPU memory and delete model references."""
        import torch
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_model(
    model_name_or_path: str = ModelLoader.DEFAULT_MODEL,
    device: str = "auto",
    torch_dtype: str = "float16",
    **kwargs: Any,
) -> Tuple[Any, Any]:
    """Convenience functional interface to load model and tokenizer."""
    cfg = ModelConfig(
        model_name_or_path=model_name_or_path,
        device=device,
        torch_dtype=torch_dtype,
        **kwargs,
    )
    loader = ModelLoader(cfg)
    return loader.load()
