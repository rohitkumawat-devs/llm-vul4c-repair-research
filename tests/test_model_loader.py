"""Unit tests for ModelLoader and ModelConfig.

All tests operate with mocked model and tokenizer factories to strictly guarantee
that no network requests or model weight downloads occur during test execution.
"""

from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from pipeline.model_loader import (
    ConfigurationError,
    GPUMemoryError,
    ModelConfig,
    ModelLoader,
    ModelLoaderError,
    ModelNotFoundError,
    load_model,
)


class TestModelConfig(unittest.TestCase):
    """Test model configuration data structure and defaults."""

    def test_default_config(self):
        cfg = ModelConfig()
        self.assertEqual(cfg.model_name_or_path, "Qwen/Qwen2.5-Coder-1.5B-Instruct")
        self.assertEqual(cfg.torch_dtype, "float16")
        self.assertEqual(cfg.device, "auto")
        self.assertTrue(cfg.trust_remote_code)
        self.assertTrue(cfg.low_cpu_mem_usage)

    def test_to_and_from_dict(self):
        data = {
            "model_name_or_path": "custom/model-path",
            "device": "cpu",
            "torch_dtype": "float32",
            "extra_irrelevant_field": 123,
        }
        cfg = ModelConfig.from_dict(data)
        self.assertEqual(cfg.model_name_or_path, "custom/model-path")
        self.assertEqual(cfg.device, "cpu")
        self.assertEqual(cfg.torch_dtype, "float32")
        as_dict = cfg.to_dict()
        self.assertNotIn("extra_irrelevant_field", as_dict)
        self.assertEqual(as_dict["device"], "cpu")


class TestModelLoader(unittest.TestCase):
    """Test ModelLoader logic with mocked dependencies."""

    def test_resolve_torch_dtype(self):
        cfg = ModelConfig(torch_dtype="float16")
        loader = ModelLoader(cfg)
        self.assertEqual(loader._resolve_torch_dtype(), torch.float16)

        cfg_bf16 = ModelConfig(torch_dtype="bfloat16")
        loader_bf16 = ModelLoader(cfg_bf16)
        self.assertEqual(loader_bf16._resolve_torch_dtype(), torch.bfloat16)

        cfg_f32 = ModelConfig(torch_dtype="float32")
        loader_f32 = ModelLoader(cfg_f32)
        self.assertEqual(loader_f32._resolve_torch_dtype(), torch.float32)

        cfg_invalid = ModelConfig(torch_dtype="int8")
        loader_invalid = ModelLoader(cfg_invalid)
        with self.assertRaises(ConfigurationError):
            loader_invalid._resolve_torch_dtype()

    def test_resolve_device(self):
        cfg_cpu = ModelConfig(device="cpu")
        loader_cpu = ModelLoader(cfg_cpu)
        self.assertEqual(loader_cpu._resolve_device(), "cpu")

        cfg_invalid = ModelConfig(device="tpu")
        loader_invalid = ModelLoader(cfg_invalid)
        with self.assertRaises(ConfigurationError):
            loader_invalid._resolve_device()

    def test_load_tokenizer_with_pad_token_fallback(self):
        mock_tok_factory = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<|endoftext|>"
        mock_tok_factory.from_pretrained.return_value = mock_tokenizer

        loader = ModelLoader(tokenizer_cls=mock_tok_factory)
        tok = loader.load_tokenizer()

        mock_tok_factory.from_pretrained.assert_called_once()
        self.assertEqual(tok.pad_token, "<|endoftext|>")
        self.assertIs(loader.tokenizer, mock_tokenizer)

    def test_load_tokenizer_model_not_found(self):
        mock_tok_factory = MagicMock()
        mock_tok_factory.from_pretrained.side_effect = Exception("Repository not found")

        loader = ModelLoader(tokenizer_cls=mock_tok_factory)
        with self.assertRaises(ModelNotFoundError):
            loader.load_tokenizer()

    def test_load_model_evaluation_mode_and_dispatch(self):
        mock_tok_factory = MagicMock()
        mock_tok = MagicMock()
        mock_tok.pad_token = "<pad>"
        mock_tok_factory.from_pretrained.return_value = mock_tok

        mock_model_factory = MagicMock()
        mock_model = MagicMock()
        mock_model_factory.from_pretrained.return_value = mock_model

        cfg = ModelConfig(device="cpu", torch_dtype="float32")
        loader = ModelLoader(config=cfg, tokenizer_cls=mock_tok_factory, model_cls=mock_model_factory)
        model, tok = loader.load()

        mock_model_factory.from_pretrained.assert_called_once()
        mock_model.eval.assert_called_once()
        self.assertIs(model, mock_model)
        self.assertIs(tok, mock_tok)

    def test_load_model_cuda_oom_handling(self):
        mock_tok_factory = MagicMock()
        mock_tok_factory.from_pretrained.return_value = MagicMock()

        mock_model_factory = MagicMock()
        mock_model_factory.from_pretrained.side_effect = RuntimeError("CUDA out of memory. Tried to allocate 4.00 GiB")

        cfg = ModelConfig(device="cuda", torch_dtype="float16")
        loader = ModelLoader(config=cfg, tokenizer_cls=mock_tok_factory, model_cls=mock_model_factory)

        with self.assertRaises(GPUMemoryError) as ctx:
            loader.load_model()
        self.assertIn("CUDA Out of Memory", str(ctx.exception))

    def test_cleanup_releases_references(self):
        mock_model = MagicMock()
        mock_tok = MagicMock()
        loader = ModelLoader()
        loader.model = mock_model
        loader.tokenizer = mock_tok

        loader.cleanup()
        self.assertIsNone(loader.model)
        self.assertIsNone(loader.tokenizer)

    def test_get_gpu_memory_info(self):
        info = ModelLoader.get_gpu_memory_info()
        self.assertIn("cuda_available", info)
        if info["cuda_available"]:
            self.assertIn("total_gb", info)
            self.assertIn("device_name", info)


if __name__ == "__main__":
    unittest.main()
