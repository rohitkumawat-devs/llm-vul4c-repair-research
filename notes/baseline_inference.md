# Baseline Inference: Architecture & Execution Guide

## 1. Overview & Research Objective

In this stage, zero-shot / pretrained baseline inference is established for C/C++ vulnerability detection and repair using the pretrained `Qwen/Qwen2.5-Coder-1.5B-Instruct` model before any fine-tuning takes place.

This baseline provides the empirical control for answering the core research question:
> *How does an existing pretrained coding LLM perform on vulnerability detection and repair compared against the same model after fine-tuning on Vul4C benchmark cases?*

---

## 2. Architecture & Modules

```text
Dataset (JSONL or Case Query)
           ?
           ?
??????????????????????????
? pipeline/inference.py  ? ???? Configuration (config/baseline_inference.json)
??????????????????????????
           ?
           ??? pipeline/model_loader.py (Loads Qwen2.5-Coder-1.5B-Instruct in FP16/BF16)
           ?
           ??? Prompt Formatter (ChatML / apply_chat_template without target leakage)
           ?
           ??? Token & Memory Bounding (max_input_tokens=8192, max_new_tokens=2048)
           ?
           ?
Results & Run Audit (results/*_predictions.jsonl + results/*_metadata.json)
```

### Module 1: `pipeline/model_loader.py`
- **Class `ModelConfig`**: Defines precision (`float16`), device (`auto`/`cuda`/`cpu`), device mapping (`auto`), and tokenizer/model parameters.
- **Class `ModelLoader`**:
  - Validates and inspects CUDA device and VRAM status.
  - Loads `AutoTokenizer` and sets `pad_token_id`.
  - Loads `AutoModelForCausalLM` in evaluation mode (`eval()`) with gradient computation disabled.
  - Catches `torch.cuda.OutOfMemoryError` and translates it to `GPUMemoryError` with diagnostic memory stats and tuning recommendations.
  - Implements clean resource deallocation (`cleanup()`, `torch.cuda.empty_cache()`, `gc.collect()`).

### Module 2: `pipeline/inference.py`
- **Class `BaselineInferenceRunner`**:
  - Reads input samples in Vul4C SFT JSONL format (`messages`, `metadata`).
  - **Zero Data Leakage Guarantee**: Extracts `system` and `user` messages for the model prompt. Strips the `assistant` message entirely from model input; stores it strictly as `ground_truth` for downstream evaluation.
  - Applies model chat templates (`<|im_start|>system...<|im_end|><|im_start|>user...<|im_end|><|im_start|>assistant\n`).
  - Bounds context input tokens (`--max-input-tokens`, default 8192) and generated output tokens (`--max-new-tokens`, default 2048).
  - Handles CUDA OOM on per-sample basis without crashing full batch runs.
  - Streams prediction records to Git-ignored JSONL files line-by-line.
  - Writes comprehensive run audit metadata (duration, sample counts, success/failure rates, OOM counts, GPU state).

---

## 3. GPU VRAM & Resource Management (RTX 4050 ~6 GB VRAM)

1. **Precision**: `float16` weights for `Qwen2.5-Coder-1.5B-Instruct` require approximately 3.0 GB VRAM, fitting comfortably within the 6 GB physical ceiling while leaving ~3 GB for KV cache and activation memory.
2. **Context Window Bounding**: Input context is bounded via `--max-input-tokens` (default 8192). Large benchmark files are safely truncated with warning logging rather than exceeding VRAM boundaries.
3. **Generation Token Limit**: Output generation is bounded via `--max-new-tokens` (default 2048).
4. **OOM Isolation**: In the event of an OOM, CUDA cache is cleared immediately (`torch.cuda.empty_cache()`), the failure is logged with diagnostic details, and the runner proceeds to subsequent samples.

---

## 4. Verification & Testing

The test suite contains 49 comprehensive unit tests across the project:
- `tests/test_case_manager.py`: 12 tests (case discovery, pairing, path safety)
- `tests/test_dataset_builder.py`: 13 tests (SFT dataset building, zero leakage, splitting)
- `tests/test_model_loader.py`: 10 tests (mocked loading, dtype resolution, device handling, OOM handling)
- `tests/test_inference.py`: 14 tests (leakage prevention, ChatML formatting, dry-run simulation, mocked inference, OOM resilience, streaming outputs, CLI)

All unit tests operate on synthetic fixtures and mock model factories without making network calls or downloading model weights.

To run the full test suite:
```bash
python3 -m unittest discover tests -v
```

---

## 5. Execution Commands

### A. Dry-Run Commands (No weight download, no GPU allocation)

1. Dry-run using existing generated dataset file:
```bash
python3 pipeline/inference.py --input-file data/generated/train.jsonl --dry-run
```

2. Dry-run on-the-fly for a specific Vul4C case:
```bash
python3 pipeline/inference.py --case audiofile/CVE-2017-6832 --dry-run
```

3. Dry-run for detection task only:
```bash
python3 pipeline/inference.py --input-file data/generated/train.jsonl --tasks detection --dry-run
```

4. Dry-run for repair task only:
```bash
python3 pipeline/inference.py --input-file data/generated/train.jsonl --tasks repair --dry-run
```

### B. Real Baseline Inference Command (Prepared for execution)

To run real baseline inference on the pilot dataset using `Qwen/Qwen2.5-Coder-1.5B-Instruct` on GPU:
```bash
python3 pipeline/inference.py \
  --model-name Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --input-file data/generated/train.jsonl \
  --device auto \
  --torch-dtype float16 \
  --max-input-tokens 8192 \
  --max-new-tokens 2048 \
  --temperature 0.0 \
  --output-dir results
```

Or using the pre-configured settings file:
```bash
python3 pipeline/inference.py \
  --config config/baseline_inference.json \
  --input-file data/generated/train.jsonl
```

Outputs are automatically written to Git-ignored files in `results/`:
- `results/baseline_Qwen_Qwen2.5-Coder-1.5B-Instruct_<timestamp>_predictions.jsonl`
- `results/baseline_Qwen_Qwen2.5-Coder-1.5B-Instruct_<timestamp>_metadata.json`


---

## 6. Root Cause Analysis: Source Code Continuation & Execution Time

### Issue 1: Model Continued Source Code (`WAVE_FORMAT_...`) Instead of Generating Assessment
- **Symptom**: In the initial 1024-token run on `audiofile_CVE-2017-6832_WAVE.cpp_detection`, the model generated C++ enum constants (`'s TrueSpeech */\n\tWAVE_FORMAT_INTERSPEECH = 0x0023...`) rather than a JSON assessment.
- **Root Cause**: The full prompt for `WAVE.cpp` spans ~13,033 tokens. Naive right-truncation at 1024 tokens cut off the prompt in the middle of the C++ file, completely severing the closing code fence, task instructions, `<|im_end|>`, and `<|im_start|>assistant\n`. The model was never placed into assistant generation mode and naturally continued the incomplete C++ code it was given.
- **Solution (Deliberate Truncation)**: In `build_bounded_prompt()`, token overhead is calculated for system instructions, untrusted input delimiters, and task-specific JSON formatting instructions. Truncation is applied *exclusively* to the source code payload inside `[BEGIN UNTRUSTED SOURCE CODE] ... [END UNTRUSTED SOURCE CODE]`. This guarantees that task instructions, JSON schema, and `<|im_start|>assistant\n` are 100% preserved at the end of the input sequence.

### Issue 2: Long First-Run Duration (~654 Seconds)
- **Verified Breakdown**:
  - Total duration recorded in run metadata: **653.981 seconds** (~10.9 minutes).
  - Actual GPU generation latency recorded in prediction record: **7.905 seconds** (for 256 tokens).
  - Elapsed time prior to generation: **646.076 seconds** (~10.7 minutes).
- **Verified Cause**: During the first real run, `ModelLoader.load()` triggered the initial download of `Qwen/Qwen2.5-Coder-1.5B-Instruct` model weights (~2.9 GB safetensors + tokenizer blobs) from Hugging Face Hub over the network into `~/.cache/huggingface/hub/`.
- **Subsequent Runs**: Because the weights are now permanently cached locally on disk in `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-1.5B-Instruct/`, subsequent runs load from disk in ~2?3 seconds without network delay.

### Updated Vulnerability Detection JSON Schema
The detection prompt explicitly requests a valid JSON object with uncertainty support:
```json
{
  "is_vulnerable": true,  // boolean: false if benign or evidence is insufficient
  "cwe": "CWE-20",        // string or null: specific CWE or null if not vulnerable
  "evidence": "...",      // string or null: concise citation of vulnerable line/construct
  "reasoning": "..."      // string: concise technical rationale
}
```
