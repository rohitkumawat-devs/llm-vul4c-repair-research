# LLM-Based Vulnerability Detection and Automated Repair for C/C++

**Investigating whether fine-tuning a pretrained coding LLM improves C/C++ vulnerability detection and secure code repair.**

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%20Enabled-ee4c2c?logo=pytorch)
![Model](https://img.shields.io/badge/LLM-Qwen2.5--Coder--1.5B-purple)
![Status](https://img.shields.io/badge/Status-Research%20in%20Progress-orange)

---

## Overview

Software vulnerabilities in C and C++ can lead to memory corruption, crashes, and security issues. This project investigates how large language models (LLMs) can help identify vulnerabilities and generate source-code repairs.

The research uses the **Vul4C benchmark** to compare a pretrained coding model against the same model after fine-tuning.

### Research Question

> Does fine-tuning an existing pretrained coding LLM improve vulnerability detection and repair quality compared with the same model before fine-tuning?

## Objectives

- Build a reproducible C/C++ vulnerability dataset preparation pipeline.
- Establish a baseline using an existing pretrained coding LLM.
- Fine-tune the same model using training data only.
- Prevent data leakage between training, validation, and test sets.
- Evaluate vulnerability detection and automated repair.
- Compare both models using consistent evaluation metrics.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | PyTorch |
| Model Framework | Hugging Face Transformers |
| Pretrained Model | Qwen2.5-Coder-1.5B-Instruct |
| Dataset | Vul4C Benchmark |
| Data Formats | JSON, JSONL |
| Testing | Python `unittest` |
| Environment | Ubuntu 24.04, WSL2 |
| Version Control | Git and GitHub |

## Supported Tasks

### 1. Vulnerability Detection

The model analyzes vulnerable C/C++ source code and produces a structured vulnerability assessment.

**Example output:**

```json
{
  "is_vulnerable": true,
  "cwe": "CWE-20",
  "evidence": "Relevant source-code construct",
  "reasoning": "Technical explanation of the potential vulnerability"
}
```

*Note: This is an illustrative output schema, not an experimental result.*

### 2. Automated Vulnerability Repair

The model receives vulnerable source code and a repair instruction and generates a proposed corrected source file.

The repair is evaluated for:

- Compilation success
- Functional correctness
- Vulnerability remediation
- Potential regressions and new security issues

## Repository Structure

```text
llm-vul4c-repair-research/
├── config/       # Model and experiment configurations
├── cases/        # Case-related resources
├── data/         # Dataset artifacts and manifests
├── notes/        # Research notes and implementation guides
├── pipeline/     # Dataset preparation and inference
├── prompts/      # Prompt templates
├── research/     # Research documentation
├── results/      # Experiment outputs
├── scripts/      # Utility scripts
├── tests/        # Unit tests
├── README.md
└── .gitignore
```

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/rohitkumawat-devs/llm-vul4c-repair-research.git
cd llm-vul4c-repair-research
```

### 2. Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 3. Verify GPU Access

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
nvidia-smi
```

### 4. Run Tests

```bash
python -m unittest discover tests -v
```

## Current Project Status

- [x] Benchmark case discovery and metadata extraction
- [x] Dataset construction pipeline
- [x] Benchmark coverage audit
- [x] Pretrained-model loading and inference runner
- [x] Initial baseline inference
- [ ] Validate structured detection output
- [ ] Finalize dataset splits and integrity checks
- [ ] Complete baseline evaluation
- [ ] Implement fine-tuning
- [ ] Compare baseline and fine-tuned models

## Author

**Rohit Kumawat**  
Computer Science and Engineering  
BMS College of Engineering, Bengaluru

GitHub: [rohitkumawat-devs](https://github.com/rohitkumawat-devs)

## Disclaimer

This is an ongoing research project. No improvement in vulnerability detection accuracy, repair success, or security performance is claimed until the corresponding experiments have been completed and evaluated.

## Important

This project is experimental. Compilation success alone does not
establish that a generated patch is secure.
