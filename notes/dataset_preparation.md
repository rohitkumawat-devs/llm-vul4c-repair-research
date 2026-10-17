# Dataset Preparation: Case Manager & Benchmark Inspection Notes

## 1. Overview and Purpose

This module establishes the first stage in the experimental research pipeline:
```text
Vul4C Benchmark -> Case Manager -> Prompt Generator -> LLM Client -> Patch Manager -> Compilation -> Functional Testing -> Security Validation -> Results
```

The `CaseManager` (`pipeline/case_manager.py`) provides safe, deterministic inspection, validation, and metadata extraction for vulnerability cases in the Vul4C benchmark.

---

## 2. Benchmark Inspection Findings

### Initial Focus Case: `audiofile/CVE-2017-6832`

- **Affected Software**: Audio File Library (`audiofile`), version 0.3.6
- **Vulnerability Type**: CWE-20 (Improper Input Validation)
- **Root Cause**:
  In `libaudiofile/modules/WAVE.cpp`, coefficient validation relied solely on an `assert()` statement:
  ```cpp
  /* numCoefficients should be at least 7. */
  assert(numCoefficients >= 7 && numCoefficients <= 255);
  ```
  In production/release builds where `NDEBUG` is defined, `assert()` statements are stripped by the compiler. Malformed WAV input files with `numCoefficients < 7` bypassed validation, leading directly to an AddressSanitizer heap-buffer-overflow write in `MSADPCM::decodeBlock()`:
  ```text
  ==210706==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x62d00001c45a at pc 0x7ffff7b7576c
  WRITE of size 2 at 0x62d00001c45a thread T0
      #0 0x7ffff7b7576b in MSADPCM::decodeBlock(unsigned char const*, short*) .../MSADPCM.cpp:222:14
  ```
- **Reference Repair (`NEW`)**:
  The developer patch adds explicit runtime bounds checking:
  ```cpp
  /* numCoefficients should be at least 7. */
  assert(numCoefficients >= 7 && numCoefficients <= 255);
  if (numCoefficients < 7 || numCoefficients > 255)
  {
      _af_error(AF_BAD_HEADER,
              "Bad number of coefficients");
      return AF_FAIL;
  }
  ```
- **Associated Files Discovered**:
  - `CVE-2017-6832_CWE-20_WAVE.cpp_OLD.cpp` (Vulnerable source)
  - `CVE-2017-6832_CWE-20_WAVE.cpp_NEW.cpp` (Reference fixed source)
  - `CVE-2017-6832_CWE-20_WAVE.cpp.diff` (Human patch diff)
  - `README.txt` (Case metadata, build command, ASan stack trace)
  - `exploit` (PoC binary payload - never executed)
  - `setup.sh` (Benchmark setup script - never executed)

---

### Benchmark-Wide Layout & Naming Patterns (144 Cases Audited)

Inspection across all 144 Vul4C benchmark case directories revealed several conventions and anomalies that `CaseManager` handles robustly:

1. **Standard OLD/NEW Source Suffixes**:
   - `_OLD.<ext>` paired with `_NEW.<ext>` (e.g. `WAVE.cpp_OLD.cpp` / `WAVE.cpp_NEW.cpp`).
   - Extensions observed: `.c`, `.cpp`, `.cc`, `.h`, `.h.in`.
2. **Multi-File Vulnerability Cases**:
   - Cases touching multiple files or functions often suffix indexed identifiers (e.g., `audiofile/CVE-2017-6834` with `sfconvert.c_OLD-1.c` and `sfconvert.c_OLD-2.c`).
   - `CaseManager` accurately pairs each index with its matching `NEW` counterpart and corresponding diff (`sfconvert.c-1.diff`, `sfconvert.c-2.diff`).
3. **Irregular Filename Conventions**:
   - **Dot-separated markers**: e.g., in `libtiff/CVE-2016-9532`, `tiffcrop.c.OLD.c` and `tiffcrop.c.NEW.c`.
   - **Extension-swapped markers**: e.g., in `libdwarf/CVE-2016-9276`, `libdwarf.h.in.OLD_h.in` vs `libdwarf.h.in_NEW.h.in`.
4. **CWE Identification**:
   - Most files encode numeric CWEs (`CWE-20`, `CWE-119`, `CWE-476`, `CWE-787`).
   - Certain cases encode non-numeric identifiers (e.g., `jasper/CVE-2016-9388` has `NVD-CWE-Other`).
5. **README Format**:
   - 140 cases have `README.txt`; 4 cases have `README.md`.
   - Both formats are seamlessly parsed for `Affected version`, `Commit hash`, `Source code`, and sanitizer error signatures.

---

## 3. Module Architecture (`pipeline/case_manager.py`)

- **Strict Static Analysis**: The module reads filesystem entries and text files only. It never invokes compilers, setup scripts, or PoC exploits.
- **Fail-Fast Error Reporting**:
  - `CaseNotFoundError`: Directory does not exist.
  - `AmbiguousCaseError`: A case ID query matches multiple projects.
  - `MissingFileError`: Missing vulnerable files, or unmatched OLD/NEW counterparts.
  - `AmbiguousMatchError`: Multiple candidate files match an OLD or diff file.
  - `SecurityPathError`: Path traversal attempts (e.g., `..`).
- **Structured Representation**:
  - `SourcePair`: Dataclass containing `file_name`, `old_path`, `new_path`, `diff_path`, `cve_id`, `cwe_id`.
  - `CaseMetadata`: Dataclass containing complete case context and paths, with `.to_dict()` for JSON serialization.

---

## 4. How to Run the Module

### Command-Line Interface (CLI)

```bash
# Basic inspection of audiofile/CVE-2017-6832
python3 pipeline/case_manager.py audiofile/CVE-2017-6832

# Lookup by bare case ID (auto-locates project)
python3 pipeline/case_manager.py CVE-2017-6832

# Multi-file case inspection
python3 pipeline/case_manager.py audiofile/CVE-2017-6834

# Output structured JSON
python3 pipeline/case_manager.py audiofile/CVE-2017-6832 --json

# Explicit benchmark root
python3 pipeline/case_manager.py audiofile/CVE-2017-6832 --benchmark-root /home/rohit_kumawat/SoK/Vul4C-Benchmark
```

### Python API

```python
from pipeline.case_manager import CaseManager

manager = CaseManager("/home/rohit_kumawat/SoK/Vul4C-Benchmark")
case = manager.load_case("audiofile/CVE-2017-6832")

print(f"Case: {case.case_id}, CWE: {case.cwe_id}")
for pair in case.source_pairs:
    print(f"File: {pair.file_name}")
    print(f"  OLD: {pair.old_path}")
    print(f"  NEW: {pair.new_path}")
    print(f"  Diff: {pair.diff_path}")
```

---

## 5. How to Run the Unit Tests

The test suite in `tests/test_case_manager.py` uses Python's standard `unittest` library and isolated `tempfile.TemporaryDirectory` fixtures. Tests never touch or execute benchmark binaries.

```bash
# Run via unittest discovery
python3 -m unittest discover tests -v

# Run the test file directly
python3 tests/test_case_manager.py -v
```

---

## 6. Dataset Preparation & Experimental Considerations

1. **Role of Reference Repair (`NEW`)**:
   - The reference `NEW` source provided in Vul4C represents how human developers resolved the issue at that commit.
   - It serves as a ground-truth *reference*, not proof that any single repair is uniquely secure or that alternate repairs generated by an LLM are invalid.
   - LLM-generated candidates will need functional and security validation (compilation check, functional tests, and AddressSanitizer verification).
2. **Multi-File Vulnerability Scope**:
   - Single-file cases (`CVE-2017-6832`) provide a clean starting point for single-file prompt generation and patching.
   - Multi-file cases (`CVE-2017-6834`) require tracking multiple diffs and coherent multi-file edits.
3. **No Synthetic Results**:
   - In accordance with research integrity, no synthetic model evaluation scores or premature conclusions are recorded at this stage.


---

## 7. Dataset Builder Architecture (`pipeline/dataset_builder.py`)

The `DatasetBuilder` transforms validated Vul4C benchmark cases into supervised fine-tuning (SFT) examples formatted in standard conversational message structures (`system`, `user`, `assistant`).

```text
CaseManager -> DatasetBuilder -> Train / Val / Test JSONL + Manifest
```

### Supported Task Types

1. **`detection`**:
   - **System Prompt**: Security auditor role analyzing C/C++ code.
   - **User Input**: Vulnerable source code (`_OLD`), project name, logical file name, and file scope context (e.g. single-file vs file $i$ of $N$).
   - **Assistant Target**: Ground-truth structured JSON report containing:
     - `vulnerability_present`: `true`
     - `cve_id`: CVE identifier extracted from benchmark metadata.
     - `cwe_id`: CWE identifier extracted from benchmark metadata.
     - `affected_file`: Logical file name.
     - `sanitizer_issue`: Sanitizer / ASan error signature extracted from benchmark `README.txt` (if available).
     - `summary`: Concise description based strictly on ground-truth benchmark metadata without inventing details.
2. **`repair`**:
   - **System Prompt**: Security software engineer repairing vulnerabilities in C/C++ source code.
   - **User Input**: Vulnerable source code (`_OLD`), repair instructions specifying project and file, vulnerability context (CVE/CWE if known), and multi-file scope. **No reference repair content or diffs are included in the prompt.**
   - **Assistant Target**: The exact reference fixed source code (`_NEW`). **Preserved byte-for-byte without automatic alteration or line-ending normalization.**

### Multi-File Case Handling & Scope Preservation

In cases with multiple modified files (e.g. `audiofile/CVE-2017-6834` with 3 source pairs), the relationship among files is explicitly preserved:
- Each source pair produces an independent training sample with explicit provenance.
- The sample metadata and user prompt explicitly state the file scope:
  `Note: This file is file 1 of 3 in case CVE-2017-6834. Companion files in this repair: [sfconvert.c-1, sfconvert.c-2].`
- The sample metadata records `file_index`, `total_files_in_case`, and `companion_files`.

### Data Leakage Prevention & Deterministic Splitting

- **Strict CVE Case-Level Splitting**: Splitting occurs at the vulnerability case level, never at the source file level.
- **Zero Cross-Split Leakage**: All samples belonging to a CVE case are assigned exclusively to one split (`train`, `validation`, or `test`).
- **Deterministic**: Case IDs are lexicographically sorted before pseudo-random shuffling using a fixed, configurable random seed (`--seed 42`).
- **Duplicate Detection**: Duplicate or conflicting case IDs in input configurations are detected and rejected (`DuplicateCaseError`) rather than silently processed.

### JSONL Schema (Chat / Messages Format)

Each line in `train.jsonl`, `validation.jsonl`, and `test.jsonl` follows standard SFT format:
```json
{
  "sample_id": "audiofile_CVE-2017-6832_WAVE.cpp_repair",
  "task_type": "repair",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "metadata": {
    "case_id": "CVE-2017-6832",
    "project": "audiofile",
    "cve_id": "CVE-2017-6832",
    "cwe_id": "CWE-20",
    "task_type": "repair",
    "split": "train",
    "file_name": "WAVE.cpp",
    "file_index": 1,
    "total_files_in_case": 1,
    "companion_files": [],
    "old_path": "/home/.../WAVE.cpp_OLD.cpp",
    "new_path": "/home/.../WAVE.cpp_NEW.cpp",
    "diff_path": "/home/.../WAVE.cpp.diff"
  }
}
```

### Audit Manifest (`manifest.json`)

The manifest generated alongside the dataset records:
- Creation timestamp (ISO 8601 UTC)
- Benchmark root path and random seed
- Configured split ratios
- Case counts by split (`train`, `validation`, `test`, `total`)
- Sample counts by split and task type
- Case and sample summary tables

---

## 8. How to Run the Dataset Builder

### Preview Mode (Dry-Run)
Inspect sample counts and metadata without writing massive source files:

```bash
# Preview single case
python3 pipeline/dataset_builder.py --case audiofile/CVE-2017-6832 --preview

# Preview single case and save preview metadata JSON
python3 pipeline/dataset_builder.py --case audiofile/CVE-2017-6832 --preview --preview-file results/dataset_preview.json

# Preview multi-file case
python3 pipeline/dataset_builder.py --case audiofile/CVE-2017-6834 --preview

# Preview multiple cases
python3 pipeline/dataset_builder.py -c audiofile/CVE-2017-6832 -c audiofile/CVE-2017-6834 --preview
```

### Full Generation & Export Mode

```bash
# Generate datasets to default output location (data/generated/)
python3 pipeline/dataset_builder.py --case audiofile/CVE-2017-6832 --output-dir data/generated

# Generate specific task types only
python3 pipeline/dataset_builder.py --case audiofile/CVE-2017-6832 --tasks detection
python3 pipeline/dataset_builder.py --case audiofile/CVE-2017-6832 --tasks repair

# Configure custom split ratios and seed
python3 pipeline/dataset_builder.py -c audiofile/CVE-2017-6832 -c audiofile/CVE-2017-6834   --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 --seed 42 --output-dir data/generated
```

### Python API

```python
from pipeline.dataset_builder import DatasetBuilder

builder = DatasetBuilder("/home/rohit_kumawat/SoK/Vul4C-Benchmark", seed=42)
samples, manifest = builder.build_dataset(["audiofile/CVE-2017-6832"])
written_files = builder.export_dataset(samples, manifest, "data/generated")
print(written_files)
```

---

## 9. Running the Unit Tests

All unit tests operate on synthetic temporary directories and never run benchmark binaries:

```bash
# Run all tests across the project
python3 -m unittest discover tests -v

# Run dataset builder tests specifically
python3 tests/test_dataset_builder.py -v

# Run case manager tests specifically
python3 tests/test_case_manager.py -v
```

---

## 10. Assumptions, Safety, & Known Limitations

1. **Reference Repairs as Ground Truth Baseline**:
   - The `_NEW` source code is treated as an upstream developer reference repair. It provides a training target for supervised fine-tuning, but does not guarantee that no secondary vulnerabilities exist.
2. **Evaluation Protocol**:
   - Test split examples must remain strictly untouched during training and hyperparameter tuning.
3. **Repository Cleanliness & Benchmark Protection**:
   - The original Vul4C benchmark remains strictly read-only.
   - Generated dataset files in `data/` are excluded by `.gitignore` (`data/*`, `!data/.gitkeep`) to prevent committing large source trees to Git.
