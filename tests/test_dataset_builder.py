"""Unit tests for pipeline.dataset_builder module.

All tests operate strictly on synthetic fixtures created in temporary directories.
Never executes benchmark binaries, exploits, or setup scripts.
"""

import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.dataset_builder import (
    ChatMessage,
    DatasetBuilder,
    DatasetBuilderError,
    DatasetManifest,
    DatasetSample,
    DuplicateCaseError,
    SplitConfigurationError,
    ValidationError,
    main,
)


class TestDatasetBuilder(unittest.TestCase):
    """Comprehensive test suite for DatasetBuilder."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.benchmark_root = Path(self.temp_dir.name) / "benchmark"
        self.benchmark_root.mkdir(parents=True, exist_ok=True)
        self.output_dir = Path(self.temp_dir.name) / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _create_synthetic_case(self, project: str, case_id: str, files: dict) -> Path:
        case_dir = self.benchmark_root / project / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        for name, content in files.items():
            (case_dir / name).write_text(content, encoding="utf-8")
        return case_dir

    def test_single_case_dataset_generation(self) -> None:
        files = {
            "CVE-2017-6832_CWE-20_WAVE.cpp_OLD.cpp": "void vuln() { assert(n >= 7); }",
            "CVE-2017-6832_CWE-20_WAVE.cpp_NEW.cpp": "void vuln() { if (n < 7) return AF_FAIL; }",
            "CVE-2017-6832_CWE-20_WAVE.cpp.diff": "--- OLD\n+++ NEW",
            "README.txt": "Affected version:\n0.3.6\nCommit hash:\nabc12345\nExpected Output:\nAddressSanitizer: heap-buffer-overflow\n",
        }
        self._create_synthetic_case("audiofile", "CVE-2017-6832", files)

        builder = DatasetBuilder(benchmark_root=self.benchmark_root)
        samples, manifest = builder.build_dataset(["audiofile/CVE-2017-6832"])

        self.assertEqual(len(samples), 2)
        task_types = [s.task_type for s in samples]
        self.assertIn("detection", task_types)
        self.assertIn("repair", task_types)

        # Check detection sample
        det_sample = next(s for s in samples if s.task_type == "detection")
        self.assertEqual(det_sample.metadata["case_id"], "CVE-2017-6832")
        self.assertEqual(det_sample.metadata["cwe_id"], "CWE-20")
        self.assertEqual(det_sample.metadata["file_name"], "WAVE.cpp")

        det_target = json.loads(det_sample.messages[2].content)
        self.assertTrue(det_target["vulnerability_present"])
        self.assertEqual(det_target["cwe_id"], "CWE-20")
        self.assertEqual(det_target["cve_id"], "CVE-2017-6832")
        self.assertIn("AddressSanitizer", det_target["sanitizer_issue"])

        # Check repair sample
        rep_sample = next(s for s in samples if s.task_type == "repair")
        self.assertEqual(rep_sample.messages[2].content, files["CVE-2017-6832_CWE-20_WAVE.cpp_NEW.cpp"])

    def test_multi_file_case_relationships(self) -> None:
        files = {
            "CVE-2017-6834_CWE-119_MSADPCM.cpp_OLD.cpp": "old msadpcm",
            "CVE-2017-6834_CWE-119_MSADPCM.cpp_NEW.cpp": "new msadpcm",
            "CVE-2017-6834_CWE-119_MSADPCM.cpp.diff": "diff 1",
            "CVE-2017-6834_CWE-119_sfconvert.c_OLD-1.c": "old sfconvert 1",
            "CVE-2017-6834_CWE-119_sfconvert.c_NEW-1.c": "new sfconvert 1",
            "CVE-2017-6834_CWE-119_sfconvert.c-1.diff": "diff sfconvert 1",
            "README.txt": "Affected version:\n0.3.6\n",
        }
        self._create_synthetic_case("audiofile", "CVE-2017-6834", files)

        builder = DatasetBuilder(benchmark_root=self.benchmark_root)
        samples, manifest = builder.build_dataset(["audiofile/CVE-2017-6834"])

        self.assertEqual(len(samples), 4)  # 2 files * 2 tasks
        for s in samples:
            self.assertEqual(s.metadata["total_files_in_case"], 2)
            self.assertTrue(len(s.metadata["companion_files"]) == 1)
            # Verify file scope is explicitly communicated in prompt
            user_msg = s.messages[1].content
            self.assertIn(f"file {s.metadata['file_index']} of 2", user_msg)
            self.assertIn("Companion files", user_msg)

    def test_exact_reference_preservation(self) -> None:
        exact_new_content = "/* EXACT PRESERVATION TEST */\n  int x = 42;\r\n\treturn x;\n"
        files = {
            "CVE-2021-0001_CWE-20_test.c_OLD.c": "int x = 0;",
            "CVE-2021-0001_CWE-20_test.c_NEW.c": exact_new_content,
        }
        self._create_synthetic_case("proj", "CVE-2021-0001", files)

        builder = DatasetBuilder(benchmark_root=self.benchmark_root)
        samples, _ = builder.build_dataset(["proj/CVE-2021-0001"], task_types=["repair"])

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].messages[2].content, exact_new_content)

    def test_no_reference_leakage_in_repair_input(self) -> None:
        secret_fixed_code = "SECRET_FIXED_CODE_XYZ_12345"
        files = {
            "CVE-2021-0002_CWE-20_test.c_OLD.c": "vulnerable_code();",
            "CVE-2021-0002_CWE-20_test.c_NEW.c": secret_fixed_code,
        }
        self._create_synthetic_case("proj", "CVE-2021-0002", files)

        builder = DatasetBuilder(benchmark_root=self.benchmark_root)
        samples, _ = builder.build_dataset(["proj/CVE-2021-0002"], task_types=["repair"])

        user_input = samples[0].messages[1].content
        self.assertNotIn(secret_fixed_code, user_input)

    def test_deterministic_case_level_splitting_and_no_leakage(self) -> None:
        case_names = []
        for i in range(10):
            cid = f"CVE-2020-00{i:02d}"
            case_names.append(cid)
            files = {
                f"{cid}_CWE-20_test.c_OLD.c": f"old {i}",
                f"{cid}_CWE-20_test.c_NEW.c": f"new {i}",
            }
            self._create_synthetic_case("testproj", cid, files)

        builder1 = DatasetBuilder(benchmark_root=self.benchmark_root, seed=123, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
        samples1, manifest1 = builder1.build_dataset([f"testproj/{c}" for c in case_names])

        builder2 = DatasetBuilder(benchmark_root=self.benchmark_root, seed=123, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
        samples2, manifest2 = builder2.build_dataset([f"testproj/{c}" for c in case_names])

        # Determinism check
        splits1 = {s.metadata["case_id"]: s.metadata["split"] for s in samples1}
        splits2 = {s.metadata["case_id"]: s.metadata["split"] for s in samples2}
        self.assertEqual(splits1, splits2)

        # Leakage check: verify no case spans multiple splits
        case_to_splits = {}
        for s in samples1:
            cid = s.metadata["case_id"]
            case_to_splits.setdefault(cid, set()).add(s.metadata["split"])

        for cid, sp_set in case_to_splits.items():
            self.assertEqual(len(sp_set), 1, f"Case {cid} has samples in multiple splits: {sp_set}")

    def test_duplicate_case_detection(self) -> None:
        files = {
            "CVE-2021-0003_CWE-20_test.c_OLD.c": "old",
            "CVE-2021-0003_CWE-20_test.c_NEW.c": "new",
        }
        self._create_synthetic_case("proj", "CVE-2021-0003", files)

        builder = DatasetBuilder(benchmark_root=self.benchmark_root)
        with self.assertRaises(DuplicateCaseError):
            builder.build_dataset(["proj/CVE-2021-0003", "proj/CVE-2021-0003"])

    def test_empty_source_file_validation(self) -> None:
        files = {
            "CVE-2021-0004_CWE-20_test.c_OLD.c": "   \n",  # empty
            "CVE-2021-0004_CWE-20_test.c_NEW.c": "valid new",
        }
        self._create_synthetic_case("proj", "CVE-2021-0004", files)

        builder = DatasetBuilder(benchmark_root=self.benchmark_root)
        with self.assertRaises(ValidationError):
            builder.build_dataset(["proj/CVE-2021-0004"])

    def test_invalid_split_ratios(self) -> None:
        with self.assertRaises(SplitConfigurationError):
            DatasetBuilder(benchmark_root=self.benchmark_root, train_ratio=0.8, val_ratio=0.3, test_ratio=0.1)
        with self.assertRaises(SplitConfigurationError):
            DatasetBuilder(benchmark_root=self.benchmark_root, train_ratio=-0.1, val_ratio=0.6, test_ratio=0.5)

    def test_export_dataset_files(self) -> None:
        files = {
            "CVE-2017-6832_CWE-20_WAVE.cpp_OLD.cpp": "old code",
            "CVE-2017-6832_CWE-20_WAVE.cpp_NEW.cpp": "new code",
        }
        self._create_synthetic_case("audiofile", "CVE-2017-6832", files)

        builder = DatasetBuilder(benchmark_root=self.benchmark_root)
        samples, manifest = builder.build_dataset(["audiofile/CVE-2017-6832"])
        written = builder.export_dataset(samples, manifest, output_dir=self.output_dir)

        self.assertTrue((self.output_dir / "train.jsonl").exists())
        self.assertTrue((self.output_dir / "validation.jsonl").exists())
        self.assertTrue((self.output_dir / "test.jsonl").exists())
        self.assertTrue((self.output_dir / "manifest.json").exists())

        train_lines = (self.output_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(train_lines), 2)

    def test_preview_mode(self) -> None:
        files = {
            "CVE-2017-6832_CWE-20_WAVE.cpp_OLD.cpp": "old code",
            "CVE-2017-6832_CWE-20_WAVE.cpp_NEW.cpp": "new code",
        }
        self._create_synthetic_case("audiofile", "CVE-2017-6832", files)

        builder = DatasetBuilder(benchmark_root=self.benchmark_root)
        preview = builder.preview_dataset(["audiofile/CVE-2017-6832"])

        self.assertEqual(preview["status"], "valid")
        self.assertEqual(preview["total_samples"], 2)
        self.assertEqual(len(preview["sample_previews"]), 2)
        self.assertIn("input_chars", preview["sample_previews"][0])

    def test_cli_execution_preview_and_export(self) -> None:
        files = {
            "CVE-2017-6832_CWE-20_WAVE.cpp_OLD.cpp": "old",
            "CVE-2017-6832_CWE-20_WAVE.cpp_NEW.cpp": "new",
        }
        self._create_synthetic_case("audiofile", "CVE-2017-6832", files)

        # CLI preview
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            code = main([
                "--benchmark-root", str(self.benchmark_root),
                "--case", "audiofile/CVE-2017-6832",
                "--preview",
            ])
            self.assertEqual(code, 0)
            self.assertIn("Preview Mode", mock_stdout.getvalue())

        # CLI export
        out_cli = self.output_dir / "cli_out"
        code = main([
            "--benchmark-root", str(self.benchmark_root),
            "--case", "audiofile/CVE-2017-6832",
            "--output-dir", str(out_cli),
        ])
        self.assertEqual(code, 0)
        self.assertTrue((out_cli / "manifest.json").exists())



    def test_nonempty_splits_with_adequate_pilot_size(self) -> None:
        case_names = []
        for i in range(8):
            cid = f"CVE-2022-00{i:02d}"
            case_names.append(cid)
            files = {
                f"{cid}_CWE-20_test.c_OLD.c": f"old {i}",
                f"{cid}_CWE-20_test.c_NEW.c": f"new {i}",
            }
            self._create_synthetic_case("testproj", cid, files)

        builder = DatasetBuilder(benchmark_root=self.benchmark_root, seed=42)
        samples, manifest = builder.build_dataset([f"testproj/{c}" for c in case_names])

        cc = manifest.case_counts_by_split
        self.assertGreater(cc["train"], 0)
        self.assertGreater(cc["validation"], 0)
        self.assertGreater(cc["test"], 0)

    def test_small_pilot_size_behavior(self) -> None:
        # N=1 case
        files = {
            "CVE-2023-0001_CWE-20_test.c_OLD.c": "old",
            "CVE-2023-0001_CWE-20_test.c_NEW.c": "new",
        }
        self._create_synthetic_case("testproj", "CVE-2023-0001", files)
        builder = DatasetBuilder(benchmark_root=self.benchmark_root)
        samples, manifest = builder.build_dataset(["testproj/CVE-2023-0001"])
        self.assertEqual(manifest.case_counts_by_split["train"], 1)
        self.assertEqual(manifest.case_counts_by_split["validation"], 0)
        self.assertEqual(manifest.case_counts_by_split["test"], 0)


if __name__ == "__main__":
    unittest.main()
