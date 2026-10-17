"""Unit tests for pipeline.case_manager module.

All tests use isolated temporary directories and synthetic fixture files.
No benchmark binaries, scripts, or vulnerable programs are executed.
"""

import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

# Ensure repository root is on sys.path for direct script execution
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from pipeline.case_manager import (
    AmbiguousCaseError,
    CaseManager,
    CaseManagerError,
    CaseMetadata,
    CaseNotFoundError,
    MissingFileError,
    SecurityPathError,
    SourcePair,
    main,
)


class TestCaseManager(unittest.TestCase):
    """Test suite for CaseManager operations and edge cases."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.benchmark_root = Path(self.temp_dir.name)
        self.manager = CaseManager(benchmark_root=self.benchmark_root)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _create_synthetic_case(
        self,
        project: str,
        case_id: str,
        files: dict,
    ) -> Path:
        """Helper to create synthetic benchmark case directories."""
        case_dir = self.benchmark_root / project / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        for name, content in files.items():
            (case_dir / name).write_text(content, encoding="utf-8")
        return case_dir

    def test_single_pair_loading(self) -> None:
        """Test loading a case with a single vulnerable/fixed file pair."""
        files = {
            "CVE-2017-6832_CWE-20_WAVE.cpp_OLD.cpp": "void vuln() { assert(x); }",
            "CVE-2017-6832_CWE-20_WAVE.cpp_NEW.cpp": "void vuln() { if (!x) return; }",
            "CVE-2017-6832_CWE-20_WAVE.cpp.diff": "--- OLD\n+++ NEW",
            "README.txt": (
                "Affected version:\n0.3.6\n\n"
                "Source code:\ngit clone https://github.com/mpruett/audiofile.git source\n\n"
                "Commit hash:\nd19a5ace5ae891c778cbc710d78634de6084b846\n\n"
                "Expected Output:\nAddressSanitizer: heap-buffer-overflow\n"
            ),
            "exploit": "raw exploit payload bytes",
            "setup.sh": "#!/bin/bash\necho setup",
        }
        case_dir = self._create_synthetic_case("audiofile", "CVE-2017-6832", files)

        metadata = self.manager.load_case("audiofile/CVE-2017-6832")

        self.assertEqual(metadata.case_id, "CVE-2017-6832")
        self.assertEqual(metadata.project, "audiofile")
        self.assertEqual(metadata.cve_id, "CVE-2017-6832")
        self.assertEqual(metadata.cwe_id, "CWE-20")
        self.assertEqual(metadata.affected_version, "0.3.6")
        self.assertEqual(metadata.commit_hash, "d19a5ace5ae891c778cbc710d78634de6084b846")
        self.assertEqual(metadata.repository_url, "https://github.com/mpruett/audiofile")
        self.assertIn("AddressSanitizer", metadata.sanitizer_issue or "")
        self.assertEqual(len(metadata.source_pairs), 1)

        pair = metadata.source_pairs[0]
        self.assertEqual(pair.file_name, "WAVE.cpp")
        self.assertEqual(pair.old_path.name, "CVE-2017-6832_CWE-20_WAVE.cpp_OLD.cpp")
        self.assertEqual(pair.new_path.name, "CVE-2017-6832_CWE-20_WAVE.cpp_NEW.cpp")
        self.assertIsNotNone(pair.diff_path)
        self.assertEqual(pair.diff_path.name, "CVE-2017-6832_CWE-20_WAVE.cpp.diff")

        extra_names = [f.name for f in metadata.extra_files]
        self.assertIn("exploit", extra_names)
        self.assertIn("setup.sh", extra_names)

    def test_multi_pair_loading(self) -> None:
        """Test loading a case with multiple source pairs and indexed files."""
        files = {
            "CVE-2017-6834_CWE-119_MSADPCM.cpp_OLD.cpp": "old msadpcm",
            "CVE-2017-6834_CWE-119_MSADPCM.cpp_NEW.cpp": "new msadpcm",
            "CVE-2017-6834_CWE-119_MSADPCM.cpp.diff": "diff 1",
            "CVE-2017-6834_CWE-119_sfconvert.c_OLD-1.c": "old sfconvert 1",
            "CVE-2017-6834_CWE-119_sfconvert.c_NEW-1.c": "new sfconvert 1",
            "CVE-2017-6834_CWE-119_sfconvert.c-1.diff": "diff sfconvert 1",
            "CVE-2017-6834_CWE-119_sfconvert.c_OLD-2.c": "old sfconvert 2",
            "CVE-2017-6834_CWE-119_sfconvert.c_NEW-2.c": "new sfconvert 2",
            "CVE-2017-6834_CWE-119_sfconvert.c-2.diff": "diff sfconvert 2",
            "README.txt": "Affected version:\n0.3.6\n",
        }
        self._create_synthetic_case("audiofile", "CVE-2017-6834", files)

        metadata = self.manager.load_case("audiofile/CVE-2017-6834")
        self.assertEqual(len(metadata.source_pairs), 3)

        logical_names = [p.file_name for p in metadata.source_pairs]
        self.assertIn("MSADPCM.cpp", logical_names)
        self.assertIn("sfconvert.c-1", logical_names)
        self.assertIn("sfconvert.c-2", logical_names)

        for pair in metadata.source_pairs:
            self.assertIsNotNone(pair.diff_path)
            self.assertTrue(pair.diff_path.exists())

    def test_dot_notation_and_irregular_naming(self) -> None:
        """Test benchmark anomalies such as .OLD. and .OLD_h.in naming."""
        files = {
            "CVE-2016-9532_CWE-125_tiffcrop.c.OLD.c": "old tiffcrop",
            "CVE-2016-9532_CWE-125_tiffcrop.c.NEW.c": "new tiffcrop",
            "CVE-2016-9532_CWE-125_tiffcrop.c.diff": "tiffcrop diff",
            "README.txt": "Affected version:\n4.0.6\n",
        }
        self._create_synthetic_case("libtiff", "CVE-2016-9532", files)
        metadata = self.manager.load_case("libtiff/CVE-2016-9532")
        self.assertEqual(len(metadata.source_pairs), 1)
        self.assertEqual(metadata.source_pairs[0].file_name, "tiffcrop.c")
        self.assertEqual(metadata.source_pairs[0].diff_path.name, "CVE-2016-9532_CWE-125_tiffcrop.c.diff")

        # Test libdwarf irregular naming
        dwarf_files = {
            "CVE-2016-9276_CWE-125_libdwarf.h.in.OLD_h.in": "old dwarf header",
            "CVE-2016-9276_CWE-125_libdwarf.h.in_NEW.h.in": "new dwarf header",
            "CVE-2016-9276_CWE-125_libdwarf.h.in.diff": "dwarf header diff",
        }
        self._create_synthetic_case("libdwarf", "CVE-2016-9276", dwarf_files)
        dwarf_meta = self.manager.load_case("libdwarf/CVE-2016-9276")
        self.assertEqual(len(dwarf_meta.source_pairs), 1)
        self.assertEqual(dwarf_meta.source_pairs[0].file_name, "libdwarf.h.in")

    def test_readme_parsing_variations(self) -> None:
        """Test README.md fallback and missing README handling."""
        # Markdown readme
        files_md = {
            "CVE-2006-2025_CWE-119_tif_dir.c_OLD.c": "old code",
            "CVE-2006-2025_CWE-119_tif_dir.c_NEW.c": "new code",
            "README.md": "# CVE-2006-2025\nAffected version:\n3.8.2\nCommit hash:\n1234567890abcdef1234567890abcdef12345678\n",
        }
        self._create_synthetic_case("libtiff", "CVE-2006-2025", files_md)
        meta_md = self.manager.load_case("libtiff/CVE-2006-2025")
        self.assertEqual(meta_md.affected_version, "3.8.2")
        self.assertEqual(meta_md.commit_hash, "1234567890abcdef1234567890abcdef12345678")

        # No readme
        files_no_readme = {
            "CVE-2017-9999_CWE-20_test.c_OLD.c": "old",
            "CVE-2017-9999_CWE-20_test.c_NEW.c": "new",
        }
        self._create_synthetic_case("testproj", "CVE-2017-9999", files_no_readme)
        meta_none = self.manager.load_case("testproj/CVE-2017-9999")
        self.assertIsNone(meta_none.readme_path)
        self.assertIsNone(meta_none.affected_version)

    def test_missing_old_source_files_raises_error(self) -> None:
        """Ensure MissingFileError is raised when no vulnerable source files exist."""
        files = {"README.txt": "just a readme"}
        self._create_synthetic_case("proj", "CVE-2020-0001", files)
        with self.assertRaises(MissingFileError):
            self.manager.load_case("proj/CVE-2020-0001")

    def test_missing_new_source_counterpart_raises_error(self) -> None:
        """Ensure MissingFileError is raised when an OLD file has no matching NEW file."""
        files = {
            "CVE-2020-0002_CWE-20_test.c_OLD.c": "old code",
            # Notice: NEW file intentionally omitted
        }
        self._create_synthetic_case("proj", "CVE-2020-0002", files)
        with self.assertRaises(MissingFileError):
            self.manager.load_case("proj/CVE-2020-0002")

    def test_unmatched_new_file_raises_error(self) -> None:
        """Ensure MissingFileError is raised when an extra un-paired NEW file exists."""
        files = {
            "CVE-2020-0003_CWE-20_test.c_OLD.c": "old",
            "CVE-2020-0003_CWE-20_test.c_NEW.c": "new",
            "CVE-2020-0003_CWE-20_orphan.c_NEW.c": "extra new",
        }
        self._create_synthetic_case("proj", "CVE-2020-0003", files)
        with self.assertRaises(MissingFileError):
            self.manager.load_case("proj/CVE-2020-0003")

    def test_case_locating_bare_id_and_ambiguity(self) -> None:
        """Test locating by bare case ID and raising error on multiple project matches."""
        files = {
            "CVE-2018-1111_CWE-20_code.c_OLD.c": "old",
            "CVE-2018-1111_CWE-20_code.c_NEW.c": "new",
        }
        self._create_synthetic_case("projA", "CVE-2018-1111", files)

        # Can locate by bare ID
        meta = self.manager.load_case("CVE-2018-1111")
        self.assertEqual(meta.project, "projA")

        # Duplicate case ID in another project should trigger AmbiguousCaseError
        self._create_synthetic_case("projB", "CVE-2018-1111", files)
        with self.assertRaises(AmbiguousCaseError):
            self.manager.load_case("CVE-2018-1111")

    def test_case_not_found(self) -> None:
        """Test that non-existent case raises CaseNotFoundError."""
        with self.assertRaises(CaseNotFoundError):
            self.manager.load_case("non_existent_project/CVE-9999-9999")

    def test_security_directory_traversal_prevention(self) -> None:
        """Ensure directory traversal attempts raise SecurityPathError."""
        with self.assertRaises(SecurityPathError):
            self.manager.locate_case_dir("../../etc/passwd")
        with self.assertRaises(SecurityPathError):
            self.manager.locate_case_dir("proj/../../../outside")

    def test_to_dict_serialization(self) -> None:
        """Test converting CaseMetadata and SourcePair to dictionary and JSON."""
        files = {
            "CVE-2017-6832_CWE-20_WAVE.cpp_OLD.cpp": "old",
            "CVE-2017-6832_CWE-20_WAVE.cpp_NEW.cpp": "new",
            "CVE-2017-6832_CWE-20_WAVE.cpp.diff": "diff",
            "README.txt": "Affected version:\n0.3.6\n",
        }
        self._create_synthetic_case("audiofile", "CVE-2017-6832", files)
        meta = self.manager.load_case("audiofile/CVE-2017-6832")
        d = meta.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["case_id"], "CVE-2017-6832")
        self.assertEqual(d["num_source_pairs"], 1)
        json_str = json.dumps(d)
        self.assertTrue(len(json_str) > 50)

    def test_cli_execution(self) -> None:
        """Test main() CLI interface with text output, json output, and error conditions."""
        files = {
            "CVE-2017-6832_CWE-20_WAVE.cpp_OLD.cpp": "old",
            "CVE-2017-6832_CWE-20_WAVE.cpp_NEW.cpp": "new",
        }
        self._create_synthetic_case("audiofile", "CVE-2017-6832", files)

        # Test text output
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            code = main([
                "--benchmark-root", str(self.benchmark_root),
                "audiofile/CVE-2017-6832",
            ])
            self.assertEqual(code, 0)
            output = mock_stdout.getvalue()
            self.assertIn("Vul4C Case Report: CVE-2017-6832", output)
            self.assertIn("WAVE.cpp", output)

        # Test json output
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            code = main([
                "--benchmark-root", str(self.benchmark_root),
                "audiofile/CVE-2017-6832",
                "--json",
            ])
            self.assertEqual(code, 0)
            parsed = json.loads(mock_stdout.getvalue())
            self.assertEqual(parsed["case_id"], "CVE-2017-6832")
            self.assertEqual(parsed["project"], "audiofile")

        # Test error handling in CLI
        with patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
            code = main([
                "--benchmark-root", str(self.benchmark_root),
                "nonexistent/case",
            ])
            self.assertEqual(code, 1)
            self.assertIn("Error:", mock_stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
