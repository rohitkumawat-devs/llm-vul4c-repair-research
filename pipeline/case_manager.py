"""Case Manager for Vul4C Benchmark Cases.

This module is responsible for safely locating, validating, and extracting
metadata and source files for CVE cases in the Vul4C benchmark.

Design Principles:
- Pure static analysis: Never executes benchmark scripts, exploits, or binaries.
- Strict pairing: Accurately pairs vulnerable (_OLD) and reference fixed (_NEW)
  source files along with corresponding .diff patch files.
- Fail-fast & clear reporting: Raises explicit exceptions rather than guessing
  when files are missing, ambiguous, or invalid.
- Safe paths: Guards against path traversal and unauthorized directory escapes.
"""

import argparse
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple, Union


class CaseManagerError(Exception):
    """Base exception for all CaseManager errors."""
    pass


class CaseNotFoundError(CaseManagerError):
    """Raised when a requested case directory cannot be located."""
    pass


class AmbiguousCaseError(CaseManagerError):
    """Raised when a case query matches multiple candidate directories."""
    pass


class MissingFileError(CaseManagerError):
    """Raised when required files (such as source pairs) are missing."""
    pass


class AmbiguousMatchError(CaseManagerError):
    """Raised when multiple candidates match a source or diff file."""
    pass


class SecurityPathError(CaseManagerError):
    """Raised when directory traversal or unauthorized path access is attempted."""
    pass


@dataclass
class SourcePair:
    """Represents a pair of vulnerable and reference fixed source files."""
    file_name: str
    old_path: Path
    new_path: Path
    diff_path: Optional[Path] = None
    cve_id: Optional[str] = None
    cwe_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert source pair to dictionary representation."""
        return {
            "file_name": self.file_name,
            "old_path": str(self.old_path),
            "new_path": str(self.new_path),
            "diff_path": str(self.diff_path) if self.diff_path else None,
            "cve_id": self.cve_id,
            "cwe_id": self.cwe_id,
        }


@dataclass
class CaseMetadata:
    """Structured metadata and discovered paths for a benchmark CVE case."""
    case_id: str
    project: str
    case_dir: Path
    cve_id: Optional[str] = None
    cwe_id: Optional[str] = None
    source_pairs: List[SourcePair] = field(default_factory=list)
    diff_files: List[Path] = field(default_factory=list)
    readme_path: Optional[Path] = None
    affected_version: Optional[str] = None
    commit_hash: Optional[str] = None
    repository_url: Optional[str] = None
    sanitizer_issue: Optional[str] = None
    extra_files: List[Path] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert case metadata to a serializable dictionary."""
        return {
            "case_id": self.case_id,
            "project": self.project,
            "case_dir": str(self.case_dir),
            "cve_id": self.cve_id,
            "cwe_id": self.cwe_id,
            "num_source_pairs": len(self.source_pairs),
            "source_pairs": [pair.to_dict() for pair in self.source_pairs],
            "diff_files": [str(d) for d in self.diff_files],
            "readme_path": str(self.readme_path) if self.readme_path else None,
            "affected_version": self.affected_version,
            "commit_hash": self.commit_hash,
            "repository_url": self.repository_url,
            "sanitizer_issue": self.sanitizer_issue,
            "extra_files": [str(f) for f in self.extra_files],
        }


class CaseManager:
    """Discovers, validates, and loads metadata for Vul4C cases."""

    CVE_PATTERN = re.compile(r"(CVE-\d{4}-\d+)", re.IGNORECASE)
    CWE_PATTERN = re.compile(r"((?:NVD-)?CWE-(?:\d+|[A-Za-z]+))", re.IGNORECASE)

    def __init__(self, benchmark_root: Optional[Union[str, Path]] = None) -> None:
        """Initialize CaseManager with a benchmark root path."""
        if benchmark_root is None:
            default_path = os.environ.get(
                "VUL4C_BENCHMARK_ROOT",
                "/home/rohit_kumawat/SoK/Vul4C-Benchmark"
            )
            benchmark_root = Path(default_path)
        else:
            benchmark_root = Path(benchmark_root)

        self.benchmark_root = benchmark_root.resolve()

    def locate_case_dir(self, case_query: Union[str, Path]) -> Path:
        """Safely locate a case directory given a relative path, case ID, or full path."""
        query_path = Path(case_query)

        # 1. Handle absolute paths
        if query_path.is_absolute():
            resolved = query_path.resolve()
            if not resolved.exists() or not resolved.is_dir():
                raise CaseNotFoundError(f"Case directory does not exist: {resolved}")
            return resolved

        # 2. Check for directory traversal in relative path
        if ".." in query_path.parts:
            raise SecurityPathError(f"Directory traversal detected in case path: '{case_query}'")

        # 3. Direct relative path within benchmark root (e.g. 'audiofile/CVE-2017-6832')
        direct_candidate = (self.benchmark_root / query_path).resolve()
        if direct_candidate.exists() and direct_candidate.is_dir():
            try:
                direct_candidate.relative_to(self.benchmark_root)
            except ValueError:
                raise SecurityPathError(f"Target path escapes benchmark root: '{case_query}'")
            return direct_candidate

        # 4. Search by case identifier across benchmark subdirectories
        if not self.benchmark_root.exists():
            raise CaseNotFoundError(
                f"Benchmark root directory does not exist: {self.benchmark_root}"
            )

        case_name = query_path.name
        matches: List[Path] = []
        for project_dir in self.benchmark_root.iterdir():
            if project_dir.is_dir() and not project_dir.name.startswith("."):
                candidate = project_dir / case_name
                if candidate.exists() and candidate.is_dir():
                    matches.append(candidate.resolve())

        if not matches:
            raise CaseNotFoundError(
                f"Could not find case '{case_query}' under benchmark root {self.benchmark_root}"
            )
        if len(matches) > 1:
            match_strs = [str(m.relative_to(self.benchmark_root)) for m in matches]
            raise AmbiguousCaseError(
                f"Ambiguous case query '{case_query}'. Found in multiple projects: {match_strs}"
            )

        return matches[0]

    def _extract_cve(self, text: str) -> Optional[str]:
        """Extract standardized CVE ID from text or filename."""
        match = self.CVE_PATTERN.search(text)
        return match.group(1).upper() if match else None

    def _extract_cwe(self, text: str) -> Optional[str]:
        """Extract CWE ID from text or filename."""
        match = self.CWE_PATTERN.search(text)
        return match.group(1) if match else None

    def _extract_logical_name(self, old_filename: str) -> str:
        """Extract logical base source file name without vulnerability prefix and OLD suffix."""
        name = old_filename
        name = re.sub(r"^(?:CVE-\d{4}-\d+_)?(?:(?:NVD-)?CWE-[A-Za-z0-9_-]+_)?", "", name)

        m = re.search(r"^(.*?)_OLD(-\d+)?(?:\.[^.]+)+$", name)
        if m:
            base, idx = m.group(1), m.group(2)
            return f"{base}{idx}" if idx else base

        m = re.search(r"^(.*?)\.OLD(?:\.[^.]+)+$", name)
        if m:
            return m.group(1)

        m = re.search(r"^(.*?)\.OLD_.*$", name)
        if m:
            return m.group(1)

        return name

    def _is_old_file(self, filename: str) -> bool:
        """Determine if a filename represents a vulnerable (_OLD) source file."""
        return ("_OLD" in filename or ".OLD." in filename or ".OLD_" in filename)

    def _is_new_file(self, filename: str) -> bool:
        """Determine if a filename represents a fixed (_NEW) source file."""
        return ("_NEW" in filename or ".NEW." in filename or "_NEW." in filename)

    def _find_matching_new(self, old_filename: str, new_filenames: List[str]) -> Optional[str]:
        """Find the corresponding reference fixed (_NEW) filename for an OLD file."""
        candidates = [
            old_filename.replace("_OLD", "_NEW"),
            old_filename.replace(".OLD.", ".NEW."),
            old_filename.replace(".OLD_", "_NEW."),
        ]
        matches = [c for c in candidates if c in new_filenames]
        if len(matches) > 1:
            raise AmbiguousMatchError(
                f"Multiple NEW candidates for '{old_filename}': {matches}"
            )
        return matches[0] if matches else None

    def _find_matching_diff(self, old_filename: str, diff_filenames: List[str]) -> Optional[str]:
        """Find the corresponding .diff file for an OLD source file."""
        cand1 = None
        m1 = re.search(r"^(.*?)_OLD(-\d+)?(?:\.[^.]+)+$", old_filename)
        if m1:
            base, idx = m1.group(1), m1.group(2)
            cand1 = f"{base}{idx if idx else ''}.diff"

        cand2 = None
        m2 = re.search(r"^(.*?)\.OLD(?:\.[^.]+)+$", old_filename)
        if m2:
            cand2 = f"{m2.group(1)}.diff"

        cand3 = None
        m3 = re.search(r"^(.*?)\.OLD_.*$", old_filename)
        if m3:
            cand3 = f"{m3.group(1)}.diff"

        candidates = [c for c in [cand1, cand2, cand3] if c]
        seen = []
        for cand in candidates:
            if cand in diff_filenames and cand not in seen:
                seen.append(cand)

        if len(seen) > 1:
            raise AmbiguousMatchError(
                f"Multiple diff candidates for '{old_filename}': {seen}"
            )
        return seen[0] if seen else None

    def _parse_readme(self, readme_path: Path) -> Dict[str, Optional[str]]:
        """Extract structured metadata from README.txt or README.md if present."""
        data: Dict[str, Optional[str]] = {
            "affected_version": None,
            "commit_hash": None,
            "repository_url": None,
            "sanitizer_issue": None,
        }

        if not readme_path.exists() or not readme_path.is_file():
            return data

        try:
            content = readme_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return data

        lines = [line.strip() for line in content.splitlines()]

        for i, line in enumerate(lines):
            low = line.lower()
            if "affected version" in low:
                for j in range(i + 1, min(i + 4, len(lines))):
                    if lines[j] and not lines[j].endswith(":"):
                        data["affected_version"] = lines[j]
                        break

            if "commit hash" in low or "commit" in low:
                for j in range(i + 1, min(i + 4, len(lines))):
                    val = lines[j]
                    if val and re.match(r"^[0-9a-fA-F]{7,40}$", val):
                        data["commit_hash"] = val
                        break

            if "source code" in low:
                for j in range(i + 1, min(i + 4, len(lines))):
                    val = lines[j]
                    if "http" in val or "git clone" in val:
                        url_match = re.search(r"(https?://[^\s]+)", val)
                        if url_match:
                            data["repository_url"] = url_match.group(1).rstrip(".git")
                        else:
                            data["repository_url"] = val
                        break

            if "addresssanitizer" in low or "assertion" in low or "error:" in low:
                if not data["sanitizer_issue"]:
                    data["sanitizer_issue"] = line

        return data

    def load_case(self, case_query: Union[str, Path]) -> CaseMetadata:
        """Load and validate a benchmark case directory."""
        case_dir = self.locate_case_dir(case_query)
        case_id = case_dir.name
        project = case_dir.parent.name

        entries = sorted([f for f in case_dir.iterdir() if f.is_file()], key=lambda p: p.name)
        file_names = [f.name for f in entries]

        old_names = [f for f in file_names if self._is_old_file(f)]
        new_names = [f for f in file_names if self._is_new_file(f)]
        diff_names = [f for f in file_names if f.endswith(".diff")]

        if not old_names:
            raise MissingFileError(
                f"Case directory '{case_dir}' contains no vulnerable (_OLD) source files."
            )

        matched_new_set = set()
        matched_diff_set = set()
        source_pairs: List[SourcePair] = []

        cve_found: Optional[str] = self._extract_cve(case_id)
        cwe_found: Optional[str] = None

        for old_name in old_names:
            matched_new_name = self._find_matching_new(old_name, new_names)
            if not matched_new_name:
                raise MissingFileError(
                    f"Vulnerable source file '{old_name}' has no matching reference fixed "
                    f"counterpart in '{case_dir}'"
                )
            matched_new_set.add(matched_new_name)

            matched_diff_name = self._find_matching_diff(old_name, diff_names)
            if matched_diff_name:
                matched_diff_set.add(matched_diff_name)

            pair_cve = self._extract_cve(old_name) or cve_found
            pair_cwe = self._extract_cwe(old_name)
            if pair_cwe and not cwe_found:
                cwe_found = pair_cwe

            source_pairs.append(
                SourcePair(
                    file_name=self._extract_logical_name(old_name),
                    old_path=case_dir / old_name,
                    new_path=case_dir / matched_new_name,
                    diff_path=(case_dir / matched_diff_name) if matched_diff_name else None,
                    cve_id=pair_cve,
                    cwe_id=pair_cwe,
                )
            )

        unmatched_new = set(new_names) - matched_new_set
        if unmatched_new:
            raise MissingFileError(
                f"Fixed source file(s) without matching vulnerable counterpart in '{case_dir}': "
                f"{sorted(unmatched_new)}"
            )

        diff_paths = [case_dir / d for d in diff_names]

        readme_path: Optional[Path] = None
        for candidate_readme in ["README.txt", "README.md", "README"]:
            p = case_dir / candidate_readme
            if p.exists() and p.is_file():
                readme_path = p
                break

        readme_data = self._parse_readme(readme_path) if readme_path else {}

        standard_names = set(old_names) | set(new_names) | set(diff_names)
        if readme_path:
            standard_names.add(readme_path.name)

        extra_files = [
            case_dir / f for f in file_names if f not in standard_names
        ]

        return CaseMetadata(
            case_id=case_id,
            project=project,
            case_dir=case_dir,
            cve_id=cve_found,
            cwe_id=cwe_found,
            source_pairs=source_pairs,
            diff_files=diff_paths,
            readme_path=readme_path,
            affected_version=readme_data.get("affected_version"),
            commit_hash=readme_data.get("commit_hash"),
            repository_url=readme_data.get("repository_url"),
            sanitizer_issue=readme_data.get("sanitizer_issue"),
            extra_files=extra_files,
        )


def build_cli_parser() -> argparse.ArgumentParser:
    """Construct command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Inspect and load Vul4C benchmark vulnerability cases safely.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "case",
        help="Case identifier or relative project path (e.g. 'audiofile/CVE-2017-6832' or 'CVE-2017-6832')",
    )
    parser.add_argument(
        "--benchmark-root",
        "-b",
        default=os.environ.get("VUL4C_BENCHMARK_ROOT", "/home/rohit_kumawat/SoK/Vul4C-Benchmark"),
        help="Root directory of the Vul4C benchmark",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output case metadata in JSON format",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint for case manager."""
    parser = build_cli_parser()
    args = parser.parse_args(argv)

    try:
        manager = CaseManager(benchmark_root=args.benchmark_root)
        case_data = manager.load_case(args.case)

        if args.json:
            print(json.dumps(case_data.to_dict(), indent=2))
        else:
            print("=" * 72)
            print(f"Vul4C Case Report: {case_data.case_id} (Project: {case_data.project})")
            print("=" * 72)
            print(f"Directory:           {case_data.case_dir}")
            print(f"CVE ID:              {case_data.cve_id or 'N/A'}")
            print(f"CWE ID:              {case_data.cwe_id or 'N/A'}")
            print(f"Affected Version:    {case_data.affected_version or 'N/A'}")
            print(f"Commit Hash:         {case_data.commit_hash or 'N/A'}")
            print(f"Repository URL:      {case_data.repository_url or 'N/A'}")
            if case_data.sanitizer_issue:
                print(f"Sanitizer Issue:     {case_data.sanitizer_issue}")
            print(f"README:              {case_data.readme_path or 'N/A'}")
            print(f"Source Pairs:        {len(case_data.source_pairs)}")
            print()
            print("Discovered Source Pairs:")
            for idx, pair in enumerate(case_data.source_pairs, 1):
                print(f"  [{idx}] Logical Name: {pair.file_name}")
                print(f"      OLD (Vulnerable): {pair.old_path.name}")
                print(f"      NEW (Reference):  {pair.new_path.name}")
                print(f"      Diff File:        {pair.diff_path.name if pair.diff_path else 'None'}")
                print(f"      Full OLD Path:    {pair.old_path}")
                print(f"      Full NEW Path:    {pair.new_path}")
            if case_data.extra_files:
                print()
                print("Extra Files (Not Executed):")
                for extra in case_data.extra_files:
                    print(f"  - {extra.name} ({extra})")
            print("=" * 72)
        return 0

    except CaseManagerError as err:
        sys.stderr.write(f"Error: {err}\n")
        return 1
    except Exception as err:
        sys.stderr.write(f"Unexpected error: {err}\n")
        return 2


if __name__ == "__main__":
    sys.exit(main())
