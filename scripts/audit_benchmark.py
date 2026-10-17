"""Audit Vul4C Benchmark Coverage and Validate Cases.

This script performs a complete static audit of the Vul4C benchmark:
- Discovers all case directories.
- Validates source file pairing (OLD/NEW/diff).
- Aggregates metrics by project, CWE, multi-pair status, and metadata availability.
- Evaluates pilot sets and split distributions.
- Outputs structured metadata report to results/ (never storing full source code).
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.case_manager import (
    CaseManager,
    CaseManagerError,
    CaseMetadata,
    SourcePair,
)
from pipeline.dataset_builder import (
    DatasetBuilder,
    DatasetBuilderError,
    ValidationError,
)

def audit_benchmark(
    benchmark_root: Path,
) -> Dict[str, Any]:
    manager = CaseManager(benchmark_root)
    builder = DatasetBuilder(benchmark_root)

    # Discover candidate directories
    all_case_dirs = sorted([
        d for d in benchmark_root.glob("*/*")
        if d.is_dir() and not d.name.startswith(".") and not d.parent.name.startswith(".")
    ])

    results: List[Dict[str, Any]] = []
    success_cases: List[CaseMetadata] = []
    failed_cases: List[Dict[str, str]] = []

    cwe_counter = Counter()
    project_counter = Counter()
    multi_pair_cases: List[Dict[str, Any]] = []
    single_pair_cases: List[Dict[str, Any]] = []

    total_pairs = 0
    missing_cwe_cases: List[str] = []
    missing_sanitizer_cases: List[str] = []
    missing_commit_cases: List[str] = []

    for case_dir in all_case_dirs:
        case_query = f"{case_dir.parent.name}/{case_dir.name}"
        try:
            meta = manager.load_case(case_query)
            # Validate all source pairs
            for pair in meta.source_pairs:
                builder.validate_source_pair(pair)

            success_cases.append(meta)
            num_pairs = len(meta.source_pairs)
            total_pairs += num_pairs
            project_counter[meta.project] += 1
            cwe_tag = meta.cwe_id or "Unknown-CWE"
            cwe_counter[cwe_tag] += 1

            if not meta.cwe_id:
                missing_cwe_cases.append(case_query)
            if not meta.sanitizer_issue:
                missing_sanitizer_cases.append(case_query)
            if not meta.commit_hash:
                missing_commit_cases.append(case_query)

            case_info = {
                "case_id": meta.case_id,
                "project": meta.project,
                "cve_id": meta.cve_id,
                "cwe_id": meta.cwe_id,
                "source_pairs_count": num_pairs,
                "source_files": [p.file_name for p in meta.source_pairs],
                "diff_files_count": len(meta.diff_files),
                "has_readme": meta.readme_path is not None,
                "affected_version": meta.affected_version,
                "commit_hash": meta.commit_hash,
                "sanitizer_issue": meta.sanitizer_issue,
            }
            results.append(case_info)

            if num_pairs > 1:
                multi_pair_cases.append(case_info)
            else:
                single_pair_cases.append(case_info)

        except Exception as err:
            failed_cases.append({
                "case_query": case_query,
                "error": str(err),
                "error_type": type(err).__name__,
            })

    report = {
        "audit_timestamp": datetime.now(timezone.utc).isoformat(),
        "benchmark_root": str(benchmark_root.resolve()),
        "summary": {
            "total_discovered_cases": len(all_case_dirs),
            "total_successfully_loaded": len(success_cases),
            "total_failed_or_skipped": len(failed_cases),
            "total_source_pairs": total_pairs,
            "single_pair_cases_count": len(single_pair_cases),
            "multi_pair_cases_count": len(multi_pair_cases),
            "total_projects": len(project_counter),
            "total_unique_cwes": len(cwe_counter),
        },
        "counts_by_project": dict(project_counter.most_common()),
        "counts_by_cwe": dict(cwe_counter.most_common()),
        "multi_pair_cases": [
            {
                "case_id": c["case_id"],
                "project": c["project"],
                "source_pairs_count": c["source_pairs_count"],
                "files": c["source_files"],
            }
            for c in multi_pair_cases
        ],
        "metadata_completeness": {
            "cases_with_cwe": len(success_cases) - len(missing_cwe_cases),
            "cases_missing_cwe": len(missing_cwe_cases),
            "cases_with_commit_hash": len(success_cases) - len(missing_commit_cases),
            "cases_missing_commit_hash": len(missing_commit_cases),
            "cases_with_sanitizer_error": len(success_cases) - len(missing_sanitizer_cases),
            "cases_missing_sanitizer_error": len(missing_sanitizer_cases),
        },
        "failed_cases": failed_cases,
        "cases": results,
    }
    return report

DEFAULT_PILOT_CASES = [
    "audiofile/CVE-2017-6832",  # CWE-20, single pair, input validation
    "audiofile/CVE-2017-6834",  # CWE-119, 3 pairs (multi-file), heap-buffer-overflow
    "libjpeg/CVE-2017-15232",   # CWE-476, 2 pairs, NULL pointer dereference
    "jasper/CVE-2016-8691",     # CWE-369, 1 pair, divide-by-zero
    "libtiff/CVE-2016-9532",    # CWE-125, 1 pair (dot naming convention), out-of-bounds read
    "libdwarf/CVE-2016-9275",   # CWE-787, 8 pairs (multi-file), out-of-bounds write
    "binutils/CVE-2017-9040",   # CWE-476, 1 pair, NULL pointer dereference
    "ngiflib/CVE-2019-16347",   # CWE-787, 1 pair, heap out-of-bounds write
]


def evaluate_pilot_split(
    benchmark_root: Path,
    pilot_cases: List[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, Any]:
    builder = DatasetBuilder(
        benchmark_root=benchmark_root,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    samples, manifest = builder.build_dataset(pilot_cases)

    # Check for cross-split leakage
    case_splits = defaultdict(set)
    for s in samples:
        cid = s.metadata["case_id"]
        case_splits[cid].add(s.metadata["split"])

    leakage_detected = any(len(splits) > 1 for splits in case_splits.values())

    return {
        "pilot_cases_count": len(pilot_cases),
        "pilot_cases": pilot_cases,
        "seed": seed,
        "split_ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "case_counts_by_split": manifest.case_counts_by_split,
        "sample_counts_by_split_and_task": manifest.sample_counts_by_split_and_task,
        "total_samples": len(samples),
        "leakage_detected": leakage_detected,
        "is_all_splits_nonempty": (
            manifest.case_counts_by_split["train"] > 0 and
            manifest.case_counts_by_split["validation"] > 0 and
            manifest.case_counts_by_split["test"] > 0
        ),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit Vul4C Benchmark coverage and evaluate pilot dataset splits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--benchmark-root",
        "-b",
        default=os.environ.get("VUL4C_BENCHMARK_ROOT", "/home/rohit_kumawat/SoK/Vul4C-Benchmark"),
        help="Root directory of the Vul4C benchmark",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="results/benchmark_coverage_report.json",
        help="Path to save metadata coverage report JSON",
    )
    parser.add_argument(
        "--pilot-cases",
        help="Comma-separated list of pilot case queries (defaults to standard 8-case pilot)",
    )
    args = parser.parse_args(argv)

    bench_path = Path(args.benchmark_root).resolve()
    if not bench_path.exists():
        sys.stderr.write(f"Error: Benchmark root does not exist: {bench_path}\n")
        return 1

    print("=" * 72)
    print("Vul4C Benchmark Coverage & Dataset Readiness Audit")
    print("=" * 72)
    print(f"Benchmark Root: {bench_path}")
    print("Auditing all benchmark cases across all project directories...")

    report = audit_benchmark(bench_path)
    summary = report["summary"]

    print()
    print("Audit Summary Results:")
    print(f"  - Total Discovered Cases:       {summary['total_discovered_cases']}")
    print(f"  - Successfully Loaded & Valid:  {summary['total_successfully_loaded']}")
    print(f"  - Failed / Rejected Cases:      {summary['total_failed_or_skipped']}")
    print(f"  - Total Available Source Pairs: {summary['total_source_pairs']}")
    print(f"  - Single-Pair Cases:            {summary['single_pair_cases_count']}")
    print(f"  - Multi-Pair Cases:             {summary['multi_pair_cases_count']}")
    print(f"  - Total Distinct Projects:      {summary['total_projects']}")
    print(f"  - Total Distinct CWE Types:     {summary['total_unique_cwes']}")

    print()
    print("Top 8 Projects by Case Count:")
    for proj, cnt in list(report["counts_by_project"].items())[:8]:
        print(f"  - {proj:18s}: {cnt:3d} cases")

    print()
    print("Top 8 CWEs by Case Count:")
    for cwe, cnt in list(report["counts_by_cwe"].items())[:8]:
        print(f"  - {cwe:18s}: {cnt:3d} cases")

    # Evaluate pilot set
    pilot_list = [c.strip() for c in args.pilot_cases.split(",") if c.strip()] if args.pilot_cases else DEFAULT_PILOT_CASES

    print()
    print("=" * 72)
    print(f"Evaluating Configurable Pilot Set ({len(pilot_list)} distinct cases)")
    print("=" * 72)
    pilot_eval = evaluate_pilot_split(bench_path, pilot_list)

    cc = pilot_eval["case_counts_by_split"]
    sc = pilot_eval["sample_counts_by_split_and_task"]
    print(f"Pilot Case Counts:   Train: {cc['train']} | Validation: {cc['validation']} | Test: {cc['test']}")
    print(f"Pilot Sample Counts: Train: {sc['train']['total']} | Validation: {sc['validation']['total']} | Test: {sc['test']['total']} (Total: {pilot_eval['total_samples']})")
    print(f"Cross-Split Leakage: {'YES (FAIL)' if pilot_eval['leakage_detected'] else 'NO (PASSED - 0% leakage)'}")
    print(f"All Splits Nonempty: {'YES' if pilot_eval['is_all_splits_nonempty'] else 'NO'}")

    report["pilot_evaluation"] = pilot_eval

    # Save output report
    out_file = Path(args.output).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print()
    print(f"Audit report saved to: {out_file}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
