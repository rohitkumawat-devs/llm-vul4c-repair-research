# LLM-Based C/C++ Vulnerability Repair Research

## Overview

This project aims to build a reproducible pipeline for evaluating
LLM-generated repairs for C/C++ vulnerabilities using the Vul4C benchmark.

## Research Objectives

- Evaluate vulnerability detection.
- Generate candidate code repairs using an LLM.
- Check compilation and functional correctness.
- Evaluate removal of the original vulnerability.
- Investigate potential security regressions.
- Record results using reproducible evaluation metrics.

## Planned Architecture

Vul4C Benchmark -> Case Manager -> Prompt Generator -> LLM Client
-> Patch Manager -> Compilation -> Functional Testing
-> Security Validation -> Results and Metrics

## Project Status

Initial setup. The architecture and pipeline are being developed.

## Reference Benchmark

https://github.com/SoK-Vul4C/SoK/tree/main/Vul4C-Benchmark

The benchmark authors and applicable license will be acknowledged
in the final documentation.

## Important

This project is experimental. Compilation success alone does not
establish that a generated patch is secure.
