import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

QUALITY_COMMANDS = {
    "uv sync --frozen --no-editable",
    "uv run --frozen --no-editable ruff check .",
    "uv run --frozen --no-editable python scripts/check_tex.py",
    "uv run --frozen --no-editable python -m pytest -q -p no:cacheprovider",
    "uv run --frozen --no-editable python -m examples.physics_qg.chain_1d",
    "uv run --frozen --no-editable python -m examples.physics_qg.grid_2d",
    "uv run --frozen --no-editable python -m examples.physics_qg.gravity_well",
    "uv run --frozen --no-editable python -m examples.physics_qg.source_law",
    "uv run --frozen --no-editable python -m examples.physics_qg.source_law_many_body",
    "uv run --frozen --no-editable python -m examples.physics_qg.ca_model",
    "uv run --frozen --no-editable python scripts/check_validation_artifacts.py "
    "--enforce-change-boundary",
}

UNGATED_MATRIX_ROWS = {
    "Refinement convergence",
    "Clock/spatial consistency",
    "Lorentz recovery",
    "Causal completeness",
    "Closure / conservation",
}

CANONICAL_QUALITY_AUTHORITY_CLAUSES = {
    "`.github/workflows/ci.yml`, which is the authoritative quality suite",
    "`.github/workflows/ci.yml` is authoritative",
}


def _uv_commands(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    return {
        match.group(1).strip()
        for match in re.finditer(
            r"(?m)^\s*(?:run:\s*)?(uv (?:sync|run) [^\r\n]+)$", text
        )
    }


def _markdown_table(text: str, header: str) -> list[list[str]]:
    lines = text.splitlines()
    header_indices = [
        index for index, line in enumerate(lines) if line.startswith(header)
    ]
    assert len(header_indices) == 1
    start = header_indices[0]
    assert start + 1 < len(lines)
    assert re.fullmatch(r"\|(?:\s*:?-+:?\s*\|)+", lines[start + 1])

    end = start + 2
    while end < len(lines) and lines[end].startswith("|"):
        end += 1
    pipe_lines = {index for index, line in enumerate(lines) if line.startswith("|")}
    assert pipe_lines == set(range(start, end))
    return [
        [cell.strip() for cell in line.strip("|").split("|")]
        for line in lines[start + 2 : end]
    ]


def _has_negative_control_marker(path: Path, function_name: str) -> bool:
    module = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    )
    decorators = {ast.unparse(decorator) for decorator in function.decorator_list}
    is_control = "pytest.mark.negative_control" in decorators
    is_skipped = any(
        decorator == "pytest.mark.skip"
        or decorator.startswith("pytest.mark.skipif(")
        for decorator in decorators
    )
    return is_control and not is_skipped


def _quality_authority_sentences() -> list[tuple[Path, str]]:
    paths = [
        ROOT / "README.md",
        *sorted((ROOT / "docs" / "governance").glob("*.md")),
        *sorted((ROOT / "docs" / "reviews").glob("*.md")),
    ]
    matches = []
    for path in paths:
        normalized = re.sub(r"\s+", " ", path.read_text(encoding="utf-8"))
        for sentence in re.split(r"(?<=[.!?])\s+", normalized):
            lowered = sentence.lower()
            if re.search(r"\b(authoritative|canonical|official|primary)\b", lowered) and (
                "suite" in lowered or ".github/workflows/ci.yml" in lowered
            ):
                matches.append((path, sentence))
    return matches


def _is_canonical_quality_authority_sentence(sentence: str) -> bool:
    return any(
        clause in sentence for clause in CANONICAL_QUALITY_AUTHORITY_CLAUSES
    )


def _collected_test_count(root: Path) -> int:
    collected = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert collected.returncode == 0, (
        f"pytest collection failed under {root}\n"
        f"stdout:\n{collected.stdout}\n"
        f"stderr:\n{collected.stderr}"
    )
    collected_match = re.search(r"(\d+) tests? collected", collected.stdout)
    assert collected_match is not None, collected.stdout
    return int(collected_match.group(1))


def _assert_no_skip_or_xfail_markers(root: Path) -> None:
    marked = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-m",
            "skip or xfail",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    output = f"{marked.stdout}\n{marked.stderr}"
    assert marked.returncode == 5 and "no tests collected" in output, (
        "the documented passed count requires a suite with no skip/xfail markers\n"
        f"{output}"
    )


def test_documented_quality_commands_match_authoritative_ci() -> None:
    workflow = ROOT / ".github" / "workflows" / "ci.yml"
    readme = ROOT / "README.md"
    runbook = ROOT / "docs" / "reviews" / "LAUNCH_INDEPENDENT_REVIEW.md"
    governance = (
        ROOT / "docs" / "governance" / "AGENT_REVIEW_WORKFLOW.md"
    ).read_text(encoding="utf-8")

    assert _uv_commands(workflow) == QUALITY_COMMANDS
    assert _uv_commands(readme) == QUALITY_COMMANDS
    assert _uv_commands(runbook) == QUALITY_COMMANDS
    assert ".github/workflows/ci.yml`, which is the authoritative quality" in governance
    assert "`.github/workflows/ci.yml` is authoritative" in runbook.read_text(
        encoding="utf-8"
    )

    authority_sentences = _quality_authority_sentences()
    assert len(authority_sentences) == len(CANONICAL_QUALITY_AUTHORITY_CLAUSES)
    for path, sentence in authority_sentences:
        assert _is_canonical_quality_authority_sentence(sentence), (
            f"{path}: {sentence}"
        )


def test_competing_quality_authority_sentence_is_rejected() -> None:
    evasion = (
        "Although .github/workflows/ci.yml exists, the authoritative quality suite "
        "is the README Quick start."
    )

    assert not _is_canonical_quality_authority_sentence(evasion)


def test_markdown_table_parser_rejects_truncated_coverage() -> None:
    truncated = "\n".join(
        [
            "| Hypothesis | Status |",
            "|---|---|",
            "| registered | active |",
            "",
            "| hidden by whitespace | active |",
        ]
    )

    with pytest.raises(AssertionError):
        _markdown_table(truncated, "| Hypothesis |")


def test_negative_control_marker_must_be_exact_and_unskipped(
    tmp_path: Path,
) -> None:
    module = tmp_path / "test_controls.py"
    module.write_text(
        "\n".join(
            [
                "import pytest",
                "",
                "@pytest.mark.negative_control",
                "def test_exact(): pass",
                "",
                "@pytest.mark.negative_control_pending",
                "def test_pending(): pass",
                "",
                "@pytest.mark.skip",
                "@pytest.mark.negative_control",
                "def test_skipped(): pass",
            ]
        ),
        encoding="utf-8",
    )

    assert _has_negative_control_marker(module, "test_exact")
    assert not _has_negative_control_marker(module, "test_pending")
    assert not _has_negative_control_marker(module, "test_skipped")


def test_review_identity_schema_has_typed_independence_declaration() -> None:
    identity = (
        ROOT / "docs" / "governance" / "REVIEWER_IDENTITY.md"
    ).read_text(encoding="utf-8")
    template = (
        ROOT / "docs" / "templates" / "INDEPENDENT_REVIEW_TEMPLATE.md"
    ).read_text(encoding="utf-8")
    required_fields = {
        "shared_operator",
        "shared_session",
        "shared_orchestrator",
        "builder_model_identity",
        "reviewer_model_differs_from_builder",
        "external_scientific_validation",
    }

    for field in required_fields:
        assert f"  {field}:" in identity
        assert f"  {field}:" in template
    assert "no agent review may set it true" in identity


def test_every_declared_gate_has_an_executable_negative_control() -> None:
    matrix = (
        ROOT / "docs" / "scientific_hardening" / "FALSIFICATION_MATRIX.md"
    ).read_text(encoding="utf-8")
    registry = (
        ROOT / "docs" / "scientific_hardening" / "GATE_TEST_REGISTRY.md"
    ).read_text(encoding="utf-8")
    governance = (
        ROOT / "docs" / "governance" / "AGENT_REVIEW_WORKFLOW.md"
    ).read_text(encoding="utf-8")
    matrix_rows = _markdown_table(matrix, "| Hypothesis |")
    registry_rows = _markdown_table(registry, "| Gate ID |")
    matrix_gates: dict[str, str] = {}
    for row in matrix_rows:
        assert len(row) == 6
        hypothesis = row[0]
        gate_ids = re.findall(r"GATE-[A-Z-]+", hypothesis)
        if not gate_ids:
            assert hypothesis in UNGATED_MATRIX_ROWS
            continue
        assert len(gate_ids) == 1
        matrix_gates[gate_ids[0]] = re.sub(
            r"\s*\(`GATE-[A-Z-]+`\)$", "", hypothesis
        )

    registry_gates: dict[str, str] = {}
    for row in registry_rows:
        assert len(row) == 5
        gate_match = re.fullmatch(r"`(GATE-[A-Z-]+)`", row[0])
        assert gate_match is not None
        gate_id = gate_match.group(1)
        assert gate_id not in registry_gates
        registry_gates[gate_id] = row[1]
        test_nodes = re.findall(
            r"`(tests/(?:unit|scientific)/[^`]+\.py)::(test_[a-z0-9_]+)`",
            row[3],
        )
        assert test_nodes, f"{gate_id} has no executable negative control"
        for relative_path, function_name in test_nodes:
            test_path = ROOT / relative_path
            assert test_path.is_file()
            assert _has_negative_control_marker(test_path, function_name)

    assert matrix_gates
    assert registry_gates == matrix_gates
    assert "statistic before → after" in registry.lower()
    assert "demonstrated failing negative" in governance


def test_reproducibility_record_matches_collected_test_count() -> None:
    reproducibility = (
        ROOT / "docs" / "scientific_hardening" / "REPRODUCIBILITY.md"
    ).read_text(encoding="utf-8")
    recorded_match = re.search(
        r"\| `pytest -q` \| [^|]+ \| (\d+) passed \|",
        reproducibility,
    )
    assert recorded_match is not None

    _assert_no_skip_or_xfail_markers(ROOT)
    assert int(recorded_match.group(1)) == _collected_test_count(ROOT)


def test_collection_failure_reports_the_offending_module(tmp_path: Path) -> None:
    broken = tmp_path / "test_broken.py"
    broken.write_text(
        "raise RuntimeError('deliberate collection failure')\n",
        encoding="utf-8",
    )

    with pytest.raises(AssertionError, match="test_broken.py"):
        _collected_test_count(tmp_path)


def test_skip_marker_invalidates_documented_pass_count(tmp_path: Path) -> None:
    skipped = tmp_path / "test_skipped.py"
    skipped.write_text(
        "import pytest\n\n@pytest.mark.skip\ndef test_skipped():\n    pass\n",
        encoding="utf-8",
    )

    with pytest.raises(AssertionError, match="skip/xfail"):
        _assert_no_skip_or_xfail_markers(tmp_path)
