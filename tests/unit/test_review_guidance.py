import ast
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

QUALITY_COMMANDS = {
    "uv sync --frozen",
    "uv run ruff check .",
    "uv run python scripts/check_tex.py",
    "uv run pytest -q",
    "uv run python -m examples.physics_qg.chain_1d",
    "uv run python -m examples.physics_qg.grid_2d",
    "uv run python -m examples.physics_qg.gravity_well",
    "uv run python -m examples.physics_qg.source_law",
    "uv run python -m examples.physics_qg.source_law_many_body",
    "uv run python -m examples.physics_qg.ca_model",
    "uv run python scripts/check_validation_artifacts.py",
}

UNGATED_MATRIX_ROWS = {
    "Refinement convergence",
    "Clock/spatial consistency",
    "Lorentz recovery",
    "Causal completeness",
    "Closure / conservation",
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
    start = next(index for index, line in enumerate(lines) if line.startswith(header))
    rows = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip("|").split("|")])
    return rows


def _has_negative_control_marker(path: Path, function_name: str) -> bool:
    module = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    )
    return any(
        ast.unparse(decorator).startswith("pytest.mark.negative_control")
        for decorator in function.decorator_list
    )


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
            if "authoritative" in lowered and (
                "quality suite" in lowered or "pre-freeze" in lowered
            ):
                matches.append((path, sentence))
    return matches


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

    authority_sentences = _quality_authority_sentences()
    assert authority_sentences
    for path, sentence in authority_sentences:
        assert ".github/workflows/ci.yml" in sentence, f"{path}: {sentence}"


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

    collected = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    collected_match = re.search(r"(\d+) tests? collected", collected.stdout)
    assert collected_match is not None
    assert int(recorded_match.group(1)) == int(collected_match.group(1))
