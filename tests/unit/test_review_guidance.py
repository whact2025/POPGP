import re
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


def _uv_commands(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    return {
        match.group(1).strip()
        for match in re.finditer(
            r"(?m)^\s*(?:run:\s*)?(uv (?:sync|run) [^\r\n]+)$", text
        )
    }


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
    matrix_ids = set(re.findall(r"GATE-[A-Z-]+", matrix))
    registry_ids = set(re.findall(r"GATE-[A-Z-]+", registry))
    test_nodes = re.findall(
        r"`(tests/(?:unit|scientific)/[^`]+\.py)::(test_[a-z0-9_]+)`",
        registry,
    )

    assert matrix_ids
    assert registry_ids == matrix_ids
    assert "statistic before → after" in registry.lower()
    assert "demonstrated failing negative" in governance
    for relative_path, function_name in test_nodes:
        test_source = (ROOT / relative_path).read_text(encoding="utf-8")
        assert f"def {function_name}(" in test_source
