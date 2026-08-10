from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_energy_conservation_claim_distinguishes_global_sum_from_profile() -> None:
    framework_md = (ROOT / "docs" / "framework.md").read_text(encoding="utf-8")
    framework_tex = (ROOT / "docs" / "framework.tex").read_text(encoding="utf-8")
    falsification_matrix = (
        ROOT / "docs" / "scientific_hardening" / "FALSIFICATION_MATRIX.md"
    ).read_text(encoding="utf-8")
    example = (
        ROOT / "examples" / "physics_qg" / "source_law_many_body" / "__main__.py"
    ).read_text(encoding="utf-8")

    combined = framework_md + framework_tex + falsification_matrix
    assert "conservation of the audited local-energy profile" not in combined
    assert "conserves and spreads the local-energy profile" not in combined
    assert "audited energy profile evolves conservatively" not in combined
    assert "global Hamiltonian expectation is conserved" in framework_md
    assert "global Hamiltonian expectation" in framework_tex
    assert "global Hamiltonian expectation is conserved" in falsification_matrix
    assert "No discrete continuity current or local conservation law" in framework_md
    assert "no discrete continuity current or local conservation law" in framework_tex
    assert "no discrete continuity current or local conservation law" in falsification_matrix
    assert 'set_title("Global energy conserved; profile spreads")' in example
    assert '"name": "localized_energy_is_globally_conserved_and_spreads"' in example
