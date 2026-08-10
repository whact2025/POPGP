import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _assert_no_unqualified_local_conservation(text: str) -> None:
    normalized = re.sub(r"\s+", " ", text)
    for sentence in re.split(r"(?<=[.!?])\s+", normalized):
        lowered = sentence.lower()
        refers_to_local_quantity = re.search(
            r"\b(local|site|profile|decomposition)\b", lowered
        )
        claims_conservation = re.search(r"\bconserv(?:e[ds]?|ation|atively)\b", lowered)
        if not (refers_to_local_quantity and claims_conservation):
            continue
        if "local conservation law" in lowered and re.search(
            r"\b(no|not|without)\b", lowered
        ):
            continue
        global_qualifier = "global" in lowered
        scope_qualifier = "spread" in lowered
        assert global_qualifier and scope_qualifier, sentence


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
    assert "global microscopic energy is conserved" in falsification_matrix
    assert "No discrete continuity current or local conservation law" in framework_md
    assert "no discrete continuity current or local conservation law" in framework_tex
    assert "no discrete continuity current or local conservation law" in falsification_matrix
    assert 'set_title("Global energy conserved; profile spreads")' in example
    assert '"name": "local_energy_decomposition_consistency_and_spreading"' in example
    assert "generator/observable consistency drift" in example
    assert "energy-conservation drift" not in example

    live_claim_documents = [
        ROOT / "README.md",
        ROOT / "docs" / "framework.md",
        ROOT / "docs" / "framework.tex",
        *sorted((ROOT / "docs" / "scientific_hardening").glob("*.md")),
        ROOT / "examples" / "physics_qg" / "source_law_many_body" / "README.md",
    ]
    for path in live_claim_documents:
        _assert_no_unqualified_local_conservation(path.read_text(encoding="utf-8"))


def test_power_law_residual_scale_is_not_labelled_as_sampling_uncertainty() -> None:
    source_law_files = [
        ROOT / "popgp" / "diagnostics.py",
        ROOT / "examples" / "physics_qg" / "source_law" / "__main__.py",
        ROOT / "examples" / "physics_qg" / "source_law_many_body" / "__main__.py",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in source_law_files)

    assert "slope_standard_error" not in combined
    assert "slope_residual_scale" in combined
    assert "residual scale=" in combined


def test_framework_locality_status_discloses_qcmi_gap_in_both_sources() -> None:
    framework_md = (ROOT / "docs" / "framework.md").read_text(encoding="utf-8")
    framework_tex = (ROOT / "docs" / "framework.tex").read_text(encoding="utf-8")
    expected_md = (
        r"| Emergent locality via QCMI-screened mutual information \(I_{ij}\) and graph "
        r"metric \(d_G\) | Definition / partial prototype | Pairwise MI and blind graph "
        r"routing are implemented; QCMI screening is not. The tested Hamiltonians contain "
        r"chain/grid interaction graphs. | Fails if non-geometric controls produce stable "
        r"geometric declarations or encoded locality is not robustly recovered. |"
    )
    expected_tex = (
        r"\item[\textbf{Emergent locality} $I_{ij}, d_G$] \textit{(Definition / partial "
        r"prototype)} Pairwise mutual information and blind multi-hop graph routing are "
        r"implemented; QCMI screening is not. The tested Hamiltonians contain chain/grid "
        r"interaction graphs. \textbf{Failure mode:} Falsified if non-geometric controls "
        r"produce stable geometric declarations or encoded locality is not robustly recovered."
    )

    for source, expected in ((framework_md, expected_md), (framework_tex, expected_tex)):
        locality_rows = [
            line.strip()
            for line in source.splitlines()
            if not line.lstrip().startswith("%") and "Emergent locality" in line
        ]
        assert locality_rows == [expected]
