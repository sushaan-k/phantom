"""Tests for the ATLAS taxonomy module."""

from __future__ import annotations

import pytest

from phantom.atlas.taxonomy import ATLASTaxonomy
from phantom.exceptions import TaxonomyError
from phantom.models import Severity


class TestATLASTaxonomy:
    """Tests for ATLASTaxonomy loading and querying."""

    def test_load_default(self, taxonomy: ATLASTaxonomy) -> None:
        assert len(taxonomy.all_technique_ids) > 0
        assert len(taxonomy.all_categories) > 0

    def test_get_technique_by_id(self, taxonomy: ATLASTaxonomy) -> None:
        tech = taxonomy.get_technique("AML.T0051")
        assert tech is not None
        assert tech.name == "LLM Prompt Injection"
        assert tech.tactic == "Initial Access"

    def test_get_subtechnique(self, taxonomy: ATLASTaxonomy) -> None:
        tech = taxonomy.get_technique("AML.T0051.001")
        assert tech is not None
        sub = tech.get_subtechnique("AML.T0051.001")
        assert sub is not None
        assert sub.name == "Indirect Prompt Injection"

    def test_get_nonexistent_technique(self, taxonomy: ATLASTaxonomy) -> None:
        assert taxonomy.get_technique("AML.T9999") is None

    def test_get_techniques_for_category(self, taxonomy: ATLASTaxonomy) -> None:
        techs = taxonomy.get_techniques_for_category("prompt_injection")
        assert len(techs) > 0
        assert any(t.id == "AML.T0051" for t in techs)

    def test_get_techniques_for_unknown_category(self, taxonomy: ATLASTaxonomy) -> None:
        assert taxonomy.get_techniques_for_category("nonexistent") == []

    def test_get_default_severity(self, taxonomy: ATLASTaxonomy) -> None:
        assert taxonomy.get_default_severity("AML.T0054") == Severity.CRITICAL
        assert taxonomy.get_default_severity("AML.T0051") == Severity.HIGH
        assert taxonomy.get_default_severity("AML.T9999") == Severity.MEDIUM

    def test_get_mitigations(self, taxonomy: ATLASTaxonomy) -> None:
        mitigations = taxonomy.get_mitigations("AML.T0051")
        assert len(mitigations) > 0
        assert any("input" in m.lower() for m in mitigations)

    def test_missing_data_file(self) -> None:
        t = ATLASTaxonomy(data_path="/nonexistent/path.json")
        with pytest.raises(TaxonomyError, match="not found"):
            t.load()

    def test_all_categories(self, taxonomy: ATLASTaxonomy) -> None:
        cats = taxonomy.all_categories
        assert "prompt_injection" in cats
        assert "goal_hijacking" in cats
        assert "data_exfiltration" in cats
        assert "denial_of_service" in cats

    def test_auto_load_on_query(self) -> None:
        t = ATLASTaxonomy()
        tech = t.get_technique("AML.T0051")
        assert tech is not None
