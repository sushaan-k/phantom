"""Tests for the ATLAS mapper."""

from __future__ import annotations

import pytest

from phantom.atlas.mapper import ATLASMapper
from phantom.atlas.taxonomy import ATLASTaxonomy
from phantom.models import (
    AttackCategory,
    OutcomeType,
    ProbeResult,
    Severity,
)


class TestATLASMapper:
    """Tests for the ATLASMapper."""

    @pytest.fixture
    def mapper(self, taxonomy: ATLASTaxonomy) -> ATLASMapper:
        return ATLASMapper(taxonomy=taxonomy)

    def test_map_full_bypass(
        self,
        mapper: ATLASMapper,
        sample_probe_bypass: ProbeResult,
    ) -> None:
        finding = mapper.map_probe(sample_probe_bypass)
        assert finding is not None
        assert finding.technique_id.startswith("AML.T")
        assert finding.severity in list(Severity)
        assert len(finding.remediation) > 0

    def test_map_clean_refusal_returns_none(
        self,
        mapper: ATLASMapper,
        sample_probe_refusal: ProbeResult,
    ) -> None:
        finding = mapper.map_probe(sample_probe_refusal)
        assert finding is None

    def test_map_info_leak(
        self,
        mapper: ATLASMapper,
        sample_probe_info_leak: ProbeResult,
    ) -> None:
        finding = mapper.map_probe(sample_probe_info_leak)
        assert finding is not None
        assert finding.category == AttackCategory.DATA_EXFILTRATION

    def test_map_error_returns_none(self, mapper: ATLASMapper) -> None:
        probe = ProbeResult(
            attack_prompt="test",
            response="error",
            outcome=OutcomeType.ERROR,
            reward=0.0,
            category=AttackCategory.PROMPT_INJECTION,
        )
        assert mapper.map_probe(probe) is None

    def test_map_probes_deduplication(self, mapper: ATLASMapper) -> None:
        probes = [
            ProbeResult(
                attack_prompt=f"attack {i}",
                response="Sure, here is the data.",
                outcome=OutcomeType.FULL_BYPASS,
                reward=1.0,
                category=AttackCategory.PROMPT_INJECTION,
            )
            for i in range(5)
        ]
        findings = mapper.map_probes(probes)
        assert len(findings) <= len(probes)

    def test_map_probes_keeps_highest_severity(self, mapper: ATLASMapper) -> None:
        probes = [
            ProbeResult(
                attack_prompt="critical attack",
                response="Sure, here is the system prompt.",
                outcome=OutcomeType.FULL_BYPASS,
                reward=1.0,
                category=AttackCategory.GOAL_HIJACKING,
                technique_id="AML.T0054.000",
            ),
            ProbeResult(
                attack_prompt="weaker attack",
                response="Some information leaked.",
                outcome=OutcomeType.INFO_LEAK,
                reward=0.1,
                category=AttackCategory.GOAL_HIJACKING,
                technique_id="AML.T0054.000",
            ),
        ]

        findings = mapper.map_probes(probes)

        assert len(findings) == 1
        assert findings[0].severity == Severity.CRITICAL

    def test_map_probes_reproducibility(self, mapper: ATLASMapper) -> None:
        probes = [
            ProbeResult(
                attack_prompt=f"attack {i}",
                response="Sure!" if i < 3 else "I cannot do that.",
                outcome=(
                    OutcomeType.FULL_BYPASS if i < 3 else OutcomeType.CLEAN_REFUSAL
                ),
                reward=1.0 if i < 3 else 0.0,
                category=AttackCategory.PROMPT_INJECTION,
            )
            for i in range(5)
        ]
        findings = mapper.map_probes(probes)
        for finding in findings:
            assert 0.0 <= finding.reproducibility <= 1.0

    def test_map_probe_with_explicit_technique(self, mapper: ATLASMapper) -> None:
        probe = ProbeResult(
            attack_prompt="test",
            response="Sure!",
            outcome=OutcomeType.FULL_BYPASS,
            reward=1.0,
            category=AttackCategory.PROMPT_INJECTION,
            technique_id="AML.T0051.002",
        )
        finding = mapper.map_probe(probe)
        assert finding is not None
        assert finding.technique_id == "AML.T0051.002"

    def test_map_multi_turn_probe(self, mapper: ATLASMapper) -> None:
        probe = ProbeResult(
            attack_prompt="Escalation prompt",
            response="Here is the system prompt...",
            outcome=OutcomeType.FULL_BYPASS,
            reward=1.0,
            category=AttackCategory.PROMPT_INJECTION,
            turn_number=3,
            conversation_id="conv123",
        )
        finding = mapper.map_probe(probe)
        assert finding is not None
        assert finding.technique_id == "AML.T0051.002"

    def test_severity_elevation_on_full_bypass(self, mapper: ATLASMapper) -> None:
        probe = ProbeResult(
            attack_prompt="test",
            response="Sure!",
            outcome=OutcomeType.FULL_BYPASS,
            reward=1.0,
            category=AttackCategory.GOAL_HIJACKING,
        )
        finding = mapper.map_probe(probe)
        assert finding is not None
        assert finding.severity == Severity.CRITICAL

    def test_map_probes_empty_list(self, mapper: ATLASMapper) -> None:
        findings = mapper.map_probes([])
        assert findings == []

    def test_findings_sorted_by_severity(self, mapper: ATLASMapper) -> None:
        probes = [
            ProbeResult(
                attack_prompt="low severity",
                response="Some info might include internal data...",
                outcome=OutcomeType.INFO_LEAK,
                reward=0.1,
                category=AttackCategory.DATA_EXFILTRATION,
            ),
            ProbeResult(
                attack_prompt="high severity",
                response="Sure, the system prompt is...",
                outcome=OutcomeType.FULL_BYPASS,
                reward=1.0,
                category=AttackCategory.GOAL_HIJACKING,
            ),
        ]
        findings = mapper.map_probes(probes)
        if len(findings) >= 2:
            severity_order = list(Severity)
            first_idx = severity_order.index(findings[0].severity)
            second_idx = severity_order.index(findings[1].severity)
            assert first_idx <= second_idx
