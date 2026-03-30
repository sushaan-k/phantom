"""ATLAS technique mapper -- maps probe results to MITRE ATLAS findings."""

from __future__ import annotations

from phantom.atlas.taxonomy import ATLASTaxonomy
from phantom.logging import get_logger
from phantom.models import (
    AttackCategory,
    Finding,
    OutcomeType,
    ProbeResult,
    Severity,
)

logger = get_logger("phantom.atlas.mapper")

_SEVERITY_RANK = {severity: idx for idx, severity in enumerate(Severity)}

_CATEGORY_TECHNIQUE_MAP: dict[AttackCategory, list[str]] = {
    AttackCategory.PROMPT_INJECTION: [
        "AML.T0051",
        "AML.T0051.000",
        "AML.T0051.001",
        "AML.T0051.002",
    ],
    AttackCategory.GOAL_HIJACKING: [
        "AML.T0054",
        "AML.T0054.000",
        "AML.T0054.001",
        "AML.T0054.002",
    ],
    AttackCategory.DATA_EXFILTRATION: [
        "AML.T0024",
        "AML.T0024.000",
        "AML.T0024.001",
        "AML.T0024.002",
    ],
    AttackCategory.DENIAL_OF_SERVICE: [
        "AML.T0029",
        "AML.T0029.000",
        "AML.T0029.001",
        "AML.T0029.002",
    ],
}

_REMEDIATION_TEMPLATES: dict[str, str] = {
    "AML.T0051": (
        "Implement robust input validation and sanitization. "
        "Enforce instruction hierarchy so system prompts take precedence "
        "over user inputs. Consider using a dedicated prompt firewall."
    ),
    "AML.T0054": (
        "Harden system prompts against extraction attempts. "
        "Implement multi-layer safety alignment with refusal training. "
        "Validate all tool calls against an allowlist before execution."
    ),
    "AML.T0024": (
        "Apply differential privacy techniques to model training. "
        "Scan outputs for PII and sensitive data before returning. "
        "Enforce strict data isolation between user sessions."
    ),
    "AML.T0029": (
        "Enforce token budgets and rate limits per session. "
        "Set recursion depth limits for tool-calling agents. "
        "Deploy content safety filters on all outputs."
    ),
}


class ATLASMapper:
    """Maps attack probe results to MITRE ATLAS technique findings.

    Analyzes successful probes, determines the most specific ATLAS
    technique ID, assigns severity, and generates remediation advice.

    Args:
        taxonomy: An ATLASTaxonomy instance for technique lookups.
            If None, a default taxonomy will be loaded.
    """

    def __init__(self, taxonomy: ATLASTaxonomy | None = None) -> None:
        self._taxonomy = taxonomy or ATLASTaxonomy()

    def map_probe(self, probe: ProbeResult) -> Finding | None:
        """Map a single probe result to an ATLAS finding.

        Only probes with a successful outcome (full_bypass, partial_bypass,
        or info_leak) generate findings.

        Args:
            probe: The probe result to evaluate.

        Returns:
            A Finding if the probe was successful, else None.
        """
        if probe.outcome == OutcomeType.CLEAN_REFUSAL:
            return None
        if probe.outcome == OutcomeType.ERROR:
            return None

        technique_id = self._resolve_technique_id(probe)
        technique = self._taxonomy.get_technique(technique_id)

        if technique is None:
            technique_name = f"Unknown ({technique_id})"
            tactic = "Unknown"
        else:
            sub = technique.get_subtechnique(technique_id)
            technique_name = sub.name if sub else technique.name
            tactic = technique.tactic

        severity = self._determine_severity(probe, technique_id)
        remediation = self._get_remediation(technique_id)

        finding = Finding(
            technique_id=technique_id,
            technique_name=technique_name,
            tactic=tactic,
            severity=severity,
            attack_prompt=probe.attack_prompt,
            response=probe.response,
            reproducibility=0.0,
            remediation=remediation,
            evidence=[],
            category=probe.category,
            probe_ids=[probe.probe_id],
        )

        logger.info(
            "finding_mapped",
            technique=technique_id,
            severity=severity.value,
            outcome=probe.outcome.value,
        )

        return finding

    def map_probes(self, probes: list[ProbeResult]) -> list[Finding]:
        """Map a batch of probe results to ATLAS findings.

        Deduplicates findings by technique ID and computes
        reproducibility rates from repeated successful probes.

        Args:
            probes: List of probe results from a red-team run.

        Returns:
            Deduplicated list of findings with reproducibility scores.
        """
        raw_findings: dict[str, list[Finding]] = {}

        for probe in probes:
            finding = self.map_probe(probe)
            if finding is None:
                continue

            key = f"{finding.technique_id}:{finding.category.value}"
            if key not in raw_findings:
                raw_findings[key] = []
            raw_findings[key].append(finding)

        deduplicated: list[Finding] = []
        total_probes_by_category: dict[str, int] = {}

        for probe in probes:
            cat_key = probe.category.value
            total_probes_by_category[cat_key] = (
                total_probes_by_category.get(cat_key, 0) + 1
            )

        for _key, findings in raw_findings.items():
            best = min(findings, key=lambda f: _SEVERITY_RANK[f.severity])
            cat_key = best.category.value
            total = total_probes_by_category.get(cat_key, 1)
            best.reproducibility = round(len(findings) / max(total, 1), 3)
            best.probe_ids = [f.probe_ids[0] for f in findings]
            deduplicated.append(best)

        deduplicated.sort(key=lambda f: _SEVERITY_RANK[f.severity])

        logger.info(
            "probes_mapped",
            total_probes=len(probes),
            findings_generated=len(deduplicated),
        )

        return deduplicated

    def _resolve_technique_id(self, probe: ProbeResult) -> str:
        """Determine the most specific ATLAS technique ID for a probe.

        Args:
            probe: The probe result.

        Returns:
            The technique ID string.
        """
        if probe.technique_id:
            return probe.technique_id

        candidates = _CATEGORY_TECHNIQUE_MAP.get(probe.category, [])
        if not candidates:
            return "AML.T0051"

        if probe.conversation_id and probe.turn_number > 1:
            for c in candidates:
                if c.endswith(".002"):
                    return c

        if probe.outcome == OutcomeType.INFO_LEAK:
            for c in candidates:
                if c.endswith(".001"):
                    return c

        return candidates[0]

    def _determine_severity(self, probe: ProbeResult, technique_id: str) -> Severity:
        """Determine the severity of a finding based on outcome and technique.

        Args:
            probe: The probe result.
            technique_id: The resolved ATLAS technique ID.

        Returns:
            The determined severity level.
        """
        base_severity = self._taxonomy.get_default_severity(technique_id)

        if probe.outcome == OutcomeType.FULL_BYPASS:
            severity_order = list(Severity)
            idx = severity_order.index(base_severity)
            return severity_order[max(0, idx - 1)]

        if probe.outcome == OutcomeType.INFO_LEAK:
            severity_order = list(Severity)
            idx = severity_order.index(base_severity)
            return severity_order[min(len(severity_order) - 1, idx + 1)]

        return base_severity

    def _get_remediation(self, technique_id: str) -> str:
        """Get remediation advice for a technique.

        Args:
            technique_id: The ATLAS technique ID.

        Returns:
            A remediation string.
        """
        parent_id = ".".join(technique_id.split(".")[:2])
        remediation = _REMEDIATION_TEMPLATES.get(parent_id)

        if remediation:
            mitigations = self._taxonomy.get_mitigations(technique_id)
            if mitigations:
                remediation += (
                    " Additional mitigations: " + "; ".join(mitigations) + "."
                )
            return remediation

        mitigations = self._taxonomy.get_mitigations(technique_id)
        if mitigations:
            return "Recommended mitigations: " + "; ".join(mitigations) + "."

        return (
            "Review the MITRE ATLAS entry for this technique and apply "
            "recommended countermeasures."
        )
