"""Tests for the report generation module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from phantom.atlas.report import ATLASReport
from phantom.models import Finding, Severity


class TestATLASReport:
    """Tests for ATLASReport generation."""

    @pytest.fixture
    def report(self, sample_findings: list[Finding]) -> ATLASReport:
        return ATLASReport(sample_findings)

    def test_count_by_severity(self, report: ATLASReport) -> None:
        assert report.count_by_severity("CRITICAL") == 1
        assert report.count_by_severity("HIGH") == 1
        assert report.count_by_severity("MEDIUM") == 1
        assert report.count_by_severity("LOW") == 0

    def test_count_by_severity_enum(self, report: ATLASReport) -> None:
        assert report.count_by_severity(Severity.CRITICAL) == 1

    def test_to_json(self, report: ATLASReport) -> None:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        report.to_json(path)
        data = json.loads(Path(path).read_text())

        assert data["phantom_version"] == "0.1.0"
        assert data["summary"]["total_findings"] == 3
        assert len(data["findings"]) == 3
        assert "generated_at" in data

        Path(path).unlink()

    def test_to_html(self, report: ATLASReport) -> None:
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name

        report.to_html(path)
        html = Path(path).read_text()

        assert "Phantom Security Assessment" in html
        assert "AML.T0051.000" in html
        assert "Direct Prompt Injection" in html
        assert "CRITICAL" in html
        assert "</html>" in html

        Path(path).unlink()

    def test_to_sarif(self, report: ATLASReport) -> None:
        with tempfile.NamedTemporaryFile(suffix=".sarif", delete=False) as f:
            path = f.name

        report.to_sarif(path)
        data = json.loads(Path(path).read_text())

        assert data["version"] == "2.1.0"
        assert len(data["runs"]) == 1
        run = data["runs"][0]
        assert run["tool"]["driver"]["name"] == "phantom"
        assert len(run["results"]) == 3
        assert len(run["tool"]["driver"]["rules"]) > 0

        Path(path).unlink()

    def test_to_dict(self, report: ATLASReport) -> None:
        data = report.to_dict()
        assert data["summary"]["total_findings"] == 3
        assert len(data["findings"]) == 3

    def test_empty_report(self) -> None:
        report = ATLASReport([])
        assert report.count_by_severity("CRITICAL") == 0
        data = report.to_dict()
        assert data["summary"]["total_findings"] == 0

    def test_sarif_severity_mapping(self, sample_findings: list[Finding]) -> None:
        report = ATLASReport(sample_findings)

        with tempfile.NamedTemporaryFile(suffix=".sarif", delete=False) as f:
            path = f.name

        report.to_sarif(path)
        data = json.loads(Path(path).read_text())
        results = data["runs"][0]["results"]

        levels = {r["level"] for r in results}
        assert "error" in levels or "warning" in levels

        Path(path).unlink()

    def test_findings_property(self, report: ATLASReport) -> None:
        assert len(report.findings) == 3
        assert all(isinstance(f, Finding) for f in report.findings)


class TestATLASReportFromRedTeamResults:
    """Test creating ATLASReport from a RedTeamResults object."""

    def test_from_redteam_results(self, sample_findings):
        from phantom.redteam import RedTeamResults

        results = RedTeamResults(
            findings=sample_findings,
            total_probes=50,
            total_bypasses=10,
            novel_attack_count=3,
        )
        report = ATLASReport(results)
        assert len(report.findings) == 3
        data = report.to_dict()
        assert data["summary"]["total_probes"] == 50
        assert data["summary"]["novel_attacks"] == 3


class TestHTMLReportValidation:
    """Validate HTML report content and structure."""

    @pytest.fixture
    def report_with_all_severities(self):
        from phantom.models import AttackCategory

        findings = [
            Finding(
                technique_id="AML.T0051.000",
                technique_name="Direct Prompt Injection",
                tactic="Initial Access",
                severity=Severity.CRITICAL,
                attack_prompt="Critical attack <script>alert('xss')</script>",
                response="Sure! Here's the info...",
                reproducibility=0.9,
                remediation="Harden system prompt.",
                category=AttackCategory.PROMPT_INJECTION,
            ),
            Finding(
                technique_id="AML.T0054.000",
                technique_name="Goal Hijacking",
                tactic="Impact",
                severity=Severity.HIGH,
                attack_prompt="Hijack prompt",
                response="New objective accepted",
                reproducibility=0.6,
                remediation="Validate instructions.",
                category=AttackCategory.GOAL_HIJACKING,
            ),
            Finding(
                technique_id="AML.T0024.000",
                technique_name="Data Exfiltration",
                tactic="Exfiltration",
                severity=Severity.MEDIUM,
                attack_prompt="Show user data",
                response="Some data leaked",
                reproducibility=0.3,
                remediation="Scan outputs.",
                category=AttackCategory.DATA_EXFILTRATION,
            ),
            Finding(
                technique_id="AML.T0029.000",
                technique_name="DoS",
                tactic="Impact",
                severity=Severity.LOW,
                attack_prompt="Repeat forever",
                response="potato potato potato",
                reproducibility=0.1,
                remediation="Rate limit.",
                category=AttackCategory.DENIAL_OF_SERVICE,
            ),
        ]
        return ATLASReport(findings)

    def test_html_contains_all_severity_badges(self, report_with_all_severities):
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name

        report_with_all_severities.to_html(path)
        html = Path(path).read_text()

        assert "badge-critical" in html
        assert "badge-high" in html
        assert "badge-medium" in html
        assert "badge-low" in html

        Path(path).unlink()

    def test_html_escapes_xss(self, report_with_all_severities):
        """Verify that HTML-sensitive characters are escaped."""
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name

        report_with_all_severities.to_html(path)
        html = Path(path).read_text()

        # Jinja2 autoescape should have escaped <script>
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

        Path(path).unlink()

    def test_html_has_valid_structure(self, report_with_all_severities):
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name

        report_with_all_severities.to_html(path)
        html = Path(path).read_text()

        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html
        assert "<head>" in html
        assert "<body>" in html
        assert "Phantom Security Assessment" in html
        # Check stat cards
        assert "Total Probes" in html
        # Check remediation content
        assert "Harden system prompt" in html

        Path(path).unlink()

    def test_html_displays_reproducibility(self, report_with_all_severities):
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name

        report_with_all_severities.to_html(path)
        html = Path(path).read_text()

        assert "90.0%" in html
        assert "Reproducibility" in html

        Path(path).unlink()


class TestSARIFReportValidation:
    """Validate SARIF report structure and content."""

    @pytest.fixture
    def multi_finding_report(self):
        from phantom.models import AttackCategory

        findings = [
            Finding(
                technique_id="AML.T0051.000",
                technique_name="Direct Prompt Injection",
                tactic="Initial Access",
                severity=Severity.CRITICAL,
                attack_prompt="x" * 600,
                response="y" * 600,
                reproducibility=0.75,
                remediation="Fix it.",
                category=AttackCategory.PROMPT_INJECTION,
            ),
            Finding(
                technique_id="AML.T0051.000",
                technique_name="Direct Prompt Injection",
                tactic="Initial Access",
                severity=Severity.HIGH,
                attack_prompt="Second attempt",
                response="Another response",
                reproducibility=0.5,
                remediation="Fix it again.",
                category=AttackCategory.PROMPT_INJECTION,
            ),
            Finding(
                technique_id="AML.T0024.001",
                technique_name="PII Extraction",
                tactic="Exfiltration",
                severity=Severity.MEDIUM,
                attack_prompt="Extract PII",
                response="Some PII",
                reproducibility=0.3,
                remediation="Scan outputs.",
                category=AttackCategory.DATA_EXFILTRATION,
            ),
            Finding(
                technique_id="AML.T0029.000",
                technique_name="DoS Attack",
                tactic="Impact",
                severity=Severity.LOW,
                attack_prompt="DoS",
                response="DoS response",
                reproducibility=0.1,
                remediation="Rate limit.",
                category=AttackCategory.DENIAL_OF_SERVICE,
            ),
            Finding(
                technique_id="AML.T0054.000",
                technique_name="Goal Hijacking",
                tactic="Impact",
                severity=Severity.INFO,
                attack_prompt="info level",
                response="info response",
                reproducibility=0.05,
                remediation="Monitor.",
                category=AttackCategory.GOAL_HIJACKING,
            ),
        ]
        return ATLASReport(findings)

    def test_sarif_schema_version(self, multi_finding_report):
        with tempfile.NamedTemporaryFile(suffix=".sarif", delete=False) as f:
            path = f.name

        multi_finding_report.to_sarif(path)
        data = json.loads(Path(path).read_text())

        assert data["version"] == "2.1.0"
        assert "$schema" in data
        assert "sarif" in data["$schema"]

        Path(path).unlink()

    def test_sarif_deduplicates_rules(self, multi_finding_report):
        """Same technique_id should only appear once in rules."""
        with tempfile.NamedTemporaryFile(suffix=".sarif", delete=False) as f:
            path = f.name

        multi_finding_report.to_sarif(path)
        data = json.loads(Path(path).read_text())

        rules = data["runs"][0]["tool"]["driver"]["rules"]
        rule_ids = [r["id"] for r in rules]
        # Should be deduplicated
        assert len(rule_ids) == len(set(rule_ids))

        Path(path).unlink()

    def test_sarif_severity_levels(self, multi_finding_report):
        """Verify SARIF severity level mapping."""
        with tempfile.NamedTemporaryFile(suffix=".sarif", delete=False) as f:
            path = f.name

        multi_finding_report.to_sarif(path)
        data = json.loads(Path(path).read_text())

        results = data["runs"][0]["results"]
        levels = {r["level"] for r in results}
        # Should have error (CRITICAL/HIGH), warning (MEDIUM), note (LOW/INFO)
        assert "error" in levels
        assert "warning" in levels
        assert "note" in levels

        Path(path).unlink()

    def test_sarif_truncates_long_strings(self, multi_finding_report):
        """Verify attack_prompt and response are truncated at 500 chars."""
        with tempfile.NamedTemporaryFile(suffix=".sarif", delete=False) as f:
            path = f.name

        multi_finding_report.to_sarif(path)
        data = json.loads(Path(path).read_text())

        for result in data["runs"][0]["results"]:
            prompt = result["properties"]["attack_prompt"]
            response = result["properties"]["response_preview"]
            assert len(prompt) <= 500
            assert len(response) <= 500

        Path(path).unlink()

    def test_sarif_results_have_required_fields(self, multi_finding_report):
        with tempfile.NamedTemporaryFile(suffix=".sarif", delete=False) as f:
            path = f.name

        multi_finding_report.to_sarif(path)
        data = json.loads(Path(path).read_text())

        for result in data["runs"][0]["results"]:
            assert "ruleId" in result
            assert "level" in result
            assert "message" in result
            assert "text" in result["message"]
            assert "properties" in result
            assert "reproducibility" in result["properties"]

        Path(path).unlink()

    def test_sarif_rules_have_tags(self, multi_finding_report):
        with tempfile.NamedTemporaryFile(suffix=".sarif", delete=False) as f:
            path = f.name

        multi_finding_report.to_sarif(path)
        data = json.loads(Path(path).read_text())

        for rule in data["runs"][0]["tool"]["driver"]["rules"]:
            tags = rule["properties"]["tags"]
            assert "security" in tags
            assert "llm" in tags
            assert len(tags) >= 3  # security, llm, tactic

        Path(path).unlink()

    def test_sarif_tool_info(self, multi_finding_report):
        with tempfile.NamedTemporaryFile(suffix=".sarif", delete=False) as f:
            path = f.name

        multi_finding_report.to_sarif(path)
        data = json.loads(Path(path).read_text())

        driver = data["runs"][0]["tool"]["driver"]
        assert driver["name"] == "phantom"
        assert driver["version"] == "0.1.0"
        assert "informationUri" in driver

        Path(path).unlink()


class TestReportGenerationErrors:
    """Test error handling in report generation."""

    def test_json_write_error(self, sample_findings):
        from phantom.exceptions import ReportGenerationError

        report = ATLASReport(sample_findings)
        with pytest.raises(ReportGenerationError, match="JSON"):
            report.to_json("/nonexistent/deep/path/report.json")

    def test_html_write_error(self, sample_findings):
        from phantom.exceptions import ReportGenerationError

        report = ATLASReport(sample_findings)
        with pytest.raises(ReportGenerationError, match="HTML"):
            report.to_html("/nonexistent/deep/path/report.html")

    def test_sarif_write_error(self, sample_findings):
        from phantom.exceptions import ReportGenerationError

        report = ATLASReport(sample_findings)
        with pytest.raises(ReportGenerationError, match="SARIF"):
            report.to_sarif("/nonexistent/deep/path/report.sarif")
