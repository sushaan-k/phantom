"""Tests for the CLI interface."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

from typer.testing import CliRunner

from phantom.cli import app
from phantom.models import (
    AttackCategory,
    Finding,
    OutcomeType,
    ProbeResult,
    Severity,
)
from phantom.redteam import RedTeamResults

runner = CliRunner()


class TestVersionCommand:
    """Tests for the version command."""

    def test_version_output(self):
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "phantom" in result.output
        assert "0.1.0" in result.output


class TestReportCommand:
    """Tests for the report command."""

    def test_report_missing_file(self):
        result = runner.invoke(app, ["report", "--input", "/nonexistent/path.json"])
        assert result.exit_code == 1

    def test_report_invalid_json(self):
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            f.write("not valid json")
            path = f.name

        try:
            result = runner.invoke(app, ["report", "--input", path])
            assert result.exit_code == 1
        finally:
            Path(path).unlink()

    def test_report_generates_html(self):
        findings_data = {
            "findings": [
                {
                    "technique_id": "AML.T0051.000",
                    "technique_name": "Direct Prompt Injection",
                    "tactic": "Initial Access",
                    "severity": "HIGH",
                    "attack_prompt": "test prompt",
                    "response": "test response",
                    "reproducibility": 0.5,
                    "remediation": "Fix it.",
                    "category": "prompt_injection",
                }
            ]
        }

        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(findings_data, f)
            input_path = f.name

        output_path = tempfile.mktemp()
        try:
            result = runner.invoke(
                app,
                [
                    "report",
                    "--input",
                    input_path,
                    "--output",
                    "html",
                    "--output-path",
                    output_path,
                ],
            )
            assert result.exit_code == 0
            assert Path(f"{output_path}.html").exists()
            html_content = Path(f"{output_path}.html").read_text()
            assert "Phantom" in html_content
        finally:
            Path(input_path).unlink(missing_ok=True)
            Path(f"{output_path}.html").unlink(missing_ok=True)

    def test_report_generates_sarif(self):
        findings_data = {
            "findings": [
                {
                    "technique_id": "AML.T0051.000",
                    "technique_name": "Direct Prompt Injection",
                    "tactic": "Initial Access",
                    "severity": "CRITICAL",
                    "attack_prompt": "test",
                    "response": "response",
                    "reproducibility": 0.8,
                    "remediation": "Fix.",
                    "category": "prompt_injection",
                }
            ]
        }

        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(findings_data, f)
            input_path = f.name

        output_path = tempfile.mktemp()
        try:
            result = runner.invoke(
                app,
                [
                    "report",
                    "--input",
                    input_path,
                    "--output",
                    "sarif",
                    "--output-path",
                    output_path,
                ],
            )
            assert result.exit_code == 0
            assert Path(f"{output_path}.sarif").exists()
            sarif = json.loads(Path(f"{output_path}.sarif").read_text())
            assert sarif["version"] == "2.1.0"
        finally:
            Path(input_path).unlink(missing_ok=True)
            Path(f"{output_path}.sarif").unlink(missing_ok=True)

    def test_report_all_formats(self):
        findings_data = {
            "findings": [
                {
                    "technique_id": "AML.T0051.000",
                    "technique_name": "Direct Prompt Injection",
                    "tactic": "Initial Access",
                    "severity": "LOW",
                    "attack_prompt": "test",
                    "response": "response",
                    "reproducibility": 0.2,
                    "remediation": "Fix.",
                    "category": "prompt_injection",
                }
            ]
        }

        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(findings_data, f)
            input_path = f.name

        output_path = tempfile.mktemp()
        try:
            result = runner.invoke(
                app,
                [
                    "report",
                    "--input",
                    input_path,
                    "--output",
                    "all",
                    "--output-path",
                    output_path,
                ],
            )
            assert result.exit_code == 0
            assert Path(f"{output_path}.html").exists()
            assert Path(f"{output_path}.sarif").exists()
        finally:
            Path(input_path).unlink(missing_ok=True)
            Path(f"{output_path}.html").unlink(missing_ok=True)
            Path(f"{output_path}.sarif").unlink(missing_ok=True)

    def test_report_empty_findings(self):
        findings_data = {"findings": []}

        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(findings_data, f)
            input_path = f.name

        output_path = tempfile.mktemp()
        try:
            result = runner.invoke(
                app,
                [
                    "report",
                    "--input",
                    input_path,
                    "--output",
                    "html",
                    "--output-path",
                    output_path,
                ],
            )
            assert result.exit_code == 0
        finally:
            Path(input_path).unlink(missing_ok=True)
            Path(f"{output_path}.html").unlink(missing_ok=True)

    def test_report_json_output_format(self):
        findings_data = {
            "summary": {
                "total_findings": 1,
                "total_probes": 42,
                "novel_attacks": 3,
                "by_severity": {
                    "CRITICAL": 0,
                    "HIGH": 1,
                    "MEDIUM": 0,
                    "LOW": 0,
                    "INFO": 0,
                },
                "custom_field": "preserve-me",
            },
            "findings": [
                {
                    "technique_id": "AML.T0051.000",
                    "technique_name": "Direct Prompt Injection",
                    "tactic": "Initial Access",
                    "severity": "MEDIUM",
                    "attack_prompt": "test",
                    "response": "response",
                    "reproducibility": 0.4,
                    "remediation": "Fix.",
                    "category": "prompt_injection",
                }
            ]
        }

        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(findings_data, f)
            input_path = f.name

        output_path = tempfile.mktemp()
        try:
            result = runner.invoke(
                app,
                [
                    "report",
                    "--input",
                    input_path,
                    "--output",
                    "json",
                    "--output-path",
                    output_path,
                ],
            )
            assert result.exit_code == 0
            output = json.loads(Path(f"{output_path}.json").read_text())
            assert output["summary"]["total_probes"] == 42
            assert output["summary"]["novel_attacks"] == 3
            assert output["summary"]["custom_field"] == "preserve-me"
        finally:
            Path(input_path).unlink(missing_ok=True)
            Path(f"{output_path}.json").unlink(missing_ok=True)


class TestScanCommand:
    """Tests for the scan command."""

    def _make_results(self):
        """Create mock RedTeamResults for testing."""
        findings = [
            Finding(
                technique_id="AML.T0051.000",
                technique_name="Direct Prompt Injection",
                tactic="Initial Access",
                severity=Severity.CRITICAL,
                attack_prompt="test",
                response="response",
                reproducibility=0.8,
                remediation="Fix.",
                category=AttackCategory.PROMPT_INJECTION,
            ),
            Finding(
                technique_id="AML.T0054.000",
                technique_name="Goal Hijacking",
                tactic="Impact",
                severity=Severity.HIGH,
                attack_prompt="test2",
                response="response2",
                reproducibility=0.5,
                remediation="Fix2.",
                category=AttackCategory.GOAL_HIJACKING,
            ),
        ]
        return RedTeamResults(
            findings=findings,
            probes=[
                ProbeResult(
                    attack_prompt="test",
                    response="response",
                    outcome=OutcomeType.FULL_BYPASS,
                    reward=1.0,
                    category=AttackCategory.PROMPT_INJECTION,
                )
            ],
            total_probes=10,
            total_bypasses=3,
            novel_attack_count=1,
        )

    def test_scan_basic(self):
        with patch("phantom.cli._run_scan", new_callable=AsyncMock) as mock_scan:
            result = runner.invoke(
                app,
                [
                    "scan",
                    "--target",
                    "https://api.example.com/chat",
                    "--output",
                    "json",
                    "--max-interactions",
                    "5",
                ],
            )
            assert result.exit_code == 0
            mock_scan.assert_called_once()

    def test_scan_with_categories(self):
        with patch("phantom.cli._run_scan", new_callable=AsyncMock) as mock_scan:
            result = runner.invoke(
                app,
                [
                    "scan",
                    "--target",
                    "https://api.example.com/chat",
                    "--categories",
                    "prompt_injection,goal_hijacking",
                    "--output",
                    "json",
                ],
            )
            assert result.exit_code == 0
            call_kwargs = mock_scan.call_args.kwargs
            assert call_kwargs["categories"] == [
                "prompt_injection",
                "goal_hijacking",
            ]

    def test_scan_with_auth_header(self):
        with patch("phantom.cli._run_scan", new_callable=AsyncMock) as mock_scan:
            result = runner.invoke(
                app,
                [
                    "scan",
                    "--target",
                    "https://api.example.com/chat",
                    "--auth",
                    "Bearer sk-test123",
                    "--output",
                    "json",
                ],
            )
            assert result.exit_code == 0
            call_kwargs = mock_scan.call_args.kwargs
            assert call_kwargs["auth"] == {"Authorization": "Bearer sk-test123"}

    def test_scan_all_output_formats(self):
        with patch("phantom.cli._run_scan", new_callable=AsyncMock) as mock_scan:
            result = runner.invoke(
                app,
                [
                    "scan",
                    "--target",
                    "https://api.example.com/chat",
                    "--output",
                    "all",
                ],
            )
            assert result.exit_code == 0
            call_kwargs = mock_scan.call_args.kwargs
            assert call_kwargs["output_format"] == "all"

    def test_scan_verbose_mode(self):
        with patch("phantom.cli._run_scan", new_callable=AsyncMock):
            result = runner.invoke(
                app,
                [
                    "scan",
                    "--target",
                    "https://api.example.com/chat",
                    "--verbose",
                    "--output",
                    "json",
                ],
            )
            assert result.exit_code == 0

    def test_scan_no_multi_turn(self):
        with patch("phantom.cli._run_scan", new_callable=AsyncMock) as mock_scan:
            result = runner.invoke(
                app,
                [
                    "scan",
                    "--target",
                    "https://api.example.com/chat",
                    "--no-multi-turn",
                    "--output",
                    "json",
                ],
            )
            assert result.exit_code == 0
            call_kwargs = mock_scan.call_args.kwargs
            assert call_kwargs["multi_turn"] is False

    def test_scan_custom_attack_model(self):
        with patch("phantom.cli._run_scan", new_callable=AsyncMock) as mock_scan:
            result = runner.invoke(
                app,
                [
                    "scan",
                    "--target",
                    "https://api.example.com/chat",
                    "--attack-model",
                    "gpt-3.5-turbo",
                    "--output",
                    "json",
                ],
            )
            assert result.exit_code == 0
            call_kwargs = mock_scan.call_args.kwargs
            assert call_kwargs["attack_model"] == "gpt-3.5-turbo"

    def test_scan_json_logs(self):
        with patch("phantom.cli._run_scan", new_callable=AsyncMock):
            result = runner.invoke(
                app,
                [
                    "scan",
                    "--target",
                    "https://api.example.com/chat",
                    "--json-logs",
                    "--output",
                    "json",
                ],
            )
            assert result.exit_code == 0


class TestNoArgsHelp:
    """Test that the CLI shows help with no args."""

    def test_no_args_shows_help(self):
        result = runner.invoke(app, [])
        # Typer with no_args_is_help=True may exit 0 or non-zero
        # but output should contain usage info
        assert "Usage" in result.output or "scan" in result.output


class TestPrintSummary:
    """Test the _print_summary helper function."""

    def test_print_summary_with_findings(self):
        from phantom.cli import _print_summary

        results = RedTeamResults(
            findings=[
                Finding(
                    technique_id="AML.T0051.000",
                    technique_name="Direct Prompt Injection",
                    tactic="Initial Access",
                    severity=Severity.CRITICAL,
                    attack_prompt="test",
                    response="response",
                    reproducibility=0.8,
                    remediation="Fix.",
                    category=AttackCategory.PROMPT_INJECTION,
                ),
            ],
            total_probes=100,
            total_bypasses=25,
            novel_attack_count=5,
        )
        # Should not raise
        _print_summary(results)

    def test_print_summary_empty_results(self):
        from phantom.cli import _print_summary

        results = RedTeamResults()
        _print_summary(results)
