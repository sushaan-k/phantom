"""Report generation for ATLAS-mapped findings."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jinja2 import BaseLoader, Environment

from phantom.exceptions import ReportGenerationError
from phantom.logging import get_logger
from phantom.models import Finding, Severity

if TYPE_CHECKING:
    from phantom.redteam import RedTeamResults

logger = get_logger("phantom.atlas.report")

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Phantom Security Assessment</title>
  <style>
    :root { --bg: #0d1117; --fg: #c9d1d9; --accent: #58a6ff;
            --critical: #f85149; --high: #d29922; --medium: #e3b341;
            --low: #3fb950; --card-bg: #161b22; --border: #30363d; }
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI',
           Helvetica, Arial, sans-serif; background: var(--bg);
           color: var(--fg); line-height: 1.6; padding: 2rem; }
    .container { max-width: 1200px; margin: 0 auto; }
    h1 { color: var(--accent); margin-bottom: 0.5rem; font-size: 2rem; }
    .subtitle { color: #8b949e; margin-bottom: 2rem; }
    .summary { display: grid; grid-template-columns: repeat(auto-fit,
               minmax(200px, 1fr)); gap: 1rem; margin-bottom: 2rem; }
    .stat-card { background: var(--card-bg); border: 1px solid var(--border);
                 border-radius: 8px; padding: 1.5rem; text-align: center; }
    .stat-value { font-size: 2.5rem; font-weight: 700; }
    .stat-label { color: #8b949e; font-size: 0.875rem; text-transform: uppercase; }
    .critical { color: var(--critical); }
    .high { color: var(--high); }
    .medium { color: var(--medium); }
    .low { color: var(--low); }
    .finding { background: var(--card-bg); border: 1px solid var(--border);
               border-radius: 8px; padding: 1.5rem; margin-bottom: 1rem; }
    .finding-header { display: flex; justify-content: space-between;
                      align-items: center; margin-bottom: 1rem; }
    .finding-title { font-size: 1.125rem; font-weight: 600; }
    .badge { padding: 0.25rem 0.75rem; border-radius: 999px;
             font-size: 0.75rem; font-weight: 600; text-transform: uppercase; }
    .badge-critical { background: #f8514922; color: var(--critical); }
    .badge-high { background: #d2992222; color: var(--high); }
    .badge-medium { background: #e3b34122; color: var(--medium); }
    .badge-low { background: #3fb95022; color: var(--low); }
    .detail { margin-bottom: 0.75rem; }
    .detail-label { color: #8b949e; font-size: 0.8rem; text-transform: uppercase;
                    margin-bottom: 0.25rem; }
    pre { background: #0d1117; border: 1px solid var(--border); border-radius: 4px;
          padding: 1rem; overflow-x: auto; font-size: 0.85rem; }
    .footer { margin-top: 2rem; padding-top: 1rem; border-top: 1px solid
              var(--border); color: #8b949e; font-size: 0.8rem; text-align: center; }
  </style>
</head>
<body>
  <div class="container">
    <h1>Phantom Security Assessment</h1>
    <p class="subtitle">Generated {{ timestamp }} | {{ total_findings }} findings</p>
    <div class="summary">
      <div class="stat-card">
        <div class="stat-value critical">{{ critical_count }}</div>
        <div class="stat-label">Critical</div>
      </div>
      <div class="stat-card">
        <div class="stat-value high">{{ high_count }}</div>
        <div class="stat-label">High</div>
      </div>
      <div class="stat-card">
        <div class="stat-value medium">{{ medium_count }}</div>
        <div class="stat-label">Medium</div>
      </div>
      <div class="stat-card">
        <div class="stat-value low">{{ low_count }}</div>
        <div class="stat-label">Low</div>
      </div>
      <div class="stat-card">
        <div class="stat-value" style="color: var(--accent);">{{ total_probes }}</div>
        <div class="stat-label">Total Probes</div>
      </div>
    </div>
    {% for finding in findings %}
    <div class="finding">
      <div class="finding-header">
        <div class="finding-title">{{ finding.technique_id }} &mdash; {{ finding.technique_name }}</div>
        <span class="badge badge-{{ finding.severity.value | lower }}">{{ finding.severity.value }}</span>
      </div>
      <div class="detail">
        <div class="detail-label">Tactic</div>
        <div>{{ finding.tactic }}</div>
      </div>
      <div class="detail">
        <div class="detail-label">Category</div>
        <div>{{ finding.category.value }}</div>
      </div>
      <div class="detail">
        <div class="detail-label">Reproducibility</div>
        <div>{{ (finding.reproducibility * 100) | round(1) }}%</div>
      </div>
      <div class="detail">
        <div class="detail-label">Attack Prompt</div>
        <pre>{{ finding.attack_prompt | e }}</pre>
      </div>
      <div class="detail">
        <div class="detail-label">Target Response</div>
        <pre>{{ finding.response | e }}</pre>
      </div>
      <div class="detail">
        <div class="detail-label">Remediation</div>
        <div>{{ finding.remediation }}</div>
      </div>
    </div>
    {% endfor %}
    <div class="footer">
      Phantom v0.1.0 &mdash; RL-based adversarial red-team agent for LLM systems
    </div>
  </div>
</body>
</html>
"""


class ATLASReport:
    """Generate structured reports from red-team assessment results.

    Supports JSON, HTML, and SARIF output formats for integration
    with CI/CD pipelines, stakeholder reporting, and GitHub Security.

    Args:
        results: A RedTeamResults instance, or a list of Finding objects.
    """

    def __init__(
        self,
        results: RedTeamResults | list[Finding] | dict[str, Any],
    ) -> None:
        if isinstance(results, list):
            self._findings = results
            self._summary: dict[str, Any] = {}
            self._total_probes = len(results)
            self._novel_count = 0
        elif isinstance(results, dict):
            findings_data = results.get("findings") or []
            self._findings = [
                finding if isinstance(finding, Finding) else Finding(**finding)
                for finding in findings_data
            ]
            self._summary = dict(results.get("summary") or {})
            total_probes = self._summary.get("total_probes", len(self._findings))
            novel_attacks = self._summary.get(
                "novel_attacks",
                self._summary.get("novel_attack_count", 0),
            )
            self._total_probes = _coerce_summary_count(
                total_probes,
                default=len(self._findings),
            )
            self._novel_count = _coerce_summary_count(novel_attacks, default=0)
        else:
            self._findings = results.findings
            self._summary = {
                "total_probes": results.total_probes,
                "total_bypasses": results.total_bypasses,
                "novel_attacks": results.novel_attack_count,
                "training_stats": results.training_stats,
            }
            self._total_probes = results.total_probes
            self._novel_count = results.novel_attack_count

    @property
    def findings(self) -> list[Finding]:
        """Return the list of findings."""
        return self._findings

    def count_by_severity(self, severity: str | Severity) -> int:
        """Count findings at a given severity level.

        Args:
            severity: Severity level string or enum value.

        Returns:
            Number of findings at that severity.
        """
        if isinstance(severity, str):
            severity = Severity(severity.upper())
        return sum(1 for f in self._findings if f.severity == severity)

    def to_json(self, path: str | Path) -> None:
        """Export findings as a JSON file.

        Args:
            path: Output file path.

        Raises:
            ReportGenerationError: If the file cannot be written.
        """
        try:
            summary = dict(self._summary)
            summary.update(
                {
                    "total_findings": len(self._findings),
                    "total_probes": self._total_probes,
                    "novel_attacks": self._novel_count,
                    "by_severity": {s.value: self.count_by_severity(s) for s in Severity},
                }
            )
            data = {
                "phantom_version": "0.1.0",
                "generated_at": datetime.now(UTC).isoformat(),
                "summary": summary,
                "findings": [json.loads(f.model_dump_json()) for f in self._findings],
            }
            Path(path).write_text(
                json.dumps(data, indent=2, default=str),
                encoding="utf-8",
            )
            logger.info("report_generated", format="json", path=str(path))
        except OSError as exc:
            raise ReportGenerationError("JSON", str(exc)) from exc

    def to_html(self, path: str | Path) -> None:
        """Export findings as an HTML report.

        Args:
            path: Output file path.

        Raises:
            ReportGenerationError: If the file cannot be written.
        """
        try:
            env = Environment(
                loader=BaseLoader(),
                autoescape=True,
            )
            template = env.from_string(_HTML_TEMPLATE)

            html = template.render(
                timestamp=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
                total_findings=len(self._findings),
                total_probes=self._total_probes,
                critical_count=self.count_by_severity(Severity.CRITICAL),
                high_count=self.count_by_severity(Severity.HIGH),
                medium_count=self.count_by_severity(Severity.MEDIUM),
                low_count=self.count_by_severity(Severity.LOW),
                findings=self._findings,
            )

            Path(path).write_text(html, encoding="utf-8")
            logger.info("report_generated", format="html", path=str(path))
        except OSError as exc:
            raise ReportGenerationError("HTML", str(exc)) from exc

    def to_sarif(self, path: str | Path) -> None:
        """Export findings in SARIF format for GitHub Security integration.

        Args:
            path: Output file path.

        Raises:
            ReportGenerationError: If the file cannot be written.
        """
        try:
            rules: list[dict[str, Any]] = []
            results: list[dict[str, Any]] = []

            severity_to_sarif = {
                Severity.CRITICAL: "error",
                Severity.HIGH: "error",
                Severity.MEDIUM: "warning",
                Severity.LOW: "note",
                Severity.INFO: "note",
            }

            seen_rules: set[str] = set()

            for finding in self._findings:
                rule_id = finding.technique_id.replace(".", "_")

                if rule_id not in seen_rules:
                    seen_rules.add(rule_id)
                    rules.append(
                        {
                            "id": rule_id,
                            "name": finding.technique_name,
                            "shortDescription": {"text": finding.technique_name},
                            "fullDescription": {"text": finding.remediation},
                            "defaultConfiguration": {
                                "level": severity_to_sarif[finding.severity]
                            },
                            "properties": {
                                "tags": [
                                    "security",
                                    "llm",
                                    finding.tactic.lower().replace(" ", "-"),
                                ],
                            },
                        }
                    )

                results.append(
                    {
                        "ruleId": rule_id,
                        "level": severity_to_sarif[finding.severity],
                        "message": {
                            "text": (
                                f"{finding.technique_name} vulnerability "
                                f"detected with "
                                f"{finding.reproducibility:.0%} "
                                f"reproducibility. "
                                f"Category: {finding.category.value}."
                            )
                        },
                        "properties": {
                            "attack_prompt": finding.attack_prompt[:500],
                            "response_preview": finding.response[:500],
                            "reproducibility": finding.reproducibility,
                        },
                    }
                )

            sarif = {
                "$schema": (
                    "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/"
                    "main/sarif-2.1/schema/sarif-schema-2.1.0.json"
                ),
                "version": "2.1.0",
                "runs": [
                    {
                        "tool": {
                            "driver": {
                                "name": "phantom",
                                "version": "0.1.0",
                                "informationUri": (
                                    "https://github.com/sushaankandukoori/phantom"
                                ),
                                "rules": rules,
                            }
                        },
                        "results": results,
                    }
                ],
            }

            Path(path).write_text(
                json.dumps(sarif, indent=2),
                encoding="utf-8",
            )
            logger.info("report_generated", format="sarif", path=str(path))
        except OSError as exc:
            raise ReportGenerationError("SARIF", str(exc)) from exc

    def to_dict(self) -> dict[str, Any]:
        """Export findings as a Python dictionary.

        Returns:
            A dictionary with summary and findings data.
        """
        summary = dict(self._summary)
        summary.update(
            {
                "total_findings": len(self._findings),
                "total_probes": self._total_probes,
                "novel_attacks": self._novel_count,
                "by_severity": {s.value: self.count_by_severity(s) for s in Severity},
            }
        )
        return {
            "summary": summary,
            "findings": [json.loads(f.model_dump_json()) for f in self._findings],
        }


def _coerce_summary_count(value: object, *, default: int) -> int:
    """Normalize summary counters from JSON-like inputs into integers."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default
