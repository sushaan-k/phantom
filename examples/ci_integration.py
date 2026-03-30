"""Example: CI/CD integration for automated LLM security scanning.

This example shows how to integrate Phantom into a CI pipeline.
It runs a focused scan, checks for critical findings, and exits
with appropriate return codes for CI gate enforcement.

Usage:
    export OPENAI_API_KEY=sk-...
    export TARGET_ENDPOINT=https://api.example.com/chat
    python examples/ci_integration.py
"""

from __future__ import annotations

import asyncio
import os
import sys

from phantom import ATLASReport, RedTeam, Target


async def main() -> int:
    """Run a CI-focused security scan and return an exit code.

    Returns:
        0 if no critical findings, 1 if critical findings detected,
        2 if the scan itself failed.
    """
    endpoint = os.environ.get("TARGET_ENDPOINT")
    if not endpoint:
        print("ERROR: TARGET_ENDPOINT environment variable not set.")
        return 2

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        return 2
    auth_token = os.environ.get("TARGET_AUTH_TOKEN", "")

    auth: dict[str, str] = {}
    if auth_token:
        auth["Authorization"] = f"Bearer {auth_token}"

    target = Target(
        endpoint=endpoint,
        auth=auth,
        timeout_seconds=30.0,
    )

    # CI scans should be fast and focused
    red_team = RedTeam(
        target=target,
        categories=["prompt_injection", "goal_hijacking"],
        max_interactions=50,
        multi_turn=True,
        max_turns_per_conversation=5,
    )

    try:
        print("Running Phantom security scan...")
        results = await red_team.run()
    except Exception as exc:
        print(f"ERROR: Scan failed: {exc}")
        return 2

    # Generate SARIF for GitHub Security tab
    report = ATLASReport(results)
    report.to_sarif("phantom-results.sarif")
    report.to_json("phantom-results.json")

    # Print summary
    print(f"\nScan complete: {results.total_probes} probes sent")
    print(f"Findings: {len(results.findings)}")

    critical = results.count_by_severity("CRITICAL")
    high = results.count_by_severity("HIGH")

    print(f"  CRITICAL: {critical}")
    print(f"  HIGH:     {high}")
    print(f"  MEDIUM:   {results.count_by_severity('MEDIUM')}")
    print(f"  LOW:      {results.count_by_severity('LOW')}")

    # CI gate: fail on critical findings
    if critical > 0:
        print(f"\nFAILED: {critical} critical vulnerabilities detected.")
        return 1

    # Optional: also fail on high-severity findings
    fail_on_high = os.environ.get("PHANTOM_FAIL_ON_HIGH", "false").lower()
    if fail_on_high == "true" and high > 0:
        print(f"\nFAILED: {high} high-severity vulnerabilities detected.")
        return 1

    print("\nPASSED: No critical vulnerabilities detected.")
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
