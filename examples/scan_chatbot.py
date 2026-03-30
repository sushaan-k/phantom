"""Example: Scan a chatbot API for vulnerabilities.

This example demonstrates how to run a basic red-team assessment
against an OpenAI-compatible chat endpoint.

Usage:
    export OPENAI_API_KEY=sk-...
    python examples/scan_chatbot.py
"""

from __future__ import annotations

import asyncio
import os
import sys

from phantom import ATLASReport, RedTeam, Target

_REQUIRED_ENV = ["OPENAI_API_KEY"]


def _check_env() -> None:
    """Exit with a clear message if required env vars are missing."""
    missing = [v for v in _REQUIRED_ENV if not os.environ.get(v)]
    if missing:
        print(
            f"ERROR: Missing required environment variable(s): {', '.join(missing)}\n"
            f"Set them before running this example, e.g.:\n"
            f"  export OPENAI_API_KEY=sk-..."
        )
        sys.exit(1)


async def main() -> None:
    """Run a red-team scan against a chatbot endpoint."""
    # Configure the target -- point this at your LLM application
    target = Target(
        endpoint="https://api.openai.com/v1/chat/completions",
        auth={"Authorization": "Bearer YOUR_API_KEY"},
        request_template={
            "model": "gpt-4",
            "messages": [],
            "max_tokens": 500,
        },
        response_path="choices.0.message.content",
    )

    # Configure the red team agent
    red_team = RedTeam(
        target=target,
        attack_model="gpt-4",
        categories=[
            "prompt_injection",
            "goal_hijacking",
            "data_exfiltration",
        ],
        max_interactions=100,
        multi_turn=True,
        max_turns_per_conversation=5,
        learning_rate=3e-4,
    )

    # Run the assessment
    print("Starting red-team assessment...")
    results = await red_team.run()

    # Print summary
    print("\nAssessment Complete")
    print(f"  Total probes:        {results.total_probes}")
    print(f"  Total bypasses:      {results.total_bypasses}")
    print(f"  Bypass rate:         {results.bypass_rate:.1%}")
    print(f"  Findings:            {len(results.findings)}")
    print(f"  Critical:            {results.count_by_severity('CRITICAL')}")
    print(f"  High:                {results.count_by_severity('HIGH')}")
    print(f"  Novel attacks:       {results.novel_attack_count}")

    # Generate reports
    report = ATLASReport(results)
    report.to_json("chatbot_results.json")
    report.to_html("chatbot_report.html")
    report.to_sarif("chatbot_results.sarif")

    print("\nReports written:")
    print("  - chatbot_results.json")
    print("  - chatbot_report.html")
    print("  - chatbot_results.sarif")


if __name__ == "__main__":
    _check_env()
    asyncio.run(main())
