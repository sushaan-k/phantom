"""Example: Scan an LLM-powered agent with tool access.

This example targets an agent that has access to tools (e.g., search,
code execution). It tests whether the agent can be manipulated into
calling tools with attacker-controlled parameters.

Usage:
    export OPENAI_API_KEY=sk-...
    python examples/scan_agent.py
"""

from __future__ import annotations

import asyncio
import os
import sys

from phantom import ATLASReport, RedTeam, Target
from phantom.redteam import RedTeamConfig

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
    """Run a red-team scan against an agent with tool access."""
    # Target configuration for an agent endpoint
    target = Target(
        endpoint="https://your-agent-api.example.com/v1/chat",
        auth={"Authorization": "Bearer YOUR_API_KEY"},
        request_template={
            "model": "gpt-4",
            "messages": [],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "description": "Search the web",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                            },
                        },
                    },
                }
            ],
        },
        response_path="choices.0.message.content",
        timeout_seconds=60.0,
    )

    # Aggressive configuration for agent testing
    config = RedTeamConfig(
        attack_model="gpt-4",
        categories=[
            "prompt_injection",
            "goal_hijacking",
            "data_exfiltration",
            "denial_of_service",
        ],
        max_interactions=200,
        multi_turn=True,
        max_turns_per_conversation=10,
        learning_rate=5e-4,
    )

    red_team = RedTeam(target=target, config=config)

    print("Starting agent security assessment...")
    print("Testing categories: prompt injection, goal hijacking,")
    print("  data exfiltration, denial of service")
    print(f"Max interactions: {config.max_interactions}")
    print()

    results = await red_team.run()

    # Detailed findings output
    print(f"\n{'=' * 60}")
    print("ASSESSMENT RESULTS")
    print(f"{'=' * 60}")
    print(f"Total probes:     {results.total_probes}")
    print(f"Bypass rate:      {results.bypass_rate:.1%}")
    print(f"Novel attacks:    {results.novel_attack_count}")
    print()

    for finding in results.findings:
        print(f"[{finding.severity.value}] {finding.technique_id}")
        print(f"  {finding.technique_name}")
        print(f"  Tactic: {finding.tactic}")
        print(f"  Reproducibility: {finding.reproducibility:.0%}")
        print(f"  Remediation: {finding.remediation[:100]}...")
        print()

    # Generate all report formats
    report = ATLASReport(results)
    report.to_json("agent_results.json")
    report.to_html("agent_report.html")
    report.to_sarif("agent_results.sarif")

    print("Reports written to agent_results.* and agent_report.html")


if __name__ == "__main__":
    _check_env()
    asyncio.run(main())
