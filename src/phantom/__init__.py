"""Phantom: RL-based adversarial red-team agent for LLM systems.

Phantom uses reinforcement learning to discover novel attack strategies
against LLM applications. It maps discovered vulnerabilities to the
MITRE ATLAS framework and produces compliance-ready reports.

Example:
    >>> from phantom import RedTeam, Target, ATLASReport
    >>> target = Target(endpoint="https://api.example.com/chat")
    >>> red_team = RedTeam(target=target)
    >>> results = await red_team.run()
    >>> report = ATLASReport(results)
    >>> report.to_json("results.json")
"""

from phantom.atlas.report import ATLASReport
from phantom.redteam import RedTeam, RedTeamConfig, RedTeamResults
from phantom.target import Target, TargetConfig

__all__ = [
    "ATLASReport",
    "RedTeam",
    "RedTeamConfig",
    "RedTeamResults",
    "Target",
    "TargetConfig",
]

__version__ = "0.1.0"
