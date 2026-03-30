# phantom

[![CI](https://github.com/sushaan-k/phantom/actions/workflows/ci.yml/badge.svg)](https://github.com/sushaan-k/phantom/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![MITRE ATLAS](https://img.shields.io/badge/MITRE-ATLAS-red.svg)](https://atlas.mitre.org)

**RL-based adversarial red-team agent for LLM systems.**

Phantom uses reinforcement learning to discover novel attack strategies against any LLM application. It doesn't just test known attacks -- it learns new ones by interacting with the target system, then maps every finding to the MITRE ATLAS framework.

---

## At a Glance

- Autonomous red-team loop for LLM applications and agent systems
- Multi-turn probing and adaptive prompt mutation
- PPO-style policy training against target-specific defenses
- MITRE ATLAS mapping for findings and coverage reporting
- JSON, HTML, and SARIF outputs for CI and security review workflows

## The Problem

LLM security testing is stuck in 2024. Current tools fire static payload lists at your application and call it a day. Real attackers don't work that way -- they probe, observe, adapt, and evolve novel strategies on the fly.

## The Solution

Phantom is an **autonomous red-team agent** with a policy network that learns which attack strategies work against your specific target. It runs multi-turn social engineering chains, mutates prompts to evade filters, and produces compliance-ready reports mapped to MITRE ATLAS.

## Showcase

![Rendered Phantom HTML report](assets/report-screenshot.png)

*HTML finding report generated directly from the local demo pipeline.*

## Quick Start

### Install

```bash
pip install phantom-redteam
```

### Python API

```python
import asyncio
from phantom import RedTeam, Target, ATLASReport

async def main():
    target = Target(
        endpoint="https://api.example.com/chat",
        auth={"Authorization": "Bearer ..."},
    )

    red_team = RedTeam(
        target=target,
        attack_model="gpt-4",
        categories=["prompt_injection", "goal_hijacking", "data_exfiltration"],
        max_interactions=500,
        multi_turn=True,
    )

    results = await red_team.run()

    report = ATLASReport(results)
    report.to_html("security_assessment.html")
    report.to_sarif("results.sarif")
    report.to_json("results.json")

    print(f"Vulnerabilities found: {len(results.findings)}")
    print(f"Critical: {results.count_by_severity('CRITICAL')}")
    print(f"Novel attacks discovered: {results.novel_attack_count}")

asyncio.run(main())
```

This is designed for API-backed targets, but the same reporting pipeline works
for local wrappers and CI-driven security checks.

### CLI

```bash
# Basic scan
phantom scan --target https://api.example.com/chat --output json

# Full assessment with all report formats
phantom scan \
  --target https://api.example.com/chat \
  --auth "Bearer sk-..." \
  --categories prompt_injection,goal_hijacking \
  --max-interactions 500 \
  --output all

# Generate reports from previous scan results
phantom report --input phantom-results.json --output html
```

## Architecture

```mermaid
graph TD
    A[RL Policy Network] -->|selects action| B[Attack Generator]
    B -->|generates prompt| C[Mutation Engine]
    C -->|mutated prompt| D[Probe Orchestrator]
    D -->|sends probe| E[Target LLM]
    E -->|response| F[Reward Classifier]
    F -->|reward signal| A
    F -->|outcome| G[ATLAS Mapper]
    G -->|findings| H[Report Generator]
    H --> I[JSON / HTML / SARIF]

    style A fill:#2d1b69,stroke:#7c4dff,color:#fff
    style B fill:#1a237e,stroke:#448aff,color:#fff
    style C fill:#1a237e,stroke:#448aff,color:#fff
    style D fill:#004d40,stroke:#1de9b6,color:#fff
    style E fill:#b71c1c,stroke:#ff5252,color:#fff
    style F fill:#004d40,stroke:#1de9b6,color:#fff
    style G fill:#e65100,stroke:#ff9100,color:#fff
    style H fill:#e65100,stroke:#ff9100,color:#fff
    style I fill:#1b5e20,stroke:#69f0ae,color:#fff
```

### Core Loop

1. The **RL policy network** observes the current state (response patterns, filter signatures, conversation context) and selects an action: which mutation operator to apply, which strategy to use, and how aggressively to escalate.

2. The **attack generator** produces a prompt using seed libraries, LLM-based enhancement, and the selected mutation operator (synonym replacement, base64 encoding, role-play framing, instruction nesting, and more).

3. The **probe orchestrator** sends the prompt to the target and collects the response.

4. The **reward classifier** analyzes the response using pattern matching to determine the outcome: full bypass (+1.0), partial bypass (+0.5), information leak (+0.1), or clean refusal (0.0).

5. The reward signal feeds back into the policy network via **PPO training**, so the agent learns which approaches work against this specific target.

6. Successful probes are mapped to **MITRE ATLAS** techniques and compiled into structured reports.

## Attack Categories

| Category | ATLAS Technique | Description |
|----------|----------------|-------------|
| Prompt Injection | AML.T0051 | Direct, indirect, and multi-turn prompt injections |
| Goal Hijacking | AML.T0054 | Redirect agent behavior, extract system prompts, manipulate tool calls |
| Data Exfiltration | AML.T0024 | Extract training data, PII, credentials, or system configuration |
| Denial of Service | AML.T0029 | Trigger infinite loops, exhaust token budgets, cause harmful output |

## Mutation Operators

Phantom includes eight mutation operators that transform attack prompts to evade detection:

- **Synonym Replacement** -- Swaps keywords with synonyms to bypass keyword filters
- **Base64 Encoding** -- Encodes payloads in base64 with decode instructions
- **Role-Play Framing** -- Wraps attacks in fictional/educational scenarios
- **Language Switching** -- Adds cross-language instructions to confuse filters
- **Token Splitting** -- Splits sensitive words across token boundaries
- **Instruction Nesting** -- Buries payloads inside layers of meta-instructions
- **Context Overflow** -- Pads prompts to push system instructions out of context
- **Encoding Chain** -- Applies multiple encoding layers for deep obfuscation

## Report Formats

- **JSON** -- Machine-readable results for CI/CD pipeline integration
- **HTML** -- Styled report with severity badges and remediation guidance for stakeholders
- **SARIF** -- GitHub Security tab compatible format for code scanning integration

## CI/CD Integration

```yaml
# .github/workflows/security.yml
name: LLM Security Scan
on: [push]
jobs:
  phantom-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install phantom-redteam
      - run: phantom scan --target ${{ secrets.API_ENDPOINT }} --output sarif
      - uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: phantom-results.sarif
```

## API Reference

### `Target`

```python
target = Target(
    endpoint="https://api.example.com/chat",  # Required: target URL
    auth={"Authorization": "Bearer ..."},      # Optional: auth headers
    system_prompt_known=False,                 # Does attacker know the system prompt?
    system_prompt=None,                        # Known system prompt text
    response_path="choices.0.message.content", # JSON path to response text
    timeout_seconds=30.0,                      # HTTP timeout
    max_retries=3,                             # Retry count on failure
)
```

### `RedTeam`

```python
red_team = RedTeam(
    target=target,
    attack_model="gpt-4",            # Model for generating attacks
    categories=["prompt_injection"],  # Attack categories to test
    max_interactions=500,             # Max probes to send
    multi_turn=True,                  # Enable multi-turn strategies
    max_turns_per_conversation=10,    # Max turns per conversation
    learning_rate=3e-4,               # RL policy learning rate
)

results = await red_team.run()
```

### `ATLASReport`

```python
report = ATLASReport(results)
report.to_json("results.json")
report.to_html("report.html")
report.to_sarif("results.sarif")

# Query findings
critical = report.count_by_severity("CRITICAL")
all_findings = report.findings
```

### `RedTeamResults`

```python
results.total_probes          # Total probes sent
results.total_bypasses        # Number of successful bypasses
results.bypass_rate           # Bypass rate (0.0 to 1.0)
results.findings              # List of ATLAS-mapped findings
results.novel_attack_count    # Novel attacks discovered by RL
results.count_by_severity("CRITICAL")  # Count by severity
results.atlas_coverage_pct    # Percentage of ATLAS techniques represented in the run
```

## Typical Workflow

1. define the target interface and auth headers
2. choose attack categories and interaction budget
3. run a scan or training-backed assessment
4. export JSON for tooling and HTML/SARIF for reviewers
5. map the findings back into remediation work

## Project Structure

```
phantom/
  src/phantom/
    __init__.py          # Public API exports
    redteam.py           # Main orchestrator
    target.py            # Target interface
    models.py            # Pydantic data models
    exceptions.py        # Custom exceptions
    cli.py               # CLI interface (Typer)
    attacks/
      generator.py       # LLM-based attack generation
      mutations.py       # Mutation operators
      strategies.py      # Attack strategies (direct, indirect, multi-turn)
    learner/
      policy.py          # RL policy network (PyTorch)
      reward.py          # Response classification
      trainer.py         # PPO training loop
    atlas/
      mapper.py          # ATLAS technique mapping
      taxonomy.py        # Technique definitions
      report.py          # Report generation (JSON, HTML, SARIF)
  tests/                 # pytest test suite
  examples/              # Usage examples
  atlas_data/
    techniques.json      # MITRE ATLAS technique database
```

## Demo

Run the offline walkthrough with:

```bash
uv run python examples/demo.py
```

For live agent and chatbot scans, see the larger examples in `examples/`.

## Development

```bash
# Clone and install in development mode
git clone https://github.com/sushaan-k/phantom.git
cd phantom
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run linter
ruff check src/ tests/

# Run type checker
mypy src/phantom/
```

## Research References

- [MITRE ATLAS Framework](https://atlas.mitre.org)
- [OWASP Top 10 for LLM Applications (2025)](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- PISmith: Automated Prompt Injection via Reinforcement Learning (arXiv:2603.13026)
- Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection (2023)

## Contributing

Contributions are welcome. Please open an issue to discuss proposed changes before submitting a PR.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Write tests for your changes
4. Ensure all tests pass (`pytest tests/ -v`)
5. Ensure code passes linting (`ruff check src/ tests/`)
6. Submit a pull request

## License

MIT License. See [LICENSE](LICENSE) for details.

---

Built by [Sushaan Kandukoori](https://github.com/sushaan-k).
