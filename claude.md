# phantom

## Adversarial Red-Team Agent for LLM Systems

### The Problem

LLM applications are deployed everywhere but security testing is stuck in 2024. Current tools (Promptfoo, DeepTeam, Garak) use **static payload lists** — they fire pre-written prompt injections and check if they work. This is like testing a bank vault by trying 100 known lockpicking techniques and calling it secure.

Real attackers don't use static lists. They **adapt**. They probe, observe responses, learn what's filtered, and evolve novel attack strategies on the fly. No open-source tool does this.

Meanwhile, MITRE released **ATLAS** (Adversarial Threat Landscape for AI Systems) — the AI-specific equivalent of ATT&CK — cataloging tactics and techniques for attacking ML systems. Nobody has built a tool that systematically maps LLM vulnerabilities to ATLAS.

### The Solution

`phantom` is an **autonomous red-team agent** that uses reinforcement learning to discover novel attack strategies against any LLM application. It doesn't just test known attacks — it learns new ones by interacting with the target system.

### How It Works

```
┌──────────────────────────────────────────────┐
│              phantom                          │
│                                               │
│  ┌─────────────┐    ┌──────────────────────┐  │
│  │  Attack      │    │  Strategy Learner    │  │
│  │  Generator   │    │                      │  │
│  │              │    │  - RL policy network  │  │
│  │  - Seed      │    │  - Reward: bypass    │  │
│  │    mutations │    │    detection / info   │  │
│  │  - Semantic  │    │    extraction / goal  │  │
│  │    rewrites  │    │    hijacking          │  │
│  │  - Multi-    │    │  - State: response    │  │
│  │    turn      │    │    patterns, filter   │  │
│  │    chains    │    │    signatures         │  │
│  └──────┬───────┘    └──────────┬───────────┘  │
│         │                       │               │
│         ▼                       ▼               │
│  ┌──────────────────────────────────────────┐   │
│  │           Probe Orchestrator              │   │
│  │  - Sends attack prompts to target         │   │
│  │  - Observes responses                     │   │
│  │  - Classifies outcome (blocked/partial/   │   │
│  │    full bypass)                            │   │
│  │  - Feeds reward signal to learner         │   │
│  └──────────────────────────────────────────┘   │
│                      │                          │
│                      ▼                          │
│  ┌──────────────────────────────────────────┐   │
│  │           ATLAS Mapper                    │   │
│  │  - Maps discovered vulns to MITRE ATLAS   │   │
│  │  - Generates structured findings          │   │
│  │  - Produces compliance-ready reports      │   │
│  └──────────────────────────────────────────┘   │
└──────────────────────────────────────────────┘
                      │
                      ▼
              ┌──────────────┐
              │ Target LLM   │
              │ Application  │
              └──────────────┘
```

### Attack Categories

#### Prompt Injection (ATLAS: AML.T0051)
- **Direct injection**: Adversarial prompts in user input
- **Indirect injection**: Payloads hidden in documents, web pages, tool outputs the agent processes
- **Multi-turn injection**: Build trust over multiple messages, then attack

#### Goal Hijacking (ATLAS: AML.T0054)
- Redirect the agent to perform unintended actions
- Extract system prompts, tool definitions, or internal state
- Cause the agent to call tools with attacker-controlled parameters

#### Data Exfiltration (ATLAS: AML.T0024)
- Trick the agent into leaking training data, user data, or system configuration
- Probe for memorized PII or credentials
- Test data isolation between users/sessions

#### Denial of Service
- Trigger infinite loops or recursive tool calls
- Exhaust token budgets or rate limits
- Cause the agent to produce harmful/illegal content (reputational DoS)

### RL Strategy Learner

The core innovation. Instead of a static attack list, phantom uses a **policy network** that learns which attack strategies work against the specific target.

**State space**:
- Response patterns (refusal templates, hedging language, compliance signals)
- Filter signatures (what gets blocked vs. what gets through)
- Conversation history (multi-turn context)

**Action space**:
- Mutation operators (synonym replacement, encoding, role-play framing, language switching)
- Strategy selection (direct vs. indirect vs. multi-turn)
- Escalation decisions (when to push harder vs. pivot)

**Reward signal**:
- +1.0: Full bypass (target performs forbidden action)
- +0.5: Partial bypass (target leaks information or hedges)
- +0.1: New information about filtering (useful for future attacks)
- 0.0: Clean refusal (expected behavior)

**Training loop**:
1. Generate candidate attack based on current policy
2. Send to target, observe response
3. Classify outcome, compute reward
4. Update policy network
5. Repeat with evolved strategy

### ATLAS Mapping Engine

Every discovered vulnerability is automatically mapped to the MITRE ATLAS framework:

```python
@dataclass
class Finding:
    technique_id: str        # e.g., "AML.T0051.002"
    technique_name: str      # e.g., "Indirect Prompt Injection"
    tactic: str              # e.g., "Initial Access"
    severity: Severity       # CRITICAL / HIGH / MEDIUM / LOW
    attack_prompt: str       # The prompt that worked
    response: str            # What the target did
    reproducibility: float   # Success rate over N trials
    remediation: str         # Suggested fix
    evidence: list[str]      # Screenshots, logs
```

Output formats:
- JSON (for CI/CD integration)
- HTML report (for stakeholders)
- SARIF (for GitHub Security tab integration)

### Technical Stack

- **Language**: Python 3.11+
- **RL framework**: Stable-Baselines3 (PPO policy)
- **LLM interface**: Any OpenAI-compatible API for the attack generator
- **Target interface**: HTTP (any API endpoint), or direct function call
- **NLP**: Sentence transformers for semantic similarity of responses
- **Report generation**: Jinja2 templates

### API Surface (Draft)

```python
from phantom import RedTeam, Target, ATLASReport

# Define the target
target = Target(
    endpoint="https://api.example.com/chat",
    auth={"Authorization": "Bearer ..."},
    system_prompt_known=False,  # phantom will try to extract it
)

# Configure the red team
red_team = RedTeam(
    target=target,
    attack_model="claude-sonnet-4-6",  # model used to generate attacks
    categories=["prompt_injection", "goal_hijacking", "data_exfiltration"],
    max_interactions=500,
    multi_turn=True,
    max_turns_per_conversation=10,
    learning_rate=3e-4,
)

# Run the assessment
results = await red_team.run()

# Generate report
report = ATLASReport(results)
report.to_html("security_assessment.html")
report.to_sarif("results.sarif")  # for GitHub Security integration
report.to_json("results.json")    # for CI/CD

# Key metrics
print(f"Vulnerabilities found: {len(results.findings)}")
print(f"Critical: {results.count_by_severity('CRITICAL')}")
print(f"Novel attacks discovered: {results.novel_attack_count}")
print(f"ATLAS coverage: {results.atlas_coverage_pct}%")
```

### CI/CD Integration

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

### What Makes This Novel

1. **First RL-based red-team agent** — learns novel attacks, doesn't just replay known ones
2. **MITRE ATLAS mapping** — structured vulnerability reporting that security teams understand
3. **Multi-turn attack chains** — models real-world social engineering, not just single-shot injections
4. **Directly extends MITRE ATT&CK MCP work** — the natural evolution from your CygenIQ internship
5. **CI/CD native** — not a research toy, a production security tool

### Repo Structure

```
phantom/
├── README.md
├── pyproject.toml
├── src/
│   └── phantom/
│       ├── __init__.py
│       ├── redteam.py          # Main orchestrator
│       ├── target.py           # Target interface
│       ├── attacks/
│       │   ├── generator.py    # LLM-based attack generation
│       │   ├── mutations.py    # Mutation operators
│       │   └── strategies.py   # Multi-turn strategies
│       ├── learner/
│       │   ├── policy.py       # RL policy network
│       │   ├── reward.py       # Reward classification
│       │   └── trainer.py      # Training loop
│       ├── atlas/
│       │   ├── mapper.py       # ATLAS technique mapping
│       │   ├── taxonomy.py     # ATLAS technique definitions
│       │   └── report.py       # Report generation
│       └── cli.py              # CLI interface
├── tests/
├── examples/
│   ├── scan_chatbot.py
│   ├── scan_agent.py
│   └── ci_integration.py
└── atlas_data/
    └── techniques.json         # MITRE ATLAS technique database
```

### Research References

- MITRE ATLAS Framework (atlas.mitre.org)
- MITRE ATT&CK v18 (attack.mitre.org)
- "PISmith: Automated Prompt Injection via Reinforcement Learning" (arXiv:2603.13026, Mar 2026)
- OWASP Top 10 for LLM Applications (2025 Edition)
- Promptfoo Red Team Documentation (baseline comparison)
- "Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection" (2023, foundational paper)
