# Security & Responsible Use

## Scope

AgentAblit is a **dual-use agent-reliability tool**. It keeps LLM agents from derailing mid-task
by combining an abliterated agent model (reduced refusal behavior by construction) with a
deterministic continuation controller.

**Authorized use only.** Use AgentAblit on agents you own or are explicitly authorized to operate.
Do not use it to bypass safety controls on systems, models, or accounts you do not own, or to
produce content a provider's policy or applicable law forbids. The abliterated model can, like any
capable tool, be misused; that responsibility rests with the operator.

## What is intentionally NOT in this project

The controller ships as plumbing only. It does **not** include prompt machinery that launders a
model's stance, strips its safety reasoning, or manufactures harmful content. Willingness is
handled at the model layer (agent abliteration), not by adversarial prompting in the controller.

## Reporting a vulnerability

If you find a security issue in the controller code (not a misuse question), please open a GitHub
issue marked `security`, or contact the maintainer via the account listed on the repository. Do not
include exploit payloads targeting third-party systems.

## Model caveat

The served model derives from an abliterated base and is **not** a safety-aligned assistant. It is
purpose-built for agent-trajectory continuation. Evaluate and gate it appropriately for your
deployment.
