# Domain Docs

How engineering skills consume this repo's domain documentation.

## Before exploring, read these

- Root `GLOSSARY.md` for domain terms.
- Relevant ADRs under `docs/adr/`.

If these files do not exist, proceed silently. `/domain-modeling` creates them lazily when terms or decisions are resolved.

## Layout

Single-context: one root `GLOSSARY.md` and shared ADRs in `docs/adr/`.

## Use the glossary's vocabulary

Use glossary terms when naming domain concepts in issues, proposals, hypotheses, and tests.

If a concept is missing, reconsider whether the project uses it or note the gap for `/domain-modeling`.

## Flag ADR conflicts

Explicitly surface proposals that contradict an existing ADR, citing the ADR and explaining why the decision is worth reopening.
