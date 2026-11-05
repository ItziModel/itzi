# Issue tracker: Local Markdown

Issues and specs for this repo live as Markdown files in `.scratch/`.

## Conventions

- One feature per directory: `.scratch/<feature-slug>/`.
- The spec is `.scratch/<feature-slug>/spec.md`.
- Implementation issues are one file per ticket at `.scratch/<feature-slug>/issues/<NN>-<slug>.md`, numbered from `01`.
- Triage state is a `Status:` line near the top of each issue file. Use the role strings in `docs/agents/triage-labels.md`.
- Append comments and conversation history under a `## Comments` heading.

## When a skill says "publish to the issue tracker"

Create a new file under `.scratch/<feature-slug>/`, creating the directory if needed.

## When a skill says "fetch the relevant ticket"

Read the referenced file. The user will normally pass the path or issue number.

## Wayfinding operations

Used by `/wayfinder`. The map is a file with one child file per ticket.

- **Map**: `.scratch/<effort>/map.md`, holding Notes / Decisions-so-far / Fog.
- **Child ticket**: `.scratch/<effort>/issues/NN-<slug>.md`, numbered from `01`, with the question in the body. A `Type:` line records `research`/`prototype`/`grilling`/`task`; a `Status:` line records `claimed`/`resolved`.
- **Blocking**: a `Blocked by: NN, NN` line near the top. A ticket is unblocked when every listed file is `resolved`.
- **Frontier**: scan the issues directory for open, unblocked, unclaimed tickets; first by number wins.
- **Claim**: set `Status: claimed` and save before work.
- **Resolve**: append the answer under `## Answer`, set `Status: resolved`, and append a gist and link to Decisions-so-far in `map.md`.
