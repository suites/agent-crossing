# Persona Seeds

Current recommendation for this repository:

- Authoring source: Markdown (`*.md`)
- Runtime shape: Parsed `AgentSeed` object via `SeedLoader`
- Optional future extension: add `*.seed.json` sidecar for strict schema validation without removing markdown authoring

Why this fits now:

- Existing persona files are already markdown and human-editable.
- `SeedLoader` already parses structured sections needed by `AgentBrain`.
- Runtime now consumes both stable identity context and current plan context when building retrieval queries.

If you add JSON sidecars later:

- Keep markdown as source of truth for writing.
- Keep JSON as machine-focused cache/schema artifact.
- Prefer loading JSON when present, fallback to markdown parser when absent.
