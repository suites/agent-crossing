# Persona Seeds

Current recommendation for this repository:

- Runtime source of truth: JSON (`*.json`)
- Runtime shape: Parsed `AgentPersona` object via `PersonaLoader`

JSON schema (version 1):

- root fields: `version`, `agent`, `fixed_persona`, `extended_persona`, `seed_memories`
- `agent`: `agent_id`, `name`, `age`, `traits`
- `fixed_persona`: `identity_stable_set`
- `extended_persona`: `lifestyle_and_routine`, `current_plan_context`
- `seed_memories`: list of `{ content, importance }`

Loading behavior:

- `PersonaLoader.load("<name>")` loads `<name>.json` only.
- If JSON is missing, loading fails with `PersonaLoadError`.
- `load_all()` loads all `*.json` files except `*.sample.json`.

Sample:

- Use `persona.sample.json` as the canonical template when creating new persona files.
