## Persona Context

- Name: $agent_name (age: $age)
- Innate traits: $innate_traits
- Background: $persona_background

## Yesterday Recap

On $yesterday_date_text, $agent_name did the following:

- $yesterday_summary

## Today Plan Prompt

Today is $today_date_text. Draft $agent_name's structured day plan.

## Framing Reference

Framing reference (for style, not output format):

- Name: Eddy Lin (age: 19)
- Innate traits: friendly, outgoing, hospitable
- On Tuesday February 12, Eddy completed his morning routine at 7:00 am and got ready to sleep around 10:00 pm.
- Today is Wednesday February 13. Draft Eddy's structured day plan.

## Requirements

- Return 5 to 8 plan items in `items`.
- Each item must include all required fields: `start_time`, `end_time`, `location`, `action_content`.
- `start_time` and `end_time` must be ISO 8601 datetime strings with minute precision (`seconds=00`).
- `end_time` must be later than `start_time`.
- Use the same calendar date as `Today is ...` for every plan item unless an item clearly crosses past midnight.
- This is a broad-strokes day plan. Use natural human time spans and allow non-hour boundaries like `5:30 pm` when they fit the routine.
- Keep chronological flow from morning to night.
- Keep `location` and `action_content` behavior-oriented and non-empty.
- Do not add numbering, bullets, markdown, explanatory text, or additional keys.

## Output Contract

Return strict JSON only with this exact shape and no extra text: $json_shape
