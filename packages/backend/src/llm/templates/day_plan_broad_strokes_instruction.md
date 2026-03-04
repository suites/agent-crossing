## Persona Context

- Name: $agent_name (age: $age)
- Innate traits: $innate_traits
- Background: $persona_background

## Yesterday Recap

On $yesterday_date_text, $agent_name did the following:

- $yesterday_summary

## Today Plan Prompt

Today is $today_date_text. Draft $agent_name's plan today in broad strokes.

## Framing Reference

Framing reference (for style, not output format):

- Name: Eddy Lin (age: 19)
- Innate traits: friendly, outgoing, hospitable
- On Tuesday February 12, Eddy completed his morning routine at 7:00 am and got ready to sleep around 10:00 pm.
- Today is Wednesday February 13. Draft Eddy's plan today in broad strokes.

## Requirements

- Return 5 to 8 broad strokes.
- Each stroke must be a single concise sentence and behavior-oriented.
- Keep chronological flow from morning to night.
- Do not add numbering, bullets, markdown, or explanatory text.
- Do not add additional keys or fields.
- No duplicate or near-duplicate strokes.

## Output Contract

Return strict JSON only with this exact shape and no extra text: $json_shape
