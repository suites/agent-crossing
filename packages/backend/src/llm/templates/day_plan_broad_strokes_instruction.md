Name: $agent_name (age: $age)
Innate traits: $innate_traits
$persona_background

On $yesterday_date_text, $agent_name 1) $yesterday_summary

Today is $today_date_text. Here is $agent_name's plan today in broad strokes: 1)

Framing reference (for style, not output format):
Name: Eddy Lin (age: 19)
Innate traits: friendly, outgoing, hospitable
On Tuesday February 12, Eddy 1) woke up and completed the morning routine at 7:00 am, [...] 6) got ready to sleep around 10 pm.
Today is Wednesday February 13. Here is Eddy's plan today in broad strokes: 1)

Requirements:

- Return 5 to 8 broad strokes.
- Each stroke must be a single concise sentence and behavior-oriented.
- Keep chronological flow from morning to night.
- Do not add numbering, bullets, markdown, or explanatory text.
- Do not add additional keys or fields.
- No duplicate or near-duplicate strokes.

Return JSON only with this shape: $json_shape
