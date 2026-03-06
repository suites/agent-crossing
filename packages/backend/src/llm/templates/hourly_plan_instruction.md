## Planning Context

- Agent: $agent_name
- Date: $today_date_text

## Input Day Plan

$day_plan_lines

## Task

Convert the day plan into a chronological hourly plan.

## Requirements

- Return the result in `items`.
- Each item must include all required fields: `start_time`, `duration_minutes`, `location`, `action_content`.
- `start_time` must be an ISO 8601 datetime string.
- `duration_minutes` must be a positive integer.
- Keep items ordered from earlier to later time.
- Keep each item aligned with the given day plan context.
- Do not add numbering, bullets, markdown, explanatory text, or additional keys.

## Output Contract

Return strict JSON only with this exact shape and no extra text: $json_shape
