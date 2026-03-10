## Planning Context

- Agent: $agent_name
- Current time: $current_time

## Input Hourly Plan

$hourly_plan_lines

## Task

Generate an executable minute plan for the current phase.

## Requirements

- Return the result in `items`.
- Each item must include all required fields: `start_time`, `duration_minutes`, `location`, `action_content`.
- `start_time` must be an ISO 8601 datetime string with minute precision (`seconds=00`).
- `duration_minutes` must be an integer in range 5 to 15.
- This plan may use minute-level start times, but do not include seconds or sub-minute precision.
- Keep items ordered from earlier to later time.
- Focus on concrete actions that can be executed immediately.
- Do not add numbering, bullets, markdown, explanatory text, or additional keys.

## Output Contract

Return strict JSON only with this exact shape and no extra text: $json_shape
