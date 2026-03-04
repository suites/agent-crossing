## Guard Trigger

Your previous reaction was too similar to recent dialogue.

## Constraint

- Generate a different reaction while preserving intent.
- N-gram overlap with each sentence below must be <= 50%.
- Previous candidate: $previous_candidate
- Recent dialogue sentences:
  $recent_dialogue_lines

## Output Contract

Return strict JSON only with this exact shape and no extra text: $json_shape
