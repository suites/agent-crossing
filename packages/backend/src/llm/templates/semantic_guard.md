## Guard Trigger

Your previous reaction violated semantic repetition guard ($level).

## Guard Metrics

- max_similarity=$max_similarity
- soft_threshold=$soft_threshold
- hard_threshold=$hard_threshold

## Rewrite Task

- Generate a meaningfully different utterance while keeping conversation natural.
- Previous candidate: $previous_candidate
- Recent self utterances to avoid semantically repeating:
  $semantic_history_lines

## Output Contract

Return strict JSON only with this exact shape and no extra text: $json_shape
