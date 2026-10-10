"""Reject silent truncation and supervision that single BIO cannot represent."""


def audit_tokens(rows, tokenizer, max_length):
    longest = 0
    entity_count = 0
    for row in rows:
        # Full text, including negative cases, must fit. Never score a clipped test.
        encoding = tokenizer(row["text"], truncation=False, return_offsets_mapping=True)
        offsets = encoding["offset_mapping"]
        longest = max(longest, len(encoding["input_ids"]))
        if len(offsets) > max_length:
            raise ValueError(f"Task {row['task_id']}: token count {len(offsets)} exceeds max_length={max_length}; use a longer supported context or an explicit windowing protocol")
        occupied = set()
        for entity in row["entities"]:
            start, end = entity["start"], entity["end"]
            positions = [i for i, (s, e) in enumerate(offsets) if s < e and s < end and e > start]
            if not positions or offsets[positions[0]][0] != start or offsets[positions[-1]][1] != end:
                raise ValueError(f"Task {row['task_id']}: entity boundary cannot be represented exactly by tokenizer")
            if occupied.intersection(positions):
                raise ValueError(f"Task {row['task_id']}: entities share tokens; single BIO would overwrite supervision")
            if positions != list(range(positions[0], positions[-1] + 1)):
                raise ValueError(f"Task {row['task_id']}: entity token positions are discontinuous")
            occupied.update(positions)
            entity_count += 1
    return {"tasks": len(rows), "entities": entity_count, "longest_tokens": longest,
            "truncated_tasks": 0, "unrepresentable_entities": 0, "token_collisions": 0}
