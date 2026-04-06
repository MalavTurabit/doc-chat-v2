def reconstruct_as_txt(
    full_text:    str,
    edit_history: list[dict],
) -> str:
    """
    Apply all edits from edit_history to full_text and return
    the updated document as a plain string.

    Strategy: replace original_text with edited_text using
    char offsets. Work backwards through edits so earlier
    offsets stay valid after each replacement.
    """
    if not edit_history:
        return full_text

    text = full_text

    # sort edits by start_char descending so replacements
    # don't shift offsets of earlier edits
    sorted_edits = sorted(edit_history, key=lambda e: e["start_char"], reverse=True)

    for edit in sorted_edits:
        original = edit["original_text"]
        edited   = edit["edited_text"]
        start    = edit["start_char"]
        end      = edit["end_char"]

        # verify the slice still matches
        current_slice = text[start:end]

        if current_slice == original:
            # clean offset match — replace directly
            text = text[:start] + edited + text[end:]
        elif original[:60] in text:
            # fallback — find by content if offset drifted
            text = text.replace(original, edited, 1)
        else:
            # last resort — find first 60 chars and replace
            snippet = original[:60]
            idx = text.find(snippet)
            if idx != -1:
                text = text[:idx] + edited + text[idx + len(original):]

    return text