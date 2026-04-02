import json
import sys
from collections import Counter
from ingestion.parser import extract


def test_offsets(result: dict):
    full = result["full_text"]
    errors = []
    for i, b in enumerate(result["blocks"]):
        extracted = full[b["start_char"]:b["end_char"]]
        if extracted != b["text"]:
            errors.append(
                f"Block {i} offset mismatch:\n"
                f"  expected : {b['text'][:60]}\n"
                f"  got      : {extracted[:60]}"
            )
    return errors


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python -m ingestion.test_parser <file_path>")
        sys.exit(1)

    path = sys.argv[1]
    print(f"\nExtracting: {path}")
    result = extract(path)

    print(f"\n=== {result['filename']} ===")
    print(f"doc_id     : {result['doc_id']}")
    print(f"extension  : {result['ext']}")
    print(f"blocks     : {len(result['blocks'])}")
    print(f"total chars: {len(result['full_text'])}")

    print(f"\n--- Block type summary ---")
    counts = Counter(b["type"] for b in result["blocks"])
    for t, n in counts.items():
        print(f"  {t}: {n}")

    print(f"\n--- First 3 blocks ---")
    for b in result["blocks"][:3]:
        print(json.dumps(b, indent=2))

    print(f"\n--- Offset validation ---")
    errors = test_offsets(result)
    if errors:
        for e in errors:
            print(f"  FAIL  {e}")
    else:
        print("  All offsets valid.")