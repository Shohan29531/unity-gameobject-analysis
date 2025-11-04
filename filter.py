import json

STATIC_FILE = "static_metadata.jsonl"

def load_jsonl(path):
    """Yield each JSON object from a newline-delimited JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"⚠️ Skipping malformed line: {e}")

def find_matching(criteria_fn, show_full=False):
    """
    Print and count entries that satisfy the criteria.
    If show_full=True, prints the full JSON object.
    """
    matches = []
    for obj in load_jsonl(STATIC_FILE):
        if criteria_fn(obj):
            matches.append(obj)

    print(f"\n✅ Found {len(matches)} matching GameObjects\n")

    for i, obj in enumerate(matches, 1):
        if show_full:
            # print entire JSON record
            print(json.dumps(obj, indent=2))
        else:
            # print selected summary fields
            print(f"{i:03d}. {obj.get('ObjectName')} "
                  f"(Type={obj.get('Type')}, Level={obj.get('Level')}, Scene={obj.get('Scene')})")

    return matches


# -------------------------
# Example criteria function
# -------------------------
def is_low_level_physicsbody(obj):
    """
    Example filter:
    include GameObjects up to level 2 (inclusive)
    and only those whose Type == 'PhysicsBody'
    """
    return (obj.get("Level", 999) <= 1 and obj.get("Type") == "PhysicsBody" 
            and obj.get("InView") == True and obj.get("Active") == True)


# Run the example
if __name__ == "__main__":
    # Change show_full=True to see all fields printed
    find_matching(is_low_level_physicsbody, show_full=False)
