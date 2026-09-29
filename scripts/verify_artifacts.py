"""Verify the expected release files against their SHA-256 manifest."""
import argparse
from pathlib import Path
from fthbf.runtime import DATA_DIR, CHECKPOINT_DIR, read_config, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=["weights", "data", "all"], default="all")
    args = parser.parse_args()
    manifest = read_config("artifacts.json")
    errors = []
    count = 0
    for group, directory in [("weights", CHECKPOINT_DIR), ("data", DATA_DIR)]:
        if args.group not in [group, "all"]:
            continue
        for entry in manifest[group]:
            path = directory / Path(entry["file"]).name
            if not path.is_file():
                errors.append(f"Missing: {path.name}")
            elif sha256(path) != entry["sha256"]:
                errors.append(f"Checksum mismatch: {path.name}")
            else:
                count += 1
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Verified {count} release files. Regenerated data may have different file hashes; compare values and metadata separately.")


if __name__ == "__main__":
    main()
