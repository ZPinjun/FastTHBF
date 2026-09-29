"""Generate channel tensors and evaluation datasets."""
import argparse
from pathlib import Path
from fthbf.data import generate
from fthbf.runtime import DATA_DIR, SCENARIO_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=["train", "test", "generalization", "all"], default="test")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--scenario-dir", type=Path, default=SCENARIO_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    generate(args.group, args.data_dir, args.scenario_dir, args.overwrite)


if __name__ == "__main__":
    main()
