"""Download only the DeepMIMO scenarios required by an experiment group."""
import argparse
from pathlib import Path
from fthbf.runtime import SCENARIO_DIR, read_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=["train", "test", "generalization", "all"], default="test")
    parser.add_argument("--scenario-dir", type=Path, default=SCENARIO_DIR)
    parser.add_argument("--list", action="store_true", help="List required scenes without downloading.")
    args = parser.parse_args()
    groups = read_config("scenarios.json")["groups"]
    names = list(dict.fromkeys(sum(groups.values(), []))) if args.group == "all" else groups[args.group]
    if args.list:
        print("\n".join(names))
        return
    import deepmimo as dm
    args.scenario_dir.mkdir(parents=True, exist_ok=True)
    dm.config.set("scenarios_folder", str(args.scenario_dir.resolve()))
    for name in names:
        if (args.scenario_dir / name / "params.json").is_file():
            print(f"Skipping existing scenario: {name}")
            continue
        dm.download(name, output_dir=str(args.scenario_dir.resolve()))
        if not (args.scenario_dir / name / "params.json").is_file():
            raise RuntimeError(f"Download did not produce {name}/params.json. See docs/DATA.md.")


if __name__ == "__main__":
    main()
