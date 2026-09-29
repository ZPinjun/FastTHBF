"""Train from scratch or initialize from a shallower checkpoint."""
import argparse
from pathlib import Path
from fthbf.runtime import read_config
from fthbf.training import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="train_scratch.json")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--warm-start-layers", type=int)
    parser.add_argument("--smoke", action="store_true", help="Run one batch using four problem instances.")
    args = parser.parse_args()
    config = read_config(args.config)
    if args.layers is not None:
        config["layers"] = args.layers
    if args.warm_start_layers is not None:
        config["warm_start_layers"] = args.warm_start_layers
    if args.smoke:
        config.update(instances=4, batch_size=1, epochs=1, max_batches=1, eval_every=1)
    print(train(config, args.output_dir))


if __name__ == "__main__":
    main()
