"""Check model loading, inference, and hybrid-precoder constraints on a few instances."""
import argparse
import torch
from fthbf.runtime import DATA_DIR, RESULT_ROOT, load_artifact, seed_everything, environment, write_json
from fthbf.inference import load_models, evaluate_instance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=int, default=2)
    parser.add_argument("--layers", type=int, default=2)
    args = parser.parse_args()
    if args.instances < 1 or args.layers < 1:
        parser.error("instances and layers must be positive")
    seed_everything(42)
    parameters = load_artifact(DATA_DIR / "parameters.pt")
    data = load_artifact(DATA_DIR / "probInsts_test.pt")["ProbInsts"]
    models = load_models(parameters, args.layers)
    records = []
    for index, instance in enumerate(data[:args.instances]):
        classical = evaluate_instance(instance, parameters, layers=args.layers)
        unfolded = evaluate_instance(instance, parameters, layers=args.layers, models=models)
        repeat = evaluate_instance(instance, parameters, layers=args.layers, models=models)
        if abs(unfolded["wsr"] - repeat["wsr"]) > 1e-5:
            raise AssertionError("Evaluation with dropout disabled did not repeat")
        records.append({"instance": index, "classical": classical, "unfolded": unfolded})
    destination = RESULT_ROOT / "smoke-check.json"
    write_json(destination, {"environment": environment(), "layers": args.layers, "results": records})
    print(f"Validated {len(records)} instances. Results: {destination}")


if __name__ == "__main__":
    main()
