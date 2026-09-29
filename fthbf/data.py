"""Generate the public experiments' channels and problem instances."""

from pathlib import Path
import hashlib
import json

import numpy as np
import torch

from .runtime import DATA_DIR, SCENARIO_DIR, environment, load_artifact, read_config, write_json


def system_parameters(cfg):
    nt, nc = cfg["Nt"], cfg["Nc"]
    wavelength = 3e8 / cfg["carrier_hz"]
    positions = torch.arange(1, nc + 1) * wavelength / 2
    eta_i = torch.exp(-positions * (1j * 2 * torch.pi / wavelength + cfg["attenuation_per_m"])).reshape(-1, 1)
    return {"Nt": nt, "Nc": nc, "BW": cfg["bandwidth_hz"], "Ns_k": cfg["streams_per_user"],
            "Mk": cfg["ue_antennas"], "P": torch.tensor(10 ** (cfg["power_dbm"] / 10)),
            "sigma2": 10 ** ((10 * torch.log10(torch.tensor(cfg["bandwidth_hz"])) - 174) / 10),
            "eta": [eta_i for _ in range(nt)]}


def channel_parameters(dm, cfg):
    params = dm.ChannelParameters()
    params.bs_antenna.shape = [cfg["Nt"], cfg["Nc"]]
    params.bs_antenna.spacing = 0.5
    params.bs_antenna.rotation = [0, 0, 0]
    params.ue_antenna.shape = [2, 2]
    params.ue_antenna.spacing = 0.5
    params.ue_antenna.rotation = [0, 0, 0]
    params.ofdm.subcarriers = cfg["subcarriers"]
    params.ofdm.bandwidth = cfg["bandwidth_hz"]
    params.ofdm.selected_subcarriers = cfg["selected_subcarriers"]
    params.doppler = False
    params.freq_domain = True
    return params


def problem_instances(channels, count, seed):
    instances, selections = [], []
    for index in range(count):
        generator = torch.Generator().manual_seed(seed + index)
        k = torch.randint(3, 6, (1,), generator=generator).item()
        selected = torch.randperm(channels.shape[0], generator=generator)[:k]
        h = torch.as_tensor(channels[selected], dtype=torch.cfloat).squeeze(-1)
        instances.append({"H": [h[j] for j in range(k)], "K": k})
        selections.append(selected.tolist())
    return {"ProbInsts": instances, "BASE_SEED": seed, "num_instances": count,
            "selected_channel_indices": selections}


def load_channels(scenario, cfg, scenario_dir=SCENARIO_DIR, max_users=None):
    import deepmimo as dm
    if dm.__version__ != "4.0.0b11":
        raise RuntimeError("This pipeline was validated with deepmimo==4.0.0b11. Install requirements-data.txt.")
    folder = Path(scenario_dir) / scenario
    if not (folder / "params.json").is_file():
        raise FileNotFoundError(f"Missing scenario {scenario}. Run python -m scripts.download_scenarios --group all.")
    dm.config.set("scenarios_folder", str(Path(scenario_dir).resolve()))
    dataset = dm.load(scenario)[cfg["dataset_index"]]
    active = dataset.get_active_idxs()
    dataset = dataset.subset(active)
    retained = [i for i in range(dataset.n_ue)
                if dataset["power"][i][0] > cfg["power_threshold_dbm"]]
    if max_users is not None:
        retained = retained[:max_users]
    original_indices = np.asarray(active)[retained]
    channels = dataset.subset(retained).compute_channels(channel_parameters(dm, cfg))
    metadata = {"scenario": scenario, "dataset_index": cfg["dataset_index"],
                "active_users": len(active), "retained_users": len(retained),
                "original_receiver_indices": original_indices.tolist(),
                "scenario_params": json.loads((folder / "params.json").read_text()),
                "channel_shape": list(channels.shape), "dtype": str(channels.dtype)}
    return torch.from_numpy(channels), metadata


def generate(group="test", data_dir=DATA_DIR, scenario_dir=SCENARIO_DIR, overwrite=False):
    cfg = read_config("data.json")
    scenarios = read_config("scenarios.json")
    destination = Path(data_dir)
    destination.mkdir(parents=True, exist_ok=True)
    parameters = system_parameters(cfg)
    parameter_path = destination / "parameters.pt"
    if parameter_path.exists():
        old = load_artifact(parameter_path)
        for key, value in parameters.items():
            a, b = old[key], value
            if key == "eta":
                equal = len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))
            else:
                equal = bool(torch.equal(a, b)) if torch.is_tensor(b) else a == b
            if not equal:
                raise ValueError(f"Existing parameters.pt differs in {key}; use a new data directory.")
    else:
        torch.save(parameters, parameter_path)
    jobs = []
    if group in ("train", "all"):
        jobs.append(("channels_deepMIMO_train.pt", scenarios["groups"]["train"], None))
    if group in ("test", "all"):
        jobs.append(("channels_deepMIMO_test.pt", scenarios["groups"]["test"], "probInsts_test.pt"))
    if group in ("generalization", "all"):
        jobs.extend((f"channels_{city}.pt", [scenario], f"probInsts_{city}.pt")
                    for city, scenario in scenarios["cities"].items())
    for channel_file, scenario_names, instance_file in jobs:
        target = destination / channel_file
        if target.exists() and not overwrite:
            channels = load_artifact(target)
            print(f"Using existing {target.name}")
        else:
            parts, records = [], []
            for scenario in scenario_names:
                part, record = load_channels(scenario, cfg, scenario_dir)
                parts.append(part)
                records.append(record)
            channels = torch.cat(parts, dim=0)
            torch.save(channels, target)
            write_json(target.with_suffix(".json"), {
                "config": cfg, "environment": environment(), "scenarios": records,
                "shape": list(channels.shape),
                "tensor_sha256": hashlib.sha256(channels.numpy().tobytes()).hexdigest()})
        if instance_file:
            output = destination / instance_file
            if output.exists() and not overwrite:
                print(f"Skipping existing {output.name}")
                continue
            count = cfg["test_instances"] if group != "generalization" and instance_file == "probInsts_test.pt" else cfg["city_instances"]
            torch.save(problem_instances(channels, count, cfg["seed"]), output)
            print(f"Saved {output.name}: {count} instances")
