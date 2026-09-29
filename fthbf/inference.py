"""Small CPU evaluation entry points shared by validation and quick-start runs."""
import torch
from solutions.trihybrid_WMMSE import GNN, TriHybridWMMSE
from solutions.utils import channelDataset, HybridPrecodingBFGS, checkHybridConstraints, computeWSR
from .runtime import checkpoint_path, load_artifact


def load_models(parameters, layers, path=None):
    nt, nc, mk, streams = (parameters[k] for k in ("Nt", "Nc", "Mk", "Ns_k"))
    input_dim = 2 * mk * nt * nc + 1
    ckpt = load_artifact(path or checkpoint_path(layers))
    if ckpt["I_max"] != layers or ckpt["in_dim"] != input_dim:
        raise ValueError("Checkpoint layer count/input dimension does not match the evaluation configuration")
    gamma = torch.nn.ModuleList([GNN(input_dim, 2 * streams ** 2 + 4, 2 * streams ** 2,
                                   num_layers=3, dp_rate=0.1) for _ in range(layers)])
    u = torch.nn.ModuleList([GNN(input_dim, nt + 10, nt, num_layers=3, dp_rate=0.1)
                            for _ in range(layers)])
    for index in range(layers):
        gamma[index].load_state_dict(ckpt["GNNs_gamma_state_dict"][index])
        u[index].load_state_dict(ckpt["GNNs_u_state_dict"][index])
    gamma.eval()
    u.eval()
    return gamma, u


def make_solver(instance, parameters):
    nt, nc, streams, mk = (parameters[k] for k in ("Nt", "Nc", "Ns_k", "Mk"))
    k = instance["K"]
    p = parameters["P"]
    u = [torch.ones(nc, 1, dtype=torch.cfloat) for _ in range(nt)]
    v = [torch.ones(streams * k, 1, dtype=torch.cfloat) for _ in range(nt)]
    for index in range(nt):
        norm = torch.linalg.norm(v[index])
        if norm > torch.sqrt(p):
            v[index] = v[index] / norm * torch.sqrt(p)
    return TriHybridWMMSE(instance["H"], [streams] * k, [mk] * k,
                         [parameters["sigma2"]] * k, [1.0] * k, [p] * nt,
                         parameters["eta"], u, v)


def corrections(instance, parameters, models):
    sample = channelDataset([instance], parameters)[0]
    graph_id = torch.zeros(instance["K"], dtype=torch.int32)
    num_graphs = torch.tensor(1, dtype=torch.int32)
    gamma, u = models
    dg, du = [], []
    for a, b in zip(gamma, u):
        dg.append(a(sample["node_feature"], sample["edge_index"]))
        du.append(b(sample["node_feature"], sample["edge_index"], graph_id, num_graphs).reshape(parameters["Nt"], 1))
    return dg, du


@torch.no_grad()
def evaluate_instance(instance, parameters, *, layers=2, models=None, rf_chains=10):
    # Preserve the notebook's order: GNN inference precedes solver initialization.
    dg, du = corrections(instance, parameters, models) if models is not None else (None, None)
    solver = make_solver(instance, parameters)
    if models is None:
        trace, fa, fd, *_ = solver.run_classical(I_max=layers, verbose=False)
    else:
        trace, fa, fd, *_ = solver.run_unfolded(dg, du, I_max=layers, verbose=False)
    frf, fbb = HybridPrecodingBFGS(fd, rf_chains).solve(max_iter=100, use_B0=True, verbose=False)
    _, power_ok, info = checkHybridConstraints(frf, fbb, parameters["P"], tol=1e-6, verbose=False)
    if not power_ok:
        fbb = fbb / torch.sqrt(torch.tensor(info["max_papc_ratio"]))
    phase_ok, strict_power_ok, info = checkHybridConstraints(frf, fbb, parameters["P"], tol=2e-6, verbose=False)
    # The legacy checker uses a strict <= 1 test for power, irrespective of tol.
    # Recheck single-precision projection with an explicit relative tolerance.
    power_ok = info["max_papc_ratio"] <= 1 + 2e-6
    rate, _ = computeWSR(fa, frf, fbb, instance["H"], parameters)
    if not torch.isfinite(rate) or not phase_ok or not power_ok:
        raise RuntimeError(f"Invalid evaluation result: WSR={rate}, phase_ok={phase_ok}, power_ratio={info['max_papc_ratio']}")
    return {"wsr": rate.item(), "virtual_wsr": trace[-1].item(),
            "phase_ok": bool(phase_ok), "power_ok": bool(power_ok), "strict_power_ok": bool(strict_power_ok),
            "max_power_ratio": float(info["max_papc_ratio"])}
