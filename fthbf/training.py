"""Configurable CPU training using the original unfolded WMMSE loss."""
import os
from pathlib import Path
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from solutions.trihybrid_WMMSE import GNN, TriHybridWMMSE
from solutions.utils import channelDataset, evaluate, init_kaiming, zero_last_linear
from .runtime import DATA_DIR, RESULT_ROOT, checkpoint_path, load_artifact, seed_everything, write_json, environment


def train(config, output_dir=None):
    config = dict(config)
    if config["instances"] < 2 or config["epochs"] < 1 or config["batch_size"] < 1 or config["eval_every"] < 1:
        raise ValueError("Training requires at least two instances and positive epochs, batch size, and eval_every")
    if not 0 < config["validation_fraction"] < 1:
        raise ValueError("validation_fraction must lie between zero and one")
    if config["max_batches"] is not None and config["max_batches"] < 1:
        raise ValueError("max_batches must be positive or null")
    output_dir = Path(output_dir) if output_dir else RESULT_ROOT / "training" / f"L{config['layers']}-seed{config['seed']}"
    checkpoint_file = output_dir / f"unfolded_wmmse_model_{config['layers']}Iter.pt"
    if checkpoint_file.exists() and not config["overwrite"]:
        raise FileExistsError(f"{checkpoint_file} already exists; choose a new output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "run.json", {"config": config, "environment": environment()})
    I_max = config["layers"]
    I_prev = config["warm_start_layers"]

    # =====================================================
    # Checkpoint saving function
    # =====================================================
    def save_checkpoint(epoch, GNNs_gamma, GNNs_u, optimizer, path):
        torch.save({
            "epoch": epoch,
            "GNNs_gamma_state_dict": [model.state_dict() for model in GNNs_gamma],
            "GNNs_u_state_dict": [model.state_dict() for model in GNNs_u],
            "optimizer_state_dict": optimizer.state_dict(),
            "I_max": I_max,
            "in_dim": in_dim,
            "training_config": config,
            "scheduler_state_dict": scheduler.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "loader_rng_state": train_loader.generator.get_state()
        }, path)

    # =====================================================
    # Basic setup
    # =====================================================
    BASE_SEED = config["seed"]
    seed_everything(BASE_SEED)
    cdtype = torch.cfloat
    NUM_EPOCHS = config["epochs"]
    BATCH_SIZE = config["batch_size"]
    LR = config["learning_rate"]

    # =====================================================
    # Read deepMIMO channels
    # =====================================================
    SAVE_DIR = DATA_DIR
    parameters = load_artifact(os.path.join(SAVE_DIR, "parameters.pt"))
    channels = load_artifact(os.path.join(SAVE_DIR, "channels_deepMIMO_train.pt"))
    print(f"[Info] Loaded channels shape: {channels.shape}")
    Nt = parameters["Nt"]
    Nc = parameters["Nc"]
    P = parameters["P"]
    Mk = parameters["Mk"]
    eta = parameters["eta"]
    Ns_k = parameters["Ns_k"]
    sigma2 = parameters["sigma2"]

    # =====================================================
    # Generate dataset
    # =====================================================
    # Generate a set of problem instances
    num_ProbInsts = config["instances"]
    ProbInsts = []
    for s_idx in tqdm(range(num_ProbInsts), desc="Generating Problem Instances"):
        g = torch.Generator()
        g.manual_seed(BASE_SEED + s_idx)
        K = torch.randint(3, 6, (1,), generator=g).item()                                # {3,4,5}
        sel_idxs = torch.randperm(channels.shape[0], generator=g)[:K]
        selected_channels = torch.tensor(channels[sel_idxs], dtype=cdtype).squeeze(-1)   # (K, Mk, Nt*Nc)
        H = [selected_channels[k] for k in range(K)]        # list of (Mk, Nt*Nc)
        Inst = {"H": H, "K": K}
        ProbInsts.append(Inst)

    # =====================================================
    # Define and initialize deep GNN model
    # =====================================================
    in_dim = 2 * ProbInsts[0]["H"][0].numel() + 1 
    GNNs_gamma = torch.nn.ModuleList()
    GNNs_u = torch.nn.ModuleList()
    Ns_k = parameters["Ns_k"]
    print(f"[Info] GNN input dimension: {in_dim}")
    for _ in range(I_max):
        GNN_gamma = GNN(
            in_dim=in_dim,
            hidden_dim=2*Ns_k**2+4,
            out_dim=2*Ns_k**2,
            num_layers=3,
            dp_rate=0.1
        )
        GNN_u = GNN(
            in_dim=in_dim,
            hidden_dim=Nt+10,
            out_dim=Nt,
            num_layers=3,
            dp_rate=0.1
        )
        GNN_gamma.apply(init_kaiming)
        GNN_u.apply(init_kaiming)
        zero_last_linear(GNN_gamma)
        zero_last_linear(GNN_u)
        GNNs_gamma.append(GNN_gamma)
        GNNs_u.append(GNN_u)

    # Optionally initialize the first layers from an existing shallower model.
    if I_prev is not None:
        if not 0 < I_prev < I_max:
            raise ValueError("warm_start_layers must be positive and smaller than layers")
        checkpoint = load_artifact(checkpoint_path(I_prev))
        if checkpoint["I_max"] != I_prev or checkpoint["in_dim"] != in_dim:
            raise ValueError("Warm-start checkpoint architecture does not match")
        for i in range(I_prev):
            GNNs_gamma[i].load_state_dict(checkpoint["GNNs_gamma_state_dict"][i])
            GNNs_u[i].load_state_dict(checkpoint["GNNs_u_state_dict"][i])

    # =====================================================
    # Training 
    # =====================================================
    # Create dataLoader
    dataset = channelDataset(ProbInsts, parameters)
    num_test = max(1, int(config["validation_fraction"] * len(dataset)))       # 0.25% for testing
    num_train = len(dataset) - num_test         # 99.75% for training
    train_dataset, test_dataset = random_split(dataset, [num_train, num_test], generator=torch.Generator().manual_seed(BASE_SEED))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(BASE_SEED + 1), collate_fn=lambda x: x)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: x)

    # Collect trainable parameters
    params = []
    for model in GNNs_gamma:
        params += list(model.parameters())
    for model in GNNs_u:
        params += list(model.parameters())

    # Define optimizer
    optimizer = torch.optim.Adam(params, lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, min_lr=1e-6)

    # Evaluate on test set every VAL_EVERY batches
    EVAL_EVERY = config["eval_every"]

    # Training loop
    ckpt_dir = output_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"unfolded_wmmse_model_{I_max}Iter.pt")
    write_json(output_dir / "split.json", {"train": train_dataset.indices, "validation": test_dataset.indices})
    GNNs_gamma.train()
    GNNs_u.train()  
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        epoch_rate = 0.0
        num_samples = 0
        batches_completed = 0
        for batch_idx, batch in enumerate(train_loader):
            if config["max_batches"] is not None and batch_idx >= config["max_batches"]:
                break

            optimizer.zero_grad()
            batch_loss = 0.0
            batch_rate = 0.0

            for pro_inst in batch:
                K = pro_inst["K"]
                node_features = pro_inst["node_feature"]
                edge_index = pro_inst["edge_index"]

                # ----- GNN output -----
                delta_gamma_list = []
                delta_u_list = []
                graph_id = torch.zeros(K, dtype=torch.int32)
                num_graphs = torch.tensor(1, dtype=torch.int32)
                for i in range(I_max):
                    # delta_gamma
                    delta_gamma = GNNs_gamma[i](node_features, edge_index)
                    delta_gamma_list.append(delta_gamma)
                    # delta_u
                    delta_u = GNNs_u[i](node_features, edge_index, graph_id, num_graphs)
                    delta_u_mt = delta_u.reshape(Nt, 1)
                    delta_u_list.append(delta_u_mt)

                # ----- WMMSE -----
                H_list = pro_inst["H_list"]
                # Initialize u0 and v0
                Ns = Ns_k * K
                u0 = [torch.ones(Nc, 1, dtype=cdtype)] * Nt
                v0 = [torch.ones(Ns, 1, dtype=cdtype)] * Nt
                for n in range(Nt):    # normalize v0
                    norm_vn = torch.linalg.norm(v0[n])
                    if norm_vn > torch.sqrt(P):
                        v0[n] = v0[n] / norm_vn * torch.sqrt(P)
                # Solver
                solver = TriHybridWMMSE(
                    H_list=H_list,
                    Ns_k=[Ns_k] * K,
                    Mk=[Mk] * K,
                    sigma2=[sigma2] * K,
                    beta=[1.0] * K,
                    P=[P] * Nt,
                    eta=eta,
                    u_init=u0,
                    v_init=v0,
                )
                rate_trace, *_ = solver.run_unfolded(
                    dgamma_list=delta_gamma_list,
                    du_list=delta_u_list,
                    I_max=I_max,
                    verbose=False
                )
                rate_final = rate_trace[-1]
                loss = -rate_final     
                batch_loss += loss
                batch_rate += rate_final
                num_samples += 1

            batch_loss = batch_loss / len(batch)
            batch_loss.backward()
            optimizer.step()
            batches_completed += 1

            epoch_loss += batch_loss.item()
            epoch_rate += batch_rate.item() / len(batch)

            # ----- evaluate -----
            if (batch_idx + 1) % EVAL_EVERY == 0:
                test_loss = evaluate(GNNs_gamma, GNNs_u, test_loader, parameters, I_max)
                scheduler.step(test_loss)
                current_lr = optimizer.param_groups[0]['lr']
                print(f"[Epoch {epoch}] [Batch {batch_idx}] | Batch Loss: {batch_loss.item():.4f}, Avg Rate: {batch_rate.item()/len(batch):.4f} | LR: {current_lr:.2e} | Test Loss: {test_loss:.4f}")
                save_checkpoint(epoch, GNNs_gamma, GNNs_u, optimizer, ckpt_path)
            else:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"[Epoch {epoch}] [Batch {batch_idx}] | Batch Loss: {batch_loss.item():.4f}, Avg Rate: {batch_rate.item()/len(batch):.4f} | LR: {current_lr:.2e}")

        print(
            f" =================== [Epoch {epoch}] "
            f"Loss: {epoch_loss / batches_completed:.4f}, "
            f"Avg Rate: {epoch_rate / batches_completed:.4f}"
        )
        save_checkpoint(epoch, GNNs_gamma, GNNs_u, optimizer, ckpt_path)
    return str(ckpt_path)
