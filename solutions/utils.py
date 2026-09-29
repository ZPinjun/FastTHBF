import torch
import math
from torch.utils.data import Dataset
from solutions.trihybrid_WMMSE import TriHybridWMMSE

class channelDataset(Dataset):
    def __init__(self, ProbInsts, parameters):
        self.ProbInsts = ProbInsts
        self.parameters = parameters

    def __len__(self):
        return len(self.ProbInsts)

    def channel_list_to_tensor(self, H_list, sigma):
        """
        Convert H_list {(Mk,NcNt),(Mk,NcNt),...,(Mk,NcNt)} -> [K, 2Mk*NcNt] tensor
        Node features: concatenated Re(H_k) and Im(H_k)
        """
        # Normalize across all channels
        node_feats = []
        for Hk in H_list:
            Hk_real_flattened = Hk.real.flatten()  # (Mk*NcNt,)
            Hk_imag_flattened = Hk.imag.flatten()  # (Mk*NcNt,)
            Hk_flattened = torch.cat([Hk_real_flattened, Hk_imag_flattened], dim=0)  # (2*Mk*NcNt,)
            node_feat = torch.cat([torch.tensor([sigma], dtype=Hk_flattened.dtype), Hk_flattened], dim=0)       # (2*Mk*NcNt+1,) 
            node_feats.append(node_feat) 
        node_feats_tensor = torch.stack(node_feats, dim=0)      # (K, 2*Mk*NcNt+1)
        scale = torch.sqrt(torch.tensor(node_feats_tensor.numel(), dtype=node_feats_tensor.dtype))
        node_feats_normalized = node_feats_tensor / torch.linalg.norm(node_feats_tensor, ord='fro') * scale     # (K, 2*Mk*NcNt+1)

        return node_feats_normalized

    def __getitem__(self, idx):
        Inst = self.ProbInsts[idx]

        # Node features
        Hs = Inst["H"]    # list of (Mk, NcNt)
        sigma = torch.sqrt(self.parameters["sigma2"])
        h_NodeFeats = self.channel_list_to_tensor(Hs,sigma)       # (K, 2*Mk*NcNt+1)

        # One edge per unordered user pair; GraphSAGELayer adds the reverse edges.
        K = Inst["K"]
        edge_index = []
        for i in range(K):
            for j in range(i+1, K):
                edge_index.append([i, j])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # (2, E)
        
        return {"H_list": Hs,
                "K": K,
                "node_feature": h_NodeFeats.float(),
                "edge_index": edge_index}
    

def evaluate(GCNs_gamma_list, GCNs_u_list, loader, parameters, I_max):
    """Calculate the average loss on a testing dataset"""

    cdtype = torch.cfloat
    for gcn in GCNs_gamma_list:
        gcn.eval()
    for gcn in GCNs_u_list:
        gcn.eval()
        
    total_loss = 0.0
    num_samples = 0

    Nt = parameters["Nt"]
    Nc = parameters["Nc"]
    P = parameters["P"]
    Mk = parameters["Mk"]
    eta = parameters["eta"]
    Ns_k = parameters["Ns_k"]
    sigma2 = parameters["sigma2"]

    with torch.no_grad():
        for batch in loader:
            for pro_inst in batch:
                K = pro_inst["K"]
                node_features = pro_inst["node_feature"]
                edge_index = pro_inst["edge_index"]

                # GCN output
                delta_gamma_list = []
                delta_u_list = []
                graph_id = torch.zeros(K, dtype=torch.int32)
                num_graphs = torch.tensor(1, dtype=torch.int32)
                for i in range(I_max):
                    # delta_gamma
                    delta_gamma = GCNs_gamma_list[i](node_features, edge_index)
                    delta_gamma_list.append(delta_gamma)
                    # delta_u
                    delta_u = GCNs_u_list[i](node_features, edge_index, graph_id, num_graphs)
                    delta_u_mt = delta_u.reshape(Nt, 1)
                    delta_u_list.append(delta_u_mt)

                # WMMSE
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
                v_init=v0
                )
                rate_trace, *_ = solver.run_unfolded(dgamma_list=delta_gamma_list, du_list=delta_u_list, I_max=I_max, verbose=False)
                rate_final = rate_trace[-1]
                loss = -rate_final

                total_loss += loss.item()
                num_samples += 1

    for gcn in GCNs_gamma_list:
        gcn.train()
    for gcn in GCNs_u_list:
        gcn.train()

    return total_loss / num_samples


def init_xavier(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


def init_kaiming(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

def zero_last_linear(model):
    last_linear = None
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            last_linear = m
    if last_linear is not None:
        torch.nn.init.zeros_(last_linear.weight)
        if last_linear.bias is not None:
            torch.nn.init.zeros_(last_linear.bias)


class HybridPrecodingBFGS:
    """
    Hybrid precoding design via BFGS optimization (Algorithm 1 in [1])
    Reference:
    [1] J. Jin, Y. R. Zheng, W. Chen and C. Xiao, "Hybrid Precoding for Millimeter Wave MIMO Systems: A Matrix Factorization Approach," in IEEE Transactions on Wireless Communications, vol. 17, no. 5, pp. 3327-3339, May 2018, doi: 10.1109/TWC.2018.2810072. 
    """
    def __init__(self, Fopt, Nrf):
        self.Fopt = Fopt
        self.Nt, self.Ns = Fopt.shape
        self.Nrf = Nrf

        # ---------- Initialization from Eq. (48) ----------
        U, _, _ = torch.linalg.svd(Fopt, full_matrices=True)
        UF = U[:, :Nrf]
        phase = torch.angle(UF)

        Phi0 = phase[1:, :] - phase[0:1, :]
        self.Phi = Phi0.clone()

    # -------- Build FRF from Phi Eq. (33) --------
    def build_FRF(self, Phi):
        Phi_RF = torch.zeros((self.Nt, self.Nrf))
        Phi_RF[1:, :] = Phi
        FRF = torch.exp(1j * Phi_RF) / math.sqrt(self.Nt)
        return FRF

    # -------- Objective φ(Φ) using Eq. (53) --------
    def loss(self, Phi):
        FRF = self.build_FRF(Phi)
        Q, _ = torch.linalg.qr(FRF)
        return torch.norm(self.Fopt, 'fro')**2 - torch.norm(Q.conj().T @ self.Fopt, 'fro')**2

    # -------- Gradient using Eq. (55) --------
    def grad(self, Phi):
        Nt, Nrf = self.Nt, self.Nrf
        FRF = self.build_FRF(Phi)

        # QR: FRF = Q R   (Eq. 52)
        Q, R = torch.linalg.qr(FRF)

        Z = Q.conj().T @ self.Fopt                 # Z_RF
        RinvH = torch.linalg.inv(R).conj().T       # (R^{-1})^H

        # core term: (QZ - Fopt) Z^H (R^{-1})^H
        A = (Q @ Z - self.Fopt) @ Z.conj().T @ RinvH

        # Hadamard with FRF*
        G = A * FRF.conj()

        # ∇ψ = 2 Im(G)
        grad_full = 2 * torch.imag(G)

        # eq.(54): remove first row
        return grad_full[1:, :]
    
    # -------- build B0 Eq. (49-51) --------
    def build_B0(self, Phi, eps=1e-4, delta_min=1e-4):
        """
        Approximate Hessian at Phi using finite difference,
        then construct B0 according to Eq. (49)-(51)
        """
        d = Phi.numel()
        g0 = self.grad(Phi).reshape(-1)

        H = torch.zeros((d, d))

        # finite difference Hessian
        for i in range(d):
            e = torch.zeros(d)
            e[i] = eps
            Phi_eps = Phi.reshape(-1) + e
            Phi_eps = Phi_eps.reshape_as(Phi)

            gi = self.grad(Phi_eps).reshape(-1)
            H[:, i] = (gi - g0) / eps

        # symmetrize
        H = 0.5 * (H + H.T)

        # eigendecomposition (eq.49)
        eigvals, U = torch.linalg.eigh(H)

        # build Σ_hat (eq.51)
        eigvals_clipped = torch.clamp(torch.abs(eigvals), min=delta_min)

        # construct B0 = U Σ^{-1} U^T (eq.50)
        B0 = U @ torch.diag(1.0 / eigvals_clipped) @ U.T

        return B0

    # -------- Full BFGS (Algorithm 1) --------
    def solve(self, max_iter=100, tol=1e-6, use_B0=True, verbose=False):
        dim = self.Phi.numel()
        Phi = self.Phi.clone()

        if use_B0:
            if verbose:
                print("Building B0 (may take a few seconds)...")
            B = self.build_B0(Phi)
        else:
            B = torch.eye(dim)

        rho_prev = 1.0

        for it in range(max_iter):
            g = self.grad(Phi).reshape(-1)
            grad_norm = torch.norm(g)

            loss = self.loss(Phi).item()
            if verbose:
                print(f"Iter {it:3d} | loss={loss:.6e} | grad_norm={grad_norm:.3e}")

            if grad_norm < tol:
                break

            # descent direction
            p = -(B @ g).reshape_as(Phi)

            # backtracking line search
            rho = rho_prev
            f0 = self.loss(Phi)
            while True:
                Phi_new = Phi + rho * p
                f1 = self.loss(Phi_new)
                if f1 <= f0 + 1e-4 * rho * (g @ p.reshape(-1)):
                    break
                rho *= 0.5
                if rho < 1e-8:
                    break

            s = (Phi_new - Phi).reshape(-1)
            y = (self.grad(Phi_new) - self.grad(Phi)).reshape(-1)

            if y @ s > 1e-8:
                Bs = B @ s
                B = B + torch.outer(s, s)/(y @ s) - torch.outer(Bs, Bs)/(s @ Bs)

            Phi = Phi_new
            rho_prev = rho

        FRF = self.build_FRF(Phi)
        FBB = torch.linalg.pinv(FRF) @ self.Fopt
        return FRF, FBB
    

def computeWSR(FA, FRF, FBB, H_list, parameters):
    K = len(H_list)
    cdtype = torch.cfloat
    Nt = parameters["Nt"]
    Nc = parameters["Nc"]
    P = parameters["P"]
    Mk = parameters["Mk"]
    eta = parameters["eta"]
    Ns_k = parameters["Ns_k"]
    sigma2 = parameters["sigma2"]
    Ns = Ns_k * K
    u0 = [torch.ones(Nc, 1, dtype=cdtype)] * Nt
    v0 = [torch.ones(Ns, 1, dtype=cdtype)] * Nt
    for n in range(Nt):    # normalize v0
        norm_vn = torch.linalg.norm(v0[n])
        if norm_vn > torch.sqrt(P):
            v0[n] = v0[n] / norm_vn * torch.sqrt(P)

    solver = TriHybridWMMSE(
        H_list=H_list,
        Ns_k=[Ns_k] * K,
        Mk=[Mk] * K,
        sigma2=[sigma2] * K,
        beta=[1.0] * K,
        P=[P] * Nt,
        eta=eta,
        u_init=u0,
        v_init=v0
    )
    solver.F_A = FA
    solver.F_D = FRF @ FBB
    R_sum, R_list = solver.compute_weighted_sum_rate()

    return R_sum, R_list


def checkHybridConstraints(FRF, FBB, P_ant, tol=1e-6, verbose=True):
    """
    Check phase-shifting constraint for FRF and PAPC for FRF @ FBB.

    Args:
        FRF: complex tensor (Nt, Nrf)
        FBB: complex tensor (Nrf, Ns)
        P_ant: scalar or tensor of shape (Nt,)  per-antenna power constraint
        tol: numerical tolerance
        verbose: print diagnostics or not

    Returns:
        phase_ok: bool
        papc_ok: bool
        info: dict with diagnostic values
    """
    Nt = FRF.shape[0]

    # ---------- Phase shifting ----------
    FRF_abs = torch.abs(FRF) * torch.sqrt(torch.tensor(Nt))
    phase_error = torch.max(torch.abs(FRF_abs - 1.0))
    phase_ok = phase_error < tol

    # ---------- PAPC ----------
    FD = FRF @ FBB   # (Nt, Ns)
    power_per_ant = torch.sum(torch.abs(FD)**2, dim=1)  # (Nt,)

    if isinstance(P_ant, (float, int)):
        P_ant = torch.full((Nt,), P_ant)

    papc_ratio = power_per_ant / P_ant
    max_ratio = torch.max(papc_ratio)
    papc_ok = max_ratio <= 1

    # ---------- info ----------
    info = {
        "max_phase_error": phase_error.item(),
        "max_papc_ratio": max_ratio.item(),
        "power_per_antenna": power_per_ant.detach().cpu()
    }

    if verbose:
        print(f"[Phase constraint] OK={phase_ok}, max | |FRF|-1/sqrt(Nt) | = {phase_error:.6e}")
        print(f"[PAPC constraint ] OK={papc_ok}, max ratio = {max_ratio:.6e}")

    return phase_ok, papc_ok, info

