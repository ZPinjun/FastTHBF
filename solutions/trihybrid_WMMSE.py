import torch
import torch.nn as nn
import torch.nn.functional as F

class TriHybridWMMSE:
    def __init__(
        self,
        H_list,
        Ns_k,
        Mk,
        sigma2,
        beta,
        P,
        eta,
        u_init,
        v_init,
        seed=42
    ):
        torch.manual_seed(seed)
        self.cdtype = torch.cfloat

        # System
        self.H = H_list
        self.K = len(H_list)
        self.Ns_k = Ns_k
        self.Mk = Mk
        self.Ns = sum(Ns_k)
        self.sigma2 = sigma2
        self.beta = beta
        self.P = P
        self.eta = eta

        # DMA dimensions
        self.Nt = len(eta)
        self.Nc = eta[0].shape[0]

        # Variables
        self.u = u_init
        self.v = v_init
        self.w = [
            self.eta[n] * (0.5 * (1j + self.u[n]))
            for n in range(self.Nt)
        ]

        self.F_A = self.build_F_A()
        self.F_D = self.build_F_D()

    # =====================================================
    # Basic builders
    # =====================================================
    def build_F_A(self):
        return torch.block_diag(*self.w)

    def build_F_D(self):
        return torch.vstack([self.v[n].conj().transpose(0, 1) for n in range(self.Nt)])

    def clone_uv(self):
        """Return detached copies of the current antenna-domain and feed-domain variables."""
        return (
            [un.detach().clone() for un in self.u],
            [vn.detach().clone() for vn in self.v],
        )

    def set_uv(self, u, v):
        """Reset the solver state from a candidate initialization."""
        self.u = [un.detach().clone().to(self.cdtype) for un in u]
        self.v = [vn.detach().clone().to(self.cdtype) for vn in v]
        self.w = [
            self.eta[n] * (0.5 * (1j + self.u[n]))
            for n in range(self.Nt)
        ]
        self.F_A = self.build_F_A()
        self.F_D = self.build_F_D()

    def _power_scalar(self, n):
        p = self.P[n]
        if torch.is_tensor(p):
            return float(p.detach().cpu())
        return float(p)

    def random_uv(self, seed=None):
        """Generate one random feasible initialization for the nonconvex WMMSE iterations."""
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(int(seed))

        u = []
        v = []
        for n in range(self.Nt):
            phase = 2 * torch.pi * torch.rand((self.Nc, 1), generator=generator)
            u.append(torch.exp(1j * phase).to(self.cdtype))

            real = torch.randn((self.Ns, 1), generator=generator)
            imag = torch.randn((self.Ns, 1), generator=generator)
            vn = (real + 1j * imag).to(self.cdtype)
            norm_vn = torch.linalg.norm(vn)
            if norm_vn > 0:
                vn = vn / norm_vn * torch.sqrt(torch.tensor(self._power_scalar(n)))
            v.append(vn)

        return u, v

    # =====================================================
    # Rate computation
    # =====================================================
    def compute_weighted_sum_rate(self):
        R_list = []
        for k in range(self.K):
            Hk = self.H[k]
            Mk = Hk.shape[0]

            idx0 = sum(self.Ns_k[:k])
            idx1 = idx0 + self.Ns_k[k]
            F_Dk = self.F_D[:, idx0:idx1]

            Sk = Hk @ self.F_A @ F_Dk @ F_Dk.conj().transpose(0, 1) @ self.F_A.conj().transpose(0, 1) @ Hk.conj().transpose(0, 1)

            Ik = self.sigma2[k] * torch.eye(Mk, dtype=self.cdtype)
            for i in range(self.K):
                if i == k:
                    continue
                id0 = sum(self.Ns_k[:i])
                id1 = id0 + self.Ns_k[i]
                F_Di = self.F_D[:, id0:id1]
                Ik += Hk @ self.F_A @ F_Di @ F_Di.conj().transpose(0, 1) @ self.F_A.conj().transpose(0, 1) @ Hk.conj().transpose(0, 1)

            eigvals = torch.linalg.eigvals(torch.eye(Mk) + torch.linalg.solve(Ik, Sk))
            R_list.append(torch.real(torch.sum(torch.log2(eigvals))))

        R_sum = sum(self.beta[k] * R_list[k] for k in range(self.K))
        return R_sum, R_list

    # =====================================================
    # WMMSE steps
    # =====================================================
    def update_Gamma(self):
        Gamma = []
        for k in range(self.K):
            Hk = self.H[k]
            Ryy = self.sigma2[k] * torch.eye(self.Mk[k], dtype=self.cdtype)

            for i in range(self.K):
                id0 = sum(self.Ns_k[:i])
                id1 = id0 + self.Ns_k[i]
                FDi = self.F_D[:, id0:id1]
                Ryy += Hk @ self.F_A @ FDi @ FDi.conj().transpose(0, 1) @ self.F_A.conj().transpose(0, 1) @ Hk.conj().transpose(0, 1)

            id0 = sum(self.Ns_k[:k])
            id1 = id0 + self.Ns_k[k]
            FDk = self.F_D[:, id0:id1]

            Gamma_k = torch.linalg.solve(Ryy, Hk @ self.F_A @ FDk)
            Gamma.append(Gamma_k)
        return Gamma

    def update_Omega(self, Gamma, dgamma=None):
        Omega = []
        for k in range(self.K):
            id0 = sum(self.Ns_k[:k])
            id1 = id0 + self.Ns_k[k]
            FDk = self.F_D[:, id0:id1]
            Ek = torch.eye(self.Ns_k[k]) - Gamma[k].conj().transpose(0, 1) @ self.H[k] @ self.F_A @ FDk
            Omega_k = torch.linalg.inv(Ek)
            l = Omega_k.shape[0]
            if dgamma is not None:
                M_a_real = dgamma[k, :l**2].reshape(l,l)
                M_a_imag = dgamma[k, l**2:2*l**2].reshape(l,l)
                M_a = M_a_real + 1j * M_a_imag
                Omega_k =  Omega_k  + M_a + M_a.conj().transpose(0, 1)
            Omega.append(Omega_k)
        return Omega

    def compute_B(self, Gamma, Omega):
        B = [[None] * self.Nt for _ in range(self.Nt)]
        Ak_all = []
        for k in range(self.K):
            Ak = Gamma[k] @ Omega[k] @ Gamma[k].conj().transpose(0, 1)
            Ak_all.append(Ak)
        for q in range(self.Nt):
            for p in range(self.Nt):
                Bqp = torch.zeros((self.Nc, self.Nc), dtype=self.cdtype)
                for k in range(self.K):
                    Hkq = self.H[k][:, q*self.Nc:(q+1)*self.Nc]
                    Hkp = self.H[k][:, p*self.Nc:(p+1)*self.Nc]
                    Bqp += self.beta[k] * Hkq.conj().transpose(0, 1) @ Ak_all[k] @ Hkp
                B[q][p] = Bqp
        return B

    # =====================================================
    # Update v_n
    # =====================================================
    def update_v_n(self, n, Bnn, Qn, Dn):
        num = (Dn.conj().transpose(0, 1) - Qn.conj().transpose(0, 1)) @ self.w[n]
        denom = torch.real(self.w[n].conj().transpose(0, 1) @ Bnn @ self.w[n]).item()

        if denom > 1e-9:
            v = num / denom
        else:
            v = num / (denom + 1e-9)

        if torch.linalg.norm(v) > torch.sqrt(torch.tensor(self.P[n])):
            v = v / torch.linalg.norm(v) * torch.sqrt(torch.tensor(self.P[n]))

        self.v[n] = v
        self.F_D = self.build_F_D()

    # =====================================================
    # Closed-form update for u_n
    # =====================================================
    def update_u_n(self, n, Bnn, Qn, Dn, du=None):
        etav = self.eta[n][:, 0]
        Am = torch.diag(etav.conj()) @ (torch.linalg.norm(self.v[n])**2 * Bnn) @ torch.diag(etav)
        bv = torch.diag(etav.conj()) @ ((Qn - Dn) @ self.v[n])

        eigvals = torch.linalg.eigvals(Am)
        max_eigval = torch.max(torch.abs(eigvals))
        if du is not None:
            max_eigval = max_eigval + du

        un = self.u[n][:, 0:1]
        vec_phase = (Am - max_eigval * torch.eye(self.Nc)) @ un \
                    + 2*bv + Am @ (1j*torch.ones((self.Nc,1)))

        u_opt = torch.exp(1j * torch.angle(-vec_phase))

        self.u[n] = u_opt
        self.w[n] = ((1j + u_opt[:, 0]) * etav / 2).reshape(-1, 1)
        self.F_A = self.build_F_A()

    # =====================================================
    # Iteration loop (classical WMMSE)
    # =====================================================
    def run_classical(self, I_max=50, verbose=False, tol = None):
        history = []

        R, _ = self.compute_weighted_sum_rate()
        history.append(R)
        if verbose:
            print(f"Initial sum-rate: {R:.4f}")

        for it in range(I_max):
            Gamma = self.update_Gamma()
            Omega = self.update_Omega(Gamma)
            B = self.compute_B(Gamma, Omega)
            for n in range(self.Nt):
                Qn = sum(B[n][m] @ self.w[m] @ self.v[m].conj().transpose(0, 1)
                         for m in range(self.Nt) if m != n)
                Dn = torch.zeros((self.Nc, self.Ns), dtype=self.cdtype)
                col = 0
                for k in range(self.K):
                    Hkn = self.H[k][:, n*self.Nc:(n+1)*self.Nc]
                    Dn[:, col:col+self.Ns_k[k]] = self.beta[k] * Hkn.conj().transpose(0, 1) @ Gamma[k] @ Omega[k]
                    col += self.Ns_k[k]
                self.update_v_n(n, B[n][n], Qn, Dn)
                #R, _ = self.compute_weighted_sum_rate()
                #if verbose:
                #    print(f"Update v_n, sum-rate: {R:.4f}")
                self.update_u_n(n, B[n][n], Qn, Dn)
                #R, _ = self.compute_weighted_sum_rate()
                #if verbose:
                #    print(f"Update u_n, sum-rate: {R:.4f}")
                
            R_prev = R
            R, _ = self.compute_weighted_sum_rate()
            history.append(R)

            # Check convergence
            rel_error = abs(R - R_prev) / (abs(R_prev) + 1e-10)
            if verbose:
                print(f"Iteration {it+1}/{I_max}, Sum-rate: {R:.4f}, Relative error: {rel_error:.6e}")
            
            if tol is not None:
                if rel_error < tol:
                    if verbose:
                        print(f"\nConverged at iteration {it+1} with relative error {rel_error:.6e}")
                    break

        return history, self.F_A, self.F_D, self.u, self.v

    def run_classical_multistart(
        self,
        I_max=50,
        verbose=False,
        tol=None,
        init_candidates=None,
        num_random_starts=0,
        random_seed=0,
    ):
        """
        Run classical WMMSE from multiple feasible initializations and keep the best result.

        With no additional candidates, the solver's current state is the sole initialization.
        """
        candidates = []
        if init_candidates is None:
            u0, v0 = self.clone_uv()
            candidates.append({"name": "current", "u": u0, "v": v0})
        else:
            for idx, candidate in enumerate(init_candidates):
                if isinstance(candidate, dict):
                    candidates.append({
                        "name": candidate.get("name", f"candidate_{idx}"),
                        "u": candidate["u"],
                        "v": candidate["v"],
                    })
                else:
                    u, v = candidate
                    candidates.append({"name": f"candidate_{idx}", "u": u, "v": v})

        for r in range(num_random_starts):
            u_rand, v_rand = self.random_uv(seed=random_seed + r)
            candidates.append({"name": f"random_{r}", "u": u_rand, "v": v_rand})

        best = None
        candidate_results = []
        for idx, candidate in enumerate(candidates):
            self.set_uv(candidate["u"], candidate["v"])
            history, _, _, _, _ = self.run_classical(
                I_max=I_max,
                verbose=verbose,
                tol=tol,
            )
            final_rate = torch.real(history[-1]).item()
            result = {
                "index": idx,
                "name": candidate["name"],
                "final_rate": final_rate,
                "num_iters": len(history) - 1,
            }
            candidate_results.append(result)

            if best is None or final_rate > best["final_rate"]:
                best_u, best_v = self.clone_uv()
                best = {
                    "index": idx,
                    "name": candidate["name"],
                    "final_rate": final_rate,
                    "history": [h.detach().clone() if torch.is_tensor(h) else h for h in history],
                    "u": best_u,
                    "v": best_v,
                }

        self.set_uv(best["u"], best["v"])
        info = {
            "best_index": best["index"],
            "best_name": best["name"],
            "best_rate": best["final_rate"],
            "candidate_results": candidate_results,
        }

        return best["history"], self.F_A, self.F_D, self.u, self.v, info
    
    # =====================================================
    # Iteration loop (unfolded WMMSE)
    # =====================================================
    def run_unfolded(self, dgamma_list, du_list, I_max=10, verbose=True):
        history = []

        R, _ = self.compute_weighted_sum_rate()
        history.append(R)
        if verbose:
            print(f"Initial sum-rate: {R:.4f}")

        for it in range(I_max):
            Gamma = self.update_Gamma()
            Omega = self.update_Omega(Gamma, dgamma_list[it])
            B = self.compute_B(Gamma, Omega)
            for n in range(self.Nt):
                Qn = sum(B[n][m] @ self.w[m] @ self.v[m].conj().transpose(0, 1)
                         for m in range(self.Nt) if m != n)
                Dn = torch.zeros((self.Nc, self.Ns), dtype=self.cdtype)
                col = 0
                for k in range(self.K):
                    Hkn = self.H[k][:, n*self.Nc:(n+1)*self.Nc]
                    Dn[:, col:col+self.Ns_k[k]] = self.beta[k] * Hkn.conj().transpose(0, 1) @ Gamma[k] @ Omega[k]
                    col += self.Ns_k[k]
                self.update_v_n(n, B[n][n], Qn, Dn)
                self.update_u_n(n, B[n][n], Qn, Dn, du_list[it][n])

            R_prev = R
            R, _ = self.compute_weighted_sum_rate()
            history.append(R)
            
            # Check convergence
            rel_error = abs(R - R_prev) / (abs(R_prev) + 1e-10)
            if verbose:
                print(f"Iteration {it+1}/{I_max}, Sum-rate: {R:.4f}, Relative error: {rel_error:.6e}")

        return history, self.F_A, self.F_D, self.u, self.v


class GraphSAGELayer(nn.Module):
    """
    Mean-aggregator GraphSAGE layer
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # since concat(self, neighbor), the input dimension is 2 * in_dim
        self.W = nn.Linear(2 * in_dim, out_dim, bias=False)

    def forward(self, x, edge_index):
        """
        x:          (N, in_dim)
        edge_index: (2, E)  -- undirected edges (i, j) only ONCE
        """
        N = x.size(0)
        device = x.device

        # -------------------------------------------------
        # 1. make edges bidirectional
        # -------------------------------------------------
        src, dst = edge_index

        src_rev = dst
        dst_rev = src

        src = torch.cat([src, src_rev], dim=0)
        dst = torch.cat([dst, dst_rev], dim=0)

        # -------------------------------------------------
        # 2. aggregate neighbor features (mean)
        # -------------------------------------------------
        neigh_sum = torch.zeros(N, x.size(1), device=device)
        neigh_sum.index_add_(0, dst, x[src])

        deg = torch.zeros(N, device=device)
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float))

        deg = deg.clamp(min=1).unsqueeze(1)

        neigh_mean = neigh_sum / deg

        # -------------------------------------------------
        # 3. concat self + neighbor
        # -------------------------------------------------
        h = torch.cat([x, neigh_mean], dim=1)

        # -------------------------------------------------
        # 4. linear transform
        # -------------------------------------------------
        out = self.W(h)

        return out


class GNN(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        num_layers=2,
        dp_rate=0.1
    ):
        super().__init__()
        assert num_layers >= 1

        self.num_layers = num_layers
        self.dp_rate = dp_rate

        # preprocess
        self.preprocessor = nn.Sequential(
            nn.Linear(in_dim, int((in_dim+hidden_dim)/2)),
            nn.ReLU(),
            nn.Linear(int((in_dim+hidden_dim)/2), hidden_dim)
        )

        # ----------- GraphSAGE -----------
        self.GCN_layers = nn.ModuleList()
        self.GCN_layers.append(GraphSAGELayer(hidden_dim, hidden_dim))

        for _ in range(num_layers - 2):
            self.GCN_layers.append(GraphSAGELayer(hidden_dim, hidden_dim))

        if num_layers > 1:
            self.GCN_layers.append(GraphSAGELayer(hidden_dim, hidden_dim))
        # ---------------------------------------------

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, int((hidden_dim + out_dim)/2)),
            nn.ReLU(),
            nn.Linear(int((hidden_dim + out_dim)/2), out_dim)
        )

    def global_mean_pool(self, x, graph_id, num_graphs):
        out = torch.zeros(num_graphs, x.size(1), device=x.device)
        out.index_add_(0, graph_id, x)

        count = torch.bincount(graph_id, minlength=num_graphs).float().unsqueeze(1)
        return out / count

    def forward(self, x, edge_index, graph_id=None, num_graphs=None):

        x = self.preprocessor(x)

        for i, layer in enumerate(self.GCN_layers):
            x = layer(x, edge_index)

            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dp_rate, training=self.training)

        if graph_id is not None and num_graphs is not None:
            x = self.global_mean_pool(x, graph_id, num_graphs)

        return self.readout(x)
