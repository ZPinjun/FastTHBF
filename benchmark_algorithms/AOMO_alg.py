# ============================================================
# AO–MO for DMA-based multi-user MIMO downlink beamforming
# Reference:
# S. F. Kimaryo and K. Lee,
# "Downlink Beamforming for Dynamic Metasurface Antennas",
# IEEE TWC, vol. 22, no. 7, pp. 4745–4755, 2023.
# ============================================================

import numpy as np
from numpy.linalg import inv
from scipy.linalg import block_diag
from pymanopt import Problem
from pymanopt.manifolds import ComplexCircle
from pymanopt.optimizers import TrustRegions
import autograd.numpy as anp
from pymanopt.function import autograd as pymanopt_ag

class AOMOBeamformer:
    """
    AOMO-based iterative beamforming solver for Problem (P1)
    """

    def __init__(
        self,
        H,              # list of H_k, Mk x (Nc*Nt)
        Ns_k,           # list of streams per user
        sigma2,         # list of noise powers
        beta,           # WSR weights
        eta,            # DMA waveguide responses
        P,              # per-DMA power constraint
        max_iter=20
    ):
        self.H = H
        self.Ns_k = Ns_k
        self.sigma2 = sigma2
        self.beta = beta
        self.eta = eta
        self.P = P
        self.max_iter = max_iter

        self.K = len(H)
        self.Nt = len(P)
        self.Nc = len(eta[0])
        self.Ns = sum(Ns_k)
        self.Mk = [H[k].shape[0] for k in range(self.K)]

        # ===== Initialization =====
        self.FD = (np.random.randn(self.Nt, self.Ns) +
                   1j * np.random.randn(self.Nt, self.Ns)) / np.sqrt(2)

        self.u = [
            np.exp(1j * 2 * np.pi * np.random.rand(self.Nc))
            for _ in range(self.Nt)
        ]

    def build_FA(self):
        blocks = []
        for n in range(self.Nt):
            Gn = np.diag(self.eta[n])
            wn = 0.5 * (Gn @ self.u[n] + 1j * self.eta[n])
            blocks.append(wn.reshape(-1, 1))
        return block_diag(*blocks)   # (Nc*Nt) x Nt
    
    def compute_WSR(self):
        """
        Compute weighted sum-rate using current (self.FD, self.u)
        """
        FA = self.build_FA()
        R = 0.0

        for k in range(self.K):
            Fk = self.FD[:, sum(self.Ns_k[:k]):sum(self.Ns_k[:k+1])]

            Sk = (
                self.H[k] @ FA @ Fk @
                Fk.conj().T @ FA.conj().T @ self.H[k].conj().T
            )

            Ik = np.zeros((self.Mk[k], self.Mk[k]), dtype=complex)
            for i in range(self.K):
                if i == k:
                    continue
                Fi = self.FD[:, sum(self.Ns_k[:i]):sum(self.Ns_k[:i+1])]
                Ik += (
                    self.H[k] @ FA @ Fi @
                    Fi.conj().T @ FA.conj().T @ self.H[k].conj().T
                )

            Ik += self.sigma2[k] * np.eye(self.Mk[k])

            Rk = np.real(
                np.log2(np.linalg.det(np.eye(self.Mk[k]) + Sk @ np.linalg.inv(Ik)))
            )
            R += self.beta[k] * Rk

        return R
    
    def update_Gamma_Omega(self, FA):
        Gamma, Omega = [], []

        for k in range(self.K):
            Sk = np.zeros((self.Mk[k], self.Mk[k]), dtype=complex)
            for i in range(self.K):
                Fi = self.FD[:, sum(self.Ns_k[:i]):sum(self.Ns_k[:i+1])]
                Sk += (
                    self.H[k] @ FA @ Fi @
                    Fi.conj().T @ FA.conj().T @ self.H[k].conj().T
                )
            Sk += self.sigma2[k] * np.eye(self.Mk[k])

            Fk = self.FD[:, sum(self.Ns_k[:k]):sum(self.Ns_k[:k+1])]
            Gk = inv(Sk) @ self.H[k] @ FA @ Fk
            Ek = np.eye(self.Ns_k[k]) - Gk.conj().T @ self.H[k] @ FA @ Fk
            Ok = inv(Ek)

            Gamma.append(Gk)
            Omega.append(Ok)

        return Gamma, Omega
    
    def update_FD(self, FA, Gamma, Omega,
              max_dual_iter=20,
              tol=1e-6,
              lambda_max=1e6):
        """
        Strict KKT-based update of F_D:
            min Tr(Fᴴ A F) - 2 Re Tr(Fᴴ B)
            s.t. ||F[n,:]||² ≤ P[n]
        """

        Nt = self.Nt
        K = self.K

        # ---------- build A ----------
        A = np.zeros((Nt, Nt), dtype=np.complex128)
        for k in range(K):
            Hk = self.H[k]
            Gk = Gamma[k]
            Ok = Omega[k]

            A += self.beta[k] * (
                FA.conj().T @ Hk.conj().T @ Gk @ Ok @
                Gk.conj().T @ Hk @ FA
            )

        # numerical stability
        A += 1e-9 * np.eye(Nt)

        # ---------- build B ----------
        B_blocks = []
        for k in range(K):
            Hk = self.H[k]
            Gk = Gamma[k]
            Ok = Omega[k]

            Bk = self.beta[k] * (
                FA.conj().T @ Hk.conj().T @ Gk @ Ok
            )
            B_blocks.append(Bk)

        B = np.hstack(B_blocks)   # Nt x Ns

        # ---------- dual variables ----------
        lam = np.zeros(Nt)

        # ---------- dual iteration ----------
        for _ in range(max_dual_iter):

            # solve (A + Λ) F = B
            M = A + np.diag(lam)
            FD = np.linalg.solve(M, B)

            power = np.sum(np.abs(FD)**2, axis=1)

            # check KKT conditions
            violation = power - self.P

            if np.max(violation) <= tol:
                break

            # update λ_n (projected gradient / bisection-like)
            for n in range(Nt):
                if violation[n] > 0:
                    lam[n] = min(lam[n] * 2 + 1e-6, lambda_max)
                else:
                    lam[n] = lam[n] * 0.5

        # final solution
        self.FD = FD

    def update_FA(self, Gamma, Omega,
                step_init=0.1,
                inner_iter=20,
                backtrack_beta=0.5):

        Nt, Nc = self.Nt, self.Nc
        H = self.H
        FD = self.FD
        eta = self.eta
        beta = self.beta
        Ns_k = self.Ns_k
        K = self.K

        NcNt = Nc * Nt

        # ===== build A and B =====
        A = np.zeros((NcNt, NcNt), dtype=complex)
        B = np.zeros((NcNt, self.Ns), dtype=complex)

        col = 0
        for k in range(K):

            Hk = H[k]
            Gk = Gamma[k]
            Ok = Omega[k]

            Ak = Hk.conj().T @ Gk @ Ok @ Gk.conj().T @ Hk
            Bk = Hk.conj().T @ Gk @ Ok

            A += beta[k] * Ak
            B[:, col:col+Ns_k[k]] = beta[k] * Bk

            col += Ns_k[k]

        FDFDH = FD @ FD.conj().T

        for _ in range(inner_iter):

            FA = self.build_FA()

            # gradient wrt FA*
            Grad_FA = A @ FA @ FDFDH - B @ FD.conj().T

            # ===== convert to gradient wrt u =====
            grad_u_all = []

            for n in range(Nt):
                rows = slice(n*Nc, (n+1)*Nc)
                grad_w = Grad_FA[rows, n]
                grad_u = 0.5 * np.conj(eta[n]) * grad_w
                grad_u_all.append(grad_u)

            g = np.hstack(grad_u_all)
            u_vec = np.hstack(self.u)

            # ===== Riemannian projection =====
            g_riem = g - np.real(g * np.conj(u_vec)) * u_vec

            step = step_init
            old_val = self.compute_WSR()

            # ===== backtracking line search =====
            for _ in range(10):

                u_new = u_vec - step * g_riem
                u_new = np.exp(1j * np.angle(u_new))

                # temporarily assign
                for n in range(Nt):
                    self.u[n] = u_new[n*Nc:(n+1)*Nc]

                new_val = self.compute_WSR()

                if new_val >= old_val:
                    break

                step *= backtrack_beta

            # commit
            for n in range(Nt):
                self.u[n] = u_new[n*Nc:(n+1)*Nc]

    def solve(self, tol=1e-6, verbose=True):
        eps = 1e-12
        prev_wsr = -np.inf

        for it in range(self.max_iter):
            # ===== AO iteration =====
            FA = self.build_FA()
            Gamma, Omega = self.update_Gamma_Omega(FA)
            self.update_FD(FA, Gamma, Omega)
            self.update_FA(Gamma, Omega)

            # ===== compute WSR =====
            curr_wsr = self.compute_WSR()

            if verbose:
                print(
                    f"[AOMO] Iter {it+1:3d}/{self.max_iter}, "
                    f"WSR = {curr_wsr:.6f} bit/s/Hz"
                )

            # ===== relative improvement stopping criterion =====
            if it > 0:
                rel_impr = abs(curr_wsr - prev_wsr) / max(abs(prev_wsr), eps)

                if rel_impr < tol:
                    if verbose:
                        print(
                            f"[AOMO] Converged at iter {it+1}, "
                            f"relative WSR improvement = {rel_impr:.2e} < {tol:.1e}"
                        )
                    break

            prev_wsr = curr_wsr

        return self.build_FA(), self.FD