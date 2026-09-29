# ============================================================
# Matrix decomposition algorithm for DMA-based multi-user MIMO downlink beamforming
# Reference:
# H. Zhang, N. Shlezinger, F. Guidi, D. Dardari, M. F. Imani and Y. C. Eldar, 
# "Beam Focusing for Near-Field Multiuser MIMO Communications," 
# IEEE TWC, vol. 21, no. 9, pp. 7476-7490, 2022.
# ============================================================

import numpy as np
from numpy.linalg import solve
from scipy.linalg import block_diag


class DecompBeamformer:
    """
    Full tri-hybrid beamforming solver:
      Stage 1: Problem P1 (WMMSE)  -> Ftilde*
      Stage 2: Problem P2 (AO)     -> F_A, F_D
    Returns optimized (F_A, F_D).
    """

    def __init__(self, H, Ns_k, sigma2, beta, eta, P, max_iterP1=20, max_iterP2=20):
        # Basic parameters
        self.H = H
        self.Ns_k = Ns_k
        self.sigma2 = sigma2
        self.beta = beta
        self.eta = eta
        self.P = np.array(P)
        self.max_iterP1 = max_iterP1
        self.max_iterP2 = max_iterP2

        # Derived dimensions
        self.K = len(H)
        self.Nt = len(P)
        self.Nc = len(eta[0])
        self.Ns = sum(Ns_k)
        self.Mk = [H[k].shape[0] for k in range(self.K)]

        # ===== Random initialization =====
        self.FD = (
            np.random.randn(self.Nt, self.Ns)
            + 1j * np.random.randn(self.Nt, self.Ns)
        ) / np.sqrt(2)

        self.u = [
            np.exp(1j * 2 * np.pi * np.random.rand(self.Nc))
            for _ in range(self.Nt)
        ]

        self.Ftilde = (
            np.random.randn(self.Nc * self.Nt, self.Ns)
            + 1j * np.random.randn(self.Nc * self.Nt, self.Ns)
        ) / np.sqrt(2)

    # =====================================================
    # Build analog beamformer F_A (pure numpy)
    # =====================================================
    def build_FA(self):
        blocks = []
        for n in range(self.Nt):
            Gn = np.diag(self.eta[n])
            wn = 0.5 * (Gn @ self.u[n] + 1j * self.eta[n])
            blocks.append(wn.reshape(-1, 1))
        return block_diag(*blocks)

    # =====================================================
    # Compute WSR using current Ftilde (numpy only)
    # =====================================================
    def compute_WSR_Ftilde(self):
        total_rate = 0.0
        col = 0

        for k in range(self.K):
            Fk = self.Ftilde[:, col:col + self.Ns_k[k]]

            # Interference + noise
            Ik = self.sigma2[k] * np.eye(self.Mk[k], dtype=np.complex128)
            for i in range(self.K):
                if i == k:
                    continue
                Fi = self.Ftilde[:, sum(self.Ns_k[:i]):sum(self.Ns_k[:i + 1])]
                Ik += self.H[k] @ Fi @ Fi.conj().T @ self.H[k].conj().T

            Sk = self.H[k] @ Fk @ Fk.conj().T @ self.H[k].conj().T

            Rk = np.real(
                np.log2(
                    np.linalg.det(
                        np.eye(self.Mk[k]) + Sk @ np.linalg.inv(Ik)
                    )
                )
            )

            total_rate += self.beta[k] * Rk
            col += self.Ns_k[k]

        return total_rate

    # =====================================================
    # P1: WMMSE update Gamma and Omega (numpy only)
    # =====================================================
    def update_Gamma_Omega(self):
        Gamma, Omega = [], []
        col = 0

        for k in range(self.K):
            Sk = self.sigma2[k] * np.eye(self.Mk[k], dtype=np.complex128)
            for i in range(self.K):
                Fi = self.Ftilde[:, sum(self.Ns_k[:i]):sum(self.Ns_k[:i + 1])]
                Sk += self.H[k] @ Fi @ Fi.conj().T @ self.H[k].conj().T

            Fk = self.Ftilde[:, col:col + self.Ns_k[k]]
            Gk = solve(Sk, self.H[k] @ Fk)

            Ek = np.eye(self.Ns_k[k], dtype=np.complex128) \
                 - Gk.conj().T @ self.H[k] @ Fk
            Ok = solve(Ek, np.eye(self.Ns_k[k], dtype=np.complex128))

            Gamma.append(Gk)
            Omega.append(Ok)
            col += self.Ns_k[k]

        return Gamma, Omega

    # =====================================================
    # P1: WMMSE update Ftilde (numpy only)
    # =====================================================
    def update_Ftilde(self, Gamma, Omega, tol=1e-6, max_iter=50):
        """
        WMMSE update of Ftilde with sum power constraint:
            ||Ftilde||_F^2 <= P_sum = sum(self.P)
        Solved via KKT + bisection.
        """
        N = self.Nc * self.Nt
        P_sum = np.sum(self.P)

        # Build A and B
        A = np.zeros((N, N), dtype=np.complex128)
        B_blocks = []

        for k in range(self.K):
            Hk = self.H[k]
            Gk = Gamma[k]
            Ok = Omega[k]

            A += self.beta[k] * (
                Hk.conj().T @ Gk @ Ok @ Gk.conj().T @ Hk
            )

            B_blocks.append(
                self.beta[k] * Hk.conj().T @ Gk @ Ok
            )

        A += 1e-9 * np.eye(N, dtype=np.complex128)
        B = np.hstack(B_blocks)

        # Unconstrained solution
        F0 = solve(A, B)
        power0 = np.linalg.norm(F0, 'fro') ** 2

        if power0 <= P_sum:
            self.Ftilde = F0
            return

        # Bisection on lambda
        lam_low = 0.0
        lam_high = 1.0

        # Find upper bound
        while True:
            F = solve(A + lam_high * np.eye(N, dtype=np.complex128), B)
            if np.linalg.norm(F, 'fro') ** 2 <= P_sum:
                break
            lam_high *= 2.0

        for _ in range(max_iter):
            lam = 0.5 * (lam_low + lam_high)
            F = solve(A + lam * np.eye(N, dtype=np.complex128), B)
            power = np.linalg.norm(F, 'fro') ** 2

            if abs(power - P_sum) / P_sum < tol:
                break

            if power > P_sum:
                lam_low = lam
            else:
                lam_high = lam

        self.Ftilde = F

    # =====================================================
    # P2: AO update F_D (numpy only)
    # =====================================================
    def update_FD(self):
        FA = self.build_FA()
        A = FA.conj().T @ FA + 1e-9 * np.eye(self.Nt, dtype=np.complex128)
        B = FA.conj().T @ self.Ftilde

        lam = np.zeros(self.Nt)

        for _ in range(30):
            FD = solve(A + np.diag(lam), B)
            power = np.sum(np.abs(FD) ** 2, axis=1)
            violation = power - self.P

            if np.max(violation) < 1e-6:
                break

            for n in range(self.Nt):
                lam[n] = lam[n] * 1.2 + 1e-6 if violation[n] > 0 else lam[n] * 0.5

        self.FD = FD

    # =====================================================
    # P2: AO update F_A (autograd-safe)
    # =====================================================
    def update_FA(self, step_size=0.2, inner_iter=10):
        """
        Explicit Riemannian gradient descent for updating F_A.
        """
        Nt, Nc = self.Nt, self.Nc
        eta = self.eta
        FD = self.FD
        Ftilde = self.Ftilde

        # Reshape Ftilde into Nt blocks: each Nc x Ns
        Ftilde_blocks = [
            Ftilde[n * Nc:(n + 1) * Nc, :]
            for n in range(Nt)
        ]

        for n in range(Nt):
            u = self.u[n]
            Gn = np.diag(eta[n])
            FDn = FD[n, :].reshape(1, -1)

            # Build An and bn
            power_n = np.sum(np.abs(FDn) ** 2)
            An = 0.25 * (Gn.conj().T @ Gn) * power_n

            bn = 0.5 * Gn.conj().T @ (
                Ftilde_blocks[n] @ FDn.conj().T
                - 0.5j * eta[n].reshape(-1, 1) @ FDn @ FDn.conj().T
            ).flatten()

            # Riemannian gradient descent
            for _ in range(inner_iter):
                grad = 2 * (An @ u - bn)

                # Project to tangent space of |u_m| = 1
                grad = grad - np.real(grad * np.conj(u)) * u

                # Gradient step + retraction
                u = u - step_size * grad
                u = np.exp(1j * np.angle(u))

            self.u[n] = u

    # =====================================================
    # Main solve: strictly two stages
    # =====================================================
    def solve(self, verbose=True, tol_wmmse=1e-6, tol_ao=1e-6):
        # -------------------------
        # Stage 1: solve P1 (WMMSE)
        # -------------------------
        if verbose:
            print("=== Stage 1: Solving Problem P1 (WMMSE) ===")

        prev_obj = -np.inf

        for it in range(self.max_iterP1):
            Gamma, Omega = self.update_Gamma_Omega()
            self.update_Ftilde(Gamma, Omega)

            curr_obj = self.compute_WSR_Ftilde()

            if verbose:
                print(
                    f"[P1] Iter {it + 1:2d}, "
                    f"WSR = {curr_obj:.6f} bit/s/Hz"
                )

            if it > 0:
                rel_impr = abs(curr_obj - prev_obj) / max(abs(prev_obj), 1e-12)
                if rel_impr < tol_wmmse:
                    if verbose:
                        print(
                            f"[P1] Converged "
                            f"(relative improvement {rel_impr:.2e})"
                        )
                    break

            prev_obj = curr_obj

        Ftilde_star = self.Ftilde.copy()

        # -------------------------
        # Stage 2: solve P2 (AO)
        # -------------------------
        if verbose:
            print("\n=== Stage 2: Solving Problem P2 (AO) ===")

        prev_err = np.inf

        for it in range(self.max_iterP2):
            self.update_FD()
            self.update_FA()

            FA = self.build_FA()
            err = np.linalg.norm(Ftilde_star - FA @ self.FD, 'fro') ** 2

            if verbose:
                print(
                    f"[P2] Iter {it + 1:2d}, "
                    f"||F̃* − F_A F_D||² = {err:.3e}"
                )

            if it > 0:
                rel_impr = abs(prev_err - err) / max(prev_err, 1e-12)
                if rel_impr < tol_ao:
                    if verbose:
                        print(
                            f"[P2] Converged "
                            f"(relative improvement {rel_impr:.2e})"
                        )
                    break

            prev_err = err

        return self.build_FA(), self.FD
