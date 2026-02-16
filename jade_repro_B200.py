#!/usr/bin/env python3
"""
JADE — REPRODUCIBILITY & STRESS TEST — NVIDIA B200 (192 GB)
=============================================================
Engine: B200 OPTIMIZADO (Diag ZZ + CUDA RX)

TRES EXPERIMENTOS EN UNO:

  1. REPRODUCTOR 20q: ¿C + γ = 1.0000006 se reproduce en hardware diferente?
     Original: H200 | Ahora: NVIDIA B200
     Si los 5 universos dan el mismo resultado → validación cruzada

  2. COSMOS 25q: Escalar de 20q a 25q (32× más dimensiones)
     Con 192 GB cabe perfecto (~512 MB por vector)

  3. STRESS TEST float32: ¿C + γ = 1 sobrevive con precisión reducida?
     Si σ~10⁻¹² es artefacto de float64 → float32 lo revelará
     Si C + γ ≈ 1 sobrevive en float32 → resultado más robusto

GPU: NVIDIA B200 (192 GB HBM3e)
Jocsan Laguna — Quantum Forensics Lab | Duriva
Febrero 2026
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import time
import json
import numpy as np
from datetime import datetime

# —— GPU ——
try:
    import cupy as cp
    cp.cuda.Device(0).use()
    props = cp.cuda.runtime.getDeviceProperties(0)
    GPU_NAME = props['name'].decode() if isinstance(props['name'], bytes) else str(props['name'])
    GPU_MEM = props['totalGlobalMem'] / (1024**3)
    USE_GPU = True
    print(f"  GPU: {GPU_NAME} ({GPU_MEM:.0f} GB)")
except Exception as e:
    print(f"  GPU no disponible ({e}). Abortando.")
    sys.exit(1)


# ============================================================================
# CUDA RAW KERNEL — RX GATE (100% GPU, elimina cuello de botella CPU)
# ============================================================================

_rx_kernel_code = r"""
extern "C" __global__
void apply_rx_qubit(
    double* psi_real,
    double* psi_imag,
    const int bit_pos,
    const double cos_a,
    const double sin_a,
    const long long half_dim,
    const long long dim,
    const int batch_size
) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = half_dim * (long long)batch_size;
    if (tid >= total) return;

    int b = (int)(tid / half_dim);
    long long local_tid = tid - (long long)b * half_dim;

    long long lo = local_tid & ((1LL << bit_pos) - 1);
    long long hi = local_tid >> bit_pos;
    long long i0 = (hi << (bit_pos + 1)) | lo;
    long long i1 = i0 | (1LL << bit_pos);

    long long offset = (long long)b * dim;
    long long g0 = offset + i0;
    long long g1 = offset + i1;

    double a_r = psi_real[g0], a_i = psi_imag[g0];
    double b_r = psi_real[g1], b_i = psi_imag[g1];

    psi_real[g0] = cos_a * a_r - sin_a * b_i;
    psi_imag[g0] = cos_a * a_i + sin_a * b_r;
    psi_real[g1] = -sin_a * a_i + cos_a * b_r;
    psi_imag[g1] = sin_a * a_r + cos_a * b_i;
}
"""

# float32 variant
_rx_kernel_code_f32 = _rx_kernel_code.replace('double', 'float')

rx_kernel_f64 = cp.RawKernel(_rx_kernel_code, 'apply_rx_qubit')
rx_kernel_f32 = cp.RawKernel(_rx_kernel_code_f32, 'apply_rx_qubit')


# ============================================================================
# CUDA RAW KERNEL — DIAGONAL PHASE MULTIPLY
# ============================================================================

_diag_kernel_code = r"""
extern "C" __global__
void apply_diagonal_batch(
    double* psi_real,
    double* psi_imag,
    const double* diag_real,
    const double* diag_imag,
    const long long dim,
    const int batch_size
) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = dim * (long long)batch_size;
    if (tid >= total) return;

    long long local_idx = tid % dim;

    double pr = psi_real[tid];
    double pi = psi_imag[tid];
    double dr = diag_real[local_idx];
    double di = diag_imag[local_idx];

    psi_real[tid] = pr * dr - pi * di;
    psi_imag[tid] = pr * di + pi * dr;
}
"""

_diag_kernel_code_f32 = _diag_kernel_code.replace('double', 'float')

diag_kernel_f64 = cp.RawKernel(_diag_kernel_code, 'apply_diagonal_batch')
diag_kernel_f32 = cp.RawKernel(_diag_kernel_code_f32, 'apply_diagonal_batch')


# ============================================================================
# OPTIMIZED TROTTER ENGINE — Diagonal ZZ + CUDA RX
# ============================================================================

class TrotterOptimized:
    """
    Motor Trotter optimizado para B200.
    - Diagonales ZZ pre-computadas (1 multiply/step vs N exp/step)
    - CUDA RawKernel para RX (0% CPU en gates)
    - Soporte float32/float64
    """

    def __init__(self, n_qubits, J, h, J2=0.0, lam=0.0, use_float32=False):
        self.n = n_qubits
        self.dim = 2**n_qubits
        self.use_float32 = use_float32

        dtype_float = cp.float32 if use_float32 else cp.float64
        dtype_idx = cp.int64 if n_qubits >= 25 else cp.int32

        self.J = J
        self.h = h
        self.J2 = J2
        self.lam = lam
        self.use_nnn = (lam > 0 and J2 != 0.0)

        # Pre-compute ZZ diagonals
        indices = cp.arange(self.dim, dtype=dtype_idx)

        # NN diagonal: sum of (1 - 2*parity) for all nearest-neighbor pairs
        diag_nn = cp.zeros(self.dim, dtype=dtype_float)
        for i in range(n_qubits - 1):
            bi = (indices >> (n_qubits - 1 - i)) & 1
            bj = (indices >> (n_qubits - 1 - (i + 1))) & 1
            parity = (bi ^ bj).astype(dtype_float)
            diag_nn += (1.0 - 2.0 * parity)
        self.diag_nn = diag_nn

        # NNN diagonal
        if self.use_nnn:
            diag_nnn = cp.zeros(self.dim, dtype=dtype_float)
            for i in range(n_qubits - 2):
                bi = (indices >> (n_qubits - 1 - i)) & 1
                bj = (indices >> (n_qubits - 1 - (i + 2))) & 1
                parity = (bi ^ bj).astype(dtype_float)
                diag_nnn += (1.0 - 2.0 * parity)
            self.diag_nnn = diag_nnn

        del indices
        cp.get_default_memory_pool().free_all_blocks()

        # Qubit bit positions for RX kernel
        self.qubit_bit_pos = [n_qubits - 1 - qi for qi in range(n_qubits)]
        self.half_dim = self.dim // 2

    def _apply_diagonal(self, psi_r, psi_i, diag_r, diag_i):
        """Single-vector diagonal multiply via CUDA kernel."""
        threads = 256
        blocks = (self.dim + threads - 1) // threads
        kernel = diag_kernel_f32 if self.use_float32 else diag_kernel_f64
        kernel(
            (blocks,), (threads,),
            (psi_r, psi_i, diag_r, diag_i,
             np.int64(self.dim), np.int32(1))
        )

    def _apply_rx_all(self, psi_r, psi_i, angle):
        """RX(angle) on all qubits via CUDA kernel."""
        c = float(np.cos(angle))
        s = float(np.sin(angle))
        threads = 256
        kernel = rx_kernel_f32 if self.use_float32 else rx_kernel_f64

        for qi in range(self.n):
            total_work = self.half_dim
            blocks_k = (total_work + threads - 1) // threads
            kernel(
                (blocks_k,), (threads,),
                (psi_r, psi_i, self.qubit_bit_pos[qi],
                 c, s, np.int64(self.half_dim), np.int64(self.dim), np.int32(1))
            )

    def evolve(self, psi_r, psi_i, t_total, n_steps):
        """Evolve split real/imag state vector."""
        dt = t_total / n_steps
        dtype_float = cp.float32 if self.use_float32 else cp.float64

        # Pre-compute phase diagonals ONCE
        phase_nn = cp.exp(
            (1j * self.J * dt * self.diag_nn).astype(
                cp.complex64 if self.use_float32 else cp.complex128))
        nn_r = phase_nn.real.astype(dtype_float).copy()
        nn_i = phase_nn.imag.astype(dtype_float).copy()
        del phase_nn

        if self.use_nnn:
            phase_nnn = cp.exp(
                (1j * self.lam * self.J2 * dt * self.diag_nnn).astype(
                    cp.complex64 if self.use_float32 else cp.complex128))
            nnn_r = phase_nnn.real.astype(dtype_float).copy()
            nnn_i = phase_nnn.imag.astype(dtype_float).copy()
            del phase_nnn

        rx_angle = self.h * dt

        for _ in range(n_steps):
            self._apply_diagonal(psi_r, psi_i, nn_r, nn_i)
            self._apply_rx_all(psi_r, psi_i, rx_angle)
            if self.use_nnn:
                self._apply_diagonal(psi_r, psi_i, nnn_r, nnn_i)

        return psi_r, psi_i

    def evolve_inverse(self, psi_r, psi_i, t_total, n_steps):
        """Evolve inverse (for Trotter validation)."""
        dt = t_total / n_steps
        dtype_float = cp.float32 if self.use_float32 else cp.float64

        phase_nn = cp.exp(
            (1j * self.J * dt * self.diag_nn).astype(
                cp.complex64 if self.use_float32 else cp.complex128))
        nn_r = phase_nn.real.astype(dtype_float).copy()
        nn_i = -phase_nn.imag.astype(dtype_float).copy()
        del phase_nn

        if self.use_nnn:
            phase_nnn = cp.exp(
                (1j * self.lam * self.J2 * dt * self.diag_nnn).astype(
                    cp.complex64 if self.use_float32 else cp.complex128))
            nnn_r = phase_nnn.real.astype(dtype_float).copy()
            nnn_i = -phase_nnn.imag.astype(dtype_float).copy()
            del phase_nnn

        rx_angle = -self.h * dt

        for _ in range(n_steps):
            if self.use_nnn:
                self._apply_diagonal(psi_r, psi_i, nnn_r, nnn_i)
            self._apply_rx_all(psi_r, psi_i, rx_angle)
            self._apply_diagonal(psi_r, psi_i, nn_r, nn_i)

        return psi_r, psi_i


# ============================================================================
# HELPERS — State prep and fidelity
# ============================================================================

def make_state(dim, seed, use_float32=False):
    """Generate a Haar-random state, return split (real, imag) on GPU."""
    np.random.seed(seed)
    dtype = np.float32 if use_float32 else np.float64
    psi_re = np.random.randn(dim).astype(dtype)
    psi_im = np.random.randn(dim).astype(dtype)
    norm = np.sqrt(np.sum(psi_re**2 + psi_im**2))
    psi_re /= norm
    psi_im /= norm
    return cp.asarray(psi_re), cp.asarray(psi_im)


def fidelity_split(r0, i0, r1, i1):
    """Compute |⟨ψ₀|ψ₁⟩|² from split real/imag vectors."""
    re = float(cp.sum(r0 * r1 + i0 * i1))
    im = float(cp.sum(r0 * i1 - i0 * r1))
    return re * re + im * im


# ============================================================================
# EXPERIMENTO 1: REPRODUCTOR 20 QUBITS — 5 UNIVERSOS
# ============================================================================

def run_reproductor_20q():
    """Reproduce el experimento v10.2 de 20 qubits en NVIDIA B200."""

    print(f"\n{'▓'*70}")
    print(f"  EXPERIMENTO 1: REPRODUCTOR 20 QUBITS")
    print(f"  Original: NVIDIA H200 → Ahora: {GPU_NAME}")
    print(f"  ¿C + γ = 1.0000006 se reproduce?")
    print(f"{'▓'*70}")

    N_QUBITS = 20
    DIM = 2**N_QUBITS  # 1,048,576
    N_TRIALS = 50
    TROTTER_STEPS = 30
    T_POINTS = np.logspace(-2, 1, 20)  # 0.01 a 10.0

    # 5 universos con diferentes (J, h)
    UNIVERSES = {
        'baseline':        {'J': 1.0, 'h': 0.5},
        'strong_field':    {'J': 1.0, 'h': 1.5},
        'weak_field':      {'J': 1.0, 'h': 0.1},
        'strong_coupling': {'J': 2.0, 'h': 0.5},
        'weak_coupling':   {'J': 0.3, 'h': 0.5},
    }

    gamma = 1.0 - 1.0/np.e
    one_over_e = 1.0 / np.e

    all_results = {}
    start = time.time()

    for uni_name, params in UNIVERSES.items():
        J, h = params['J'], params['h']
        print(f"\n  ── Universo: {uni_name} (J={J}, h={h}) ──")

        evolver = TrotterOptimized(N_QUBITS, J, h)

        # Validación Trotter: evolucionar y des-evolucionar
        psi0_r, psi0_i = make_state(DIM, 42)
        psi_r, psi_i = psi0_r.copy(), psi0_i.copy()
        evolver.evolve(psi_r, psi_i, 5.0, TROTTER_STEPS)
        evolver.evolve_inverse(psi_r, psi_i, 5.0, TROTTER_STEPS)
        trotter_F = fidelity_split(psi0_r, psi0_i, psi_r, psi_i)
        print(f"    Trotter F = {trotter_F:.16f}")
        del psi0_r, psi0_i, psi_r, psi_i
        cp.get_default_memory_pool().free_all_blocks()

        # Barrido temporal
        C_vs_t = []

        for t in T_POINTS:
            Cs = []
            for trial in range(N_TRIALS):
                seed = trial * 1000 + int(t * 100)
                psi0_r, psi0_i = make_state(DIM, seed)
                psi_r, psi_i = psi0_r.copy(), psi0_i.copy()

                evolver.evolve(psi_r, psi_i, t, TROTTER_STEPS)

                # Fidelidad
                fid = fidelity_split(psi0_r, psi0_i, psi_r, psi_i)

                # C analítico bajo canal despolarizante
                C = fid * (1 - gamma) + gamma / DIM
                Cs.append(C)

                del psi0_r, psi0_i, psi_r, psi_i

            C_mean = float(np.mean(Cs))
            C_vs_t.append({
                't': float(t),
                'C_mean': C_mean,
                'C_std': float(np.std(Cs)),
                'C_plus_gamma': C_mean + gamma
            })

        cp.get_default_memory_pool().free_all_blocks()

        # C asintótico (último punto)
        C_inf = C_vs_t[-1]['C_mean']
        C_plus_gamma = C_inf + gamma
        delta_vs_1e = abs(C_inf - one_over_e)

        all_results[uni_name] = {
            'J': J, 'h': h,
            'trotter_F': trotter_F,
            'C_inf': C_inf,
            'C_plus_gamma': C_plus_gamma,
            'delta_vs_1e': delta_vs_1e,
            'C_vs_t': C_vs_t
        }

        print(f"    C∞ = {C_inf:.10f}")
        print(f"    C + γ = {C_plus_gamma:.10f}")
        print(f"    Δ vs 1/e = {delta_vs_1e:.2e}")
        print(f"    Trotter F = {trotter_F:.16f}")

        del evolver
        cp.get_default_memory_pool().free_all_blocks()

    elapsed = time.time() - start

    # Resumen
    C_values = [all_results[u]['C_inf'] for u in all_results]
    C_mean_all = np.mean(C_values)
    C_std_all = np.std(C_values)

    print(f"\n  ══════════════════════════════════════════")
    print(f"  RESUMEN REPRODUCTOR 20q en {GPU_NAME}")
    print(f"  C∞ promedio: {C_mean_all:.12f}")
    print(f"  σ entre universos: {C_std_all:.2e}")
    print(f"  1/e teórico: {one_over_e:.12f}")
    print(f"  Δ: {abs(C_mean_all - one_over_e):.2e}")
    print(f"  Tiempo: {elapsed/60:.1f} min")
    print(f"  ══════════════════════════════════════════")

    return {
        'experiment': 'Reproductor 20q',
        'gpu': GPU_NAME,
        'engine': 'B200 Optimizado (Diag ZZ + CUDA RX)',
        'original_gpu': 'NVIDIA H200',
        'n_qubits': N_QUBITS,
        'n_trials': N_TRIALS,
        'trotter_steps': TROTTER_STEPS,
        'C_mean_all': float(C_mean_all),
        'C_std_all': float(C_std_all),
        'delta_vs_1e': float(abs(C_mean_all - one_over_e)),
        'elapsed_seconds': elapsed,
        'universes': all_results
    }


# ============================================================================
# EXPERIMENTO 2: COSMOS 25 QUBITS
# ============================================================================

def run_cosmos_25q():
    """COSMOS a 25 qubits — 32× más dimensiones que 20q."""

    print(f"\n{'▓'*70}")
    print(f"  EXPERIMENTO 2: COSMOS 25 QUBITS")
    print(f"  20q → 25q = 32× salto en dimensiones")
    print(f"  dim = {2**25:,} = 33,554,432")
    print(f"{'▓'*70}")

    N_QUBITS = 25
    DIM = 2**N_QUBITS
    N_TRIALS = 30     # Menos trials (cada uno tarda más)
    TROTTER_STEPS = 30
    T_POINTS = np.logspace(-2, 1, 15)

    UNIVERSES = {
        'baseline':     {'J': 1.0, 'h': 0.5},
        'strong_field': {'J': 1.0, 'h': 1.5},
        'weak_field':   {'J': 1.0, 'h': 0.1},
    }

    gamma = 1.0 - 1.0/np.e
    one_over_e = 1.0 / np.e

    all_results = {}
    start = time.time()

    for uni_name, params in UNIVERSES.items():
        J, h = params['J'], params['h']
        print(f"\n  ── Universo: {uni_name} (J={J}, h={h}) ──")

        evolver = TrotterOptimized(N_QUBITS, J, h)

        # Validación Trotter
        psi0_r, psi0_i = make_state(DIM, 42)
        psi_r, psi_i = psi0_r.copy(), psi0_i.copy()
        evolver.evolve(psi_r, psi_i, 5.0, TROTTER_STEPS)
        evolver.evolve_inverse(psi_r, psi_i, 5.0, TROTTER_STEPS)
        trotter_F = fidelity_split(psi0_r, psi0_i, psi_r, psi_i)
        print(f"    Trotter F = {trotter_F:.16f}")
        del psi0_r, psi0_i, psi_r, psi_i
        cp.get_default_memory_pool().free_all_blocks()

        C_vs_t = []

        for t_idx, t in enumerate(T_POINTS):
            Cs = []
            for trial in range(N_TRIALS):
                seed = trial * 1000 + int(t * 100)
                psi0_r, psi0_i = make_state(DIM, seed)
                psi_r, psi_i = psi0_r.copy(), psi0_i.copy()

                evolver.evolve(psi_r, psi_i, t, TROTTER_STEPS)
                fid = fidelity_split(psi0_r, psi0_i, psi_r, psi_i)
                C = fid * (1 - gamma) + gamma / DIM
                Cs.append(C)

                del psi0_r, psi0_i, psi_r, psi_i

            cp.get_default_memory_pool().free_all_blocks()

            C_mean = float(np.mean(Cs))
            C_vs_t.append({
                't': float(t),
                'C_mean': C_mean,
                'C_std': float(np.std(Cs)),
                'C_plus_gamma': C_mean + gamma
            })

            if t_idx % 5 == 0:
                print(f"    t={t:.4f}: C={C_mean:.10f}")

        C_inf = C_vs_t[-1]['C_mean']
        C_plus_gamma = C_inf + gamma
        delta = abs(C_inf - one_over_e)

        all_results[uni_name] = {
            'J': J, 'h': h,
            'trotter_F': trotter_F,
            'C_inf': C_inf,
            'C_plus_gamma': C_plus_gamma,
            'delta_vs_1e': delta,
            'C_vs_t': C_vs_t
        }

        print(f"    C∞ = {C_inf:.10f}")
        print(f"    C + γ = {C_plus_gamma:.10f}")
        print(f"    Δ vs 1/e = {delta:.2e}")

        del evolver
        cp.get_default_memory_pool().free_all_blocks()

    elapsed = time.time() - start

    C_values = [all_results[u]['C_inf'] for u in all_results]
    C_mean_all = np.mean(C_values)
    C_std_all = np.std(C_values)

    # Predicción teórica: Δ debería ser ~1/d = 1/2^25 ≈ 3e-8
    delta_predicted = gamma / DIM
    delta_actual = abs(C_mean_all - one_over_e)

    print(f"\n  ══════════════════════════════════════════")
    print(f"  RESUMEN COSMOS 25q")
    print(f"  C∞ promedio: {C_mean_all:.12f}")
    print(f"  σ entre universos: {C_std_all:.2e}")
    print(f"  Δ vs 1/e: {delta_actual:.2e}")
    print(f"  Δ predicho (γ/d): {delta_predicted:.2e}")
    print(f"  Ratio Δ_actual/Δ_predicho: {delta_actual/delta_predicted:.2f}")
    print(f"  Tiempo: {elapsed/60:.1f} min")
    print(f"  ══════════════════════════════════════════")

    return {
        'experiment': 'COSMOS 25q',
        'gpu': GPU_NAME,
        'engine': 'B200 Optimizado (Diag ZZ + CUDA RX)',
        'n_qubits': N_QUBITS,
        'dimensions': DIM,
        'n_trials': N_TRIALS,
        'trotter_steps': TROTTER_STEPS,
        'C_mean_all': float(C_mean_all),
        'C_std_all': float(C_std_all),
        'delta_vs_1e': float(delta_actual),
        'delta_predicted': float(delta_predicted),
        'elapsed_seconds': elapsed,
        'universes': all_results
    }


# ============================================================================
# EXPERIMENTO 3: STRESS TEST — float32
# ============================================================================

def run_stress_float32():
    """¿C + γ = 1 sobrevive con precisión float32?"""

    print(f"\n{'▓'*70}")
    print(f"  EXPERIMENTO 3: STRESS TEST — float32 vs float64")
    print(f"  Si σ~10⁻¹² es artefacto de float64, float32 lo revela")
    print(f"  Machine epsilon: float64 ≈ 2.2e-16, float32 ≈ 1.2e-7")
    print(f"{'▓'*70}")

    N_QUBITS = 20  # Mismo que reproductor para comparar
    DIM = 2**N_QUBITS
    N_TRIALS = 50
    TROTTER_STEPS = 30
    T_POINTS = np.logspace(-2, 1, 15)

    J, h = 1.0, 0.5  # Baseline
    gamma = 1.0 - 1.0/np.e
    one_over_e = 1.0 / np.e

    results_by_precision = {}
    start = time.time()

    for precision_name, use_f32 in [('float64', False), ('float32', True)]:
        print(f"\n  ── Precisión: {precision_name} ──")

        evolver = TrotterOptimized(N_QUBITS, J, h, use_float32=use_f32)

        # Validación Trotter
        psi0_r, psi0_i = make_state(DIM, 42, use_float32=use_f32)
        psi_r, psi_i = psi0_r.copy(), psi0_i.copy()
        evolver.evolve(psi_r, psi_i, 5.0, TROTTER_STEPS)
        evolver.evolve_inverse(psi_r, psi_i, 5.0, TROTTER_STEPS)
        trotter_F = fidelity_split(psi0_r, psi0_i, psi_r, psi_i)
        print(f"    Trotter F = {trotter_F:.16f}")
        del psi0_r, psi0_i, psi_r, psi_i
        cp.get_default_memory_pool().free_all_blocks()

        C_vs_t = []

        for t in T_POINTS:
            Cs = []
            for trial in range(N_TRIALS):
                seed = trial * 1000 + int(t * 100)
                psi0_r, psi0_i = make_state(DIM, seed, use_float32=use_f32)
                psi_r, psi_i = psi0_r.copy(), psi0_i.copy()

                evolver.evolve(psi_r, psi_i, t, TROTTER_STEPS)
                fid = fidelity_split(psi0_r, psi0_i, psi_r, psi_i)
                C = fid * (1 - gamma) + gamma / DIM
                Cs.append(C)

                del psi0_r, psi0_i, psi_r, psi_i

            C_mean = float(np.mean(Cs))
            C_vs_t.append({
                't': float(t),
                'C_mean': C_mean,
                'C_std': float(np.std(Cs)),
                'C_plus_gamma': C_mean + gamma
            })

        cp.get_default_memory_pool().free_all_blocks()

        C_inf = C_vs_t[-1]['C_mean']
        delta = abs(C_inf - one_over_e)

        results_by_precision[precision_name] = {
            'trotter_F': trotter_F,
            'C_inf': C_inf,
            'C_plus_gamma': C_inf + gamma,
            'delta_vs_1e': delta,
            'C_vs_t': C_vs_t
        }

        print(f"    C∞ = {C_inf:.10f}")
        print(f"    C + γ = {C_inf + gamma:.10f}")
        print(f"    Δ vs 1/e = {delta:.2e}")

        del evolver
        cp.get_default_memory_pool().free_all_blocks()

    elapsed = time.time() - start

    # Comparación
    f64_delta = results_by_precision['float64']['delta_vs_1e']
    f32_delta = results_by_precision['float32']['delta_vs_1e']
    f64_C = results_by_precision['float64']['C_inf']
    f32_C = results_by_precision['float32']['C_inf']

    print(f"\n  ══════════════════════════════════════════")
    print(f"  COMPARACIÓN float32 vs float64")
    print(f"  float64: C∞ = {f64_C:.10f}, Δ = {f64_delta:.2e}")
    print(f"  float32: C∞ = {f32_C:.10f}, Δ = {f32_delta:.2e}")
    print(f"  |C_f64 - C_f32| = {abs(f64_C - f32_C):.2e}")

    if f32_delta < 1e-4:
        print(f"  → C + γ = 1 SOBREVIVE en float32 (Δ < 10⁻⁴)")
        print(f"  → Resultado ROBUSTO, no artefacto de precisión")
    else:
        print(f"  → C + γ = 1 se DEGRADA en float32")
        print(f"  → La precisión numérica SÍ importa")
    print(f"  Tiempo: {elapsed/60:.1f} min")
    print(f"  ══════════════════════════════════════════")

    return {
        'experiment': 'Stress Test float32 vs float64',
        'gpu': GPU_NAME,
        'engine': 'B200 Optimizado (Diag ZZ + CUDA RX)',
        'n_qubits': N_QUBITS,
        'n_trials': N_TRIALS,
        'elapsed_seconds': elapsed,
        'results': results_by_precision
    }


# ============================================================================
# MAIN — CORRE LOS 3 EXPERIMENTOS EN SECUENCIA
# ============================================================================

def main():
    print(f"\n{'█'*70}")
    print(f"  JADE — REPRODUCIBILITY & STRESS SUITE")
    print(f"  NVIDIA B200 ({GPU_MEM:.0f} GB) — Engine Optimizado")
    print(f"  3 experimentos: Reproductor 20q | COSMOS 25q | Stress float32")
    print(f"{'█'*70}")

    start_total = time.time()
    all_output = {
        'metadata': {
            'suite': 'JADE Reproducibility & Stress Test',
            'engine': 'B200 Optimizado (Diag ZZ + CUDA RX)',
            'gpu': GPU_NAME,
            'gpu_mem_gb': GPU_MEM,
            'timestamp': datetime.now().isoformat(),
        }
    }

    # Archivo base para guardados parciales
    partial_file = f"jade_repro_B200_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    def save_partial(label):
        all_output['metadata']['last_save'] = datetime.now().isoformat()
        all_output['metadata']['partial'] = label
        with open(partial_file, 'w') as f:
            json.dump(all_output, f, indent=2, default=str)
        print(f"  💾 Guardado parcial ({label}): {partial_file}")

    # ── Experimento 1 ──
    try:
        result1 = run_reproductor_20q()
        all_output['reproductor_20q'] = result1
    except Exception as e:
        print(f"\n  ⚠ Error en Reproductor 20q: {e}")
        all_output['reproductor_20q'] = {'error': str(e)}
    save_partial('post_reproductor_20q')

    # ── Experimento 2 ──
    try:
        result2 = run_cosmos_25q()
        all_output['cosmos_25q'] = result2
    except Exception as e:
        print(f"\n  ⚠ Error en COSMOS 25q: {e}")
        all_output['cosmos_25q'] = {'error': str(e)}
    save_partial('post_cosmos_25q')

    # ── Experimento 3 ──
    try:
        result3 = run_stress_float32()
        all_output['stress_float32'] = result3
    except Exception as e:
        print(f"\n  ⚠ Error en Stress float32: {e}")
        all_output['stress_float32'] = {'error': str(e)}
    save_partial('post_stress_float32')

    elapsed_total = time.time() - start_total
    all_output['metadata']['total_elapsed_seconds'] = elapsed_total

    # ═══════════════════════════════════════════════════════════════
    # RESUMEN FINAL
    # ═══════════════════════════════════════════════════════════════

    print(f"\n{'█'*70}")
    print(f"  RESUMEN FINAL — NVIDIA B200")
    print(f"{'█'*70}")

    if 'reproductor_20q' in all_output and 'error' not in all_output['reproductor_20q']:
        r = all_output['reproductor_20q']
        print(f"\n  1. REPRODUCTOR 20q:")
        print(f"     C∞ = {r['C_mean_all']:.12f}")
        print(f"     σ = {r['C_std_all']:.2e}")
        print(f"     Original (H200): C∞ = 0.367880044009")
        print(f"     Match: {'✓' if abs(r['C_mean_all'] - 0.367880044009) < 1e-6 else '✗'}")

    if 'cosmos_25q' in all_output and 'error' not in all_output['cosmos_25q']:
        r = all_output['cosmos_25q']
        print(f"\n  2. COSMOS 25q:")
        print(f"     C∞ = {r['C_mean_all']:.12f}")
        print(f"     Δ vs 1/e = {r['delta_vs_1e']:.2e}")
        print(f"     Δ predicho = {r['delta_predicted']:.2e}")
        expected_improvement = 6.03e-7 / r['delta_vs_1e'] if r['delta_vs_1e'] > 0 else float('inf')
        print(f"     Mejora vs 20q: {expected_improvement:.1f}×")

    if 'stress_float32' in all_output and 'error' not in all_output['stress_float32']:
        r = all_output['stress_float32']
        if 'results' in r:
            f64 = r['results'].get('float64', {})
            f32 = r['results'].get('float32', {})
            print(f"\n  3. STRESS float32:")
            print(f"     float64 Δ = {f64.get('delta_vs_1e', 'N/A')}")
            print(f"     float32 Δ = {f32.get('delta_vs_1e', 'N/A')}")

    print(f"\n  Tiempo total: {elapsed_total/60:.1f} min")

    # JSON final (mismo archivo, ahora completo)
    all_output['metadata']['partial'] = 'COMPLETE'
    all_output['metadata']['last_save'] = datetime.now().isoformat()
    with open(partial_file, 'w') as f:
        json.dump(all_output, f, indent=2, default=str)

    print(f"  Archivo: {partial_file}")
    print(f"\n{'█'*70}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
