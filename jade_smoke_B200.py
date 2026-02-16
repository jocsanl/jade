#!/usr/bin/env python3
"""
JADE SMOKE TEST — B200 OPTIMIZED
=================================

Prueba de humo en 3 fases para validar B200 y estimar tiempos de cluster.

OPTIMIZACIONES vs v11.2:
  1. Diagonal ZZ pre-computada → 1 multiply vs 47 exp+multiply por paso
  2. CUDA RawKernel para RX    → GPU pura, 0% CPU en gates
  3. Seeds en batch            → explota 192 GB VRAM (64 seeds en paralelo)
  4. Fases pre-computadas      → exp() se calcula 1 vez por (universo, tiempo)

PROBLEMA DIAGNOSTICADO (v11.2 en B200):
  CPU al 98%, GPU VRAM al 0%, GPU utilization al 23%.
  Causa: apply_rx_all hace loop Python con .copy() → CPU-bound.

FASES DEL SMOKE TEST:
  Fase 1: Benchmark (1 universo × 10 seeds)     → seeds/s en B200
  Fase 2: Muestreo  (6 universos × 20 seeds)    → verificar C+γ=1
  Fase 3: Extrapolación automática               → plan para 340K trayectorias

Compatible con formato JSON de v11.2.
Jocsan Laguna — Quantum Forensics Lab | Duriva — Febrero 2026
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import time
import json
import hashlib
import numpy as np
from datetime import datetime

# ============================================================================
# GPU INIT
# ============================================================================

try:
    import cupy as cp
    cp.cuda.Device(0).use()
    props = cp.cuda.runtime.getDeviceProperties(0)
    GPU_NAME = props['name'].decode() if isinstance(props['name'], bytes) else str(props['name'])
    GPU_MEM = props['totalGlobalMem'] / (1024**3)
    free_mem, total_mem = cp.cuda.Device(0).mem_info
    FREE_GB = free_mem / (1024**3)
    print(f"  GPU: {GPU_NAME} ({GPU_MEM:.0f} GB total, {FREE_GB:.0f} GB libre)")
except Exception as e:
    print(f"  ERROR GPU: {e}")
    sys.exit(1)

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

N_QUBITS = 25
DIM = 2**N_QUBITS      # 33,554,432
TROTTER_STEPS = 80
K_VALUE = 0.1
TEMPERATURE = 1.0
J2_BASE = 1.0
N_TIMES = 20

# --- Tamaño de batch: auto-detección según VRAM ---
# Cada vector complex128: DIM × 16 bytes = 0.5 GB
# Necesitamos ~3 vectores por seed (psi_0, psi_fwd, psi_inv)
BYTES_PER_VECTOR = DIM * 16
GB_PER_VECTOR = BYTES_PER_VECTOR / (1024**3)
PARITIES_GB = 47 * DIM * 8 / (1024**3)  # 47 arrays float64
DIAG_GB = 2 * DIM * 8 / (1024**3)       # 2 diagonals float64

AVAILABLE_GB = FREE_GB - PARITIES_GB - DIAG_GB - 5  # 5 GB margen
MAX_BATCH = int(AVAILABLE_GB / (3 * GB_PER_VECTOR))
BATCH_SIZE = min(max(MAX_BATCH, 1), 128)  # cap a 128

print(f"  Vector: {GB_PER_VECTOR:.2f} GB | Paridades: {PARITIES_GB:.1f} GB")
print(f"  VRAM disponible: {AVAILABLE_GB:.0f} GB → batch_size = {BATCH_SIZE}")

# --- Smoke test: universos seleccionados ---
SMOKE_PHASE1 = [
    {'nombre': 'baseline', 'J1': 1.0, 'h': 0.5, 'J2': J2_BASE, 'lam': 0.25}
]
SMOKE_PHASE1_SEEDS = 10

SMOKE_PHASE2 = [
    {'nombre': 'integrable',     'J1': 1.0, 'h': 0.5,  'J2': J2_BASE, 'lam': 0.0},
    {'nombre': 'weak_coupling',  'J1': 0.3, 'h': 0.5,  'J2': J2_BASE, 'lam': 0.25},
    {'nombre': 'strong_field',   'J1': 1.0, 'h': 2.0,  'J2': J2_BASE, 'lam': 0.5},
    {'nombre': 'strong_J',       'J1': 8.0, 'h': 0.1,  'J2': J2_BASE, 'lam': 0.5},
    {'nombre': 'chaotic_mid',    'J1': 2.0, 'h': 1.0,  'J2': J2_BASE, 'lam': 0.5},
    {'nombre': 'extreme',        'J1': 5.0, 'h': 2.0,  'J2': J2_BASE, 'lam': 0.5},
]
SMOKE_PHASE2_SEEDS = 20

# Pares de qubits
NN_PAIRS = [(i, i+1) for i in range(N_QUBITS - 1)]    # 24 pares
NNN_PAIRS = [(i, i+2) for i in range(N_QUBITS - 2)]   # 23 pares


# ============================================================================
# CUDA RAW KERNEL — RX GATE (ELIMINA CUELLO DE BOTELLA CPU)
# ============================================================================

_rx_kernel_code = r"""
extern "C" __global__
void apply_rx_qubit(
    double* psi_real,       // parte real, shape (batch*DIM,)
    double* psi_imag,       // parte imag, shape (batch*DIM,)
    const int bit_pos,      // posición del bit del qubit
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
    
    // Calcular índices del par (i0, i1) donde difieren en bit_pos
    long long lo = local_tid & ((1LL << bit_pos) - 1);
    long long hi = local_tid >> bit_pos;
    long long i0 = (hi << (bit_pos + 1)) | lo;
    long long i1 = i0 | (1LL << bit_pos);
    
    long long offset = (long long)b * dim;
    long long g0 = offset + i0;
    long long g1 = offset + i1;
    
    // Leer amplitudes
    double a_r = psi_real[g0], a_i = psi_imag[g0];
    double b_r = psi_real[g1], b_i = psi_imag[g1];
    
    // RX(theta): |0⟩ → cos(θ/2)|0⟩ + i·sin(θ/2)|1⟩
    //            |1⟩ → i·sin(θ/2)|0⟩ + cos(θ/2)|1⟩
    // Multiplicar por i = (real*0 - imag*1, real*1 + imag*0) = (-imag, real)
    // new_a = cos*a + i*sin*b = (cos*a_r - sin*b_i, cos*a_i + sin*b_r)
    // new_b = i*sin*a + cos*b = (-sin*a_i + cos*b_r, sin*a_r + cos*b_i)
    
    psi_real[g0] = cos_a * a_r - sin_a * b_i;
    psi_imag[g0] = cos_a * a_i + sin_a * b_r;
    psi_real[g1] = -sin_a * a_i + cos_a * b_r;
    psi_imag[g1] = sin_a * a_r + cos_a * b_i;
}
"""

rx_kernel = cp.RawKernel(_rx_kernel_code, 'apply_rx_qubit')

# Qubit bit positions (pre-compute)
QUBIT_BIT_POS = [N_QUBITS - 1 - qi for qi in range(N_QUBITS)]
HALF_DIM = DIM // 2


def apply_rx_all_gpu(psi_real, psi_imag, angle, batch_size):
    """RX(angle) a todos los qubits via CUDA kernel. 100% GPU."""
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    threads = 256
    
    for qi in range(N_QUBITS):
        total_work = HALF_DIM * batch_size
        blocks = (total_work + threads - 1) // threads
        rx_kernel(
            (blocks,), (threads,),
            (psi_real, psi_imag, QUBIT_BIT_POS[qi],
             c, s, np.int64(HALF_DIM), np.int64(DIM), np.int32(batch_size))
        )


# ============================================================================
# PARIDADES Y DIAGONALES PRE-COMPUTADAS
# ============================================================================

DIAG_NN = None   # sum of (1-2*parity) for NN pairs → shape (DIM,)
DIAG_NNN = None  # sum of (1-2*parity) for NNN pairs → shape (DIM,)

def init_diagonals():
    """Pre-computar diagonales ZZ. Reemplaza 47 arrays de paridad con 2."""
    global DIAG_NN, DIAG_NNN
    
    indices = cp.arange(DIM, dtype=cp.int32)
    
    print("  Calculando diagonal NN...", end=" ", flush=True)
    t0 = time.time()
    DIAG_NN = cp.zeros(DIM, dtype=cp.float64)
    for i, j in NN_PAIRS:
        bi = (indices >> (N_QUBITS - 1 - i)) & 1
        bj = (indices >> (N_QUBITS - 1 - j)) & 1
        parity = (bi ^ bj).astype(cp.float64)
        DIAG_NN += (1.0 - 2.0 * parity)
    print(f"{len(NN_PAIRS)} pares, {time.time()-t0:.1f}s")
    
    print("  Calculando diagonal NNN...", end=" ", flush=True)
    t0 = time.time()
    DIAG_NNN = cp.zeros(DIM, dtype=cp.float64)
    for i, j in NNN_PAIRS:
        bi = (indices >> (N_QUBITS - 1 - i)) & 1
        bj = (indices >> (N_QUBITS - 1 - j)) & 1
        parity = (bi ^ bj).astype(cp.float64)
        DIAG_NNN += (1.0 - 2.0 * parity)
    print(f"{len(NNN_PAIRS)} pares, {time.time()-t0:.1f}s")
    
    del indices
    cp.get_default_memory_pool().free_all_blocks()
    
    mem_diag = 2 * DIM * 8 / (1024**3)
    free, _ = cp.cuda.Device(0).mem_info
    print(f"  Diagonales: {mem_diag:.2f} GB | VRAM libre: {free/(1024**3):.0f} GB")


# ============================================================================
# MOTOR TROTTER — B200 OPTIMIZADO
# ============================================================================

def evolve_batch(psi_real, psi_imag, J1, h, J2, lam, t_total, n_steps, batch_size, inverse=False):
    """
    Evolucionar batch de estados. 100% GPU.
    
    psi_real, psi_imag: (batch_size * DIM,) arrays contiguos
    Si inverse=True: aplica e^{+iHt} (orden inverso, signos invertidos)
    """
    dt = t_total / n_steps
    use_nnn = (lam > 0)
    
    # Pre-computar fases diagonales UNA VEZ (vs 47 exp por paso en v11.2)
    phase_nn = cp.exp(1j * J1 * dt * DIAG_NN)          # (DIM,)
    phase_nnn = cp.exp(1j * lam * J2 * dt * DIAG_NNN) if use_nnn else None
    
    if inverse:
        phase_nn = cp.conj(phase_nn)
        if phase_nnn is not None:
            phase_nnn = cp.conj(phase_nnn)
    
    # Separar parte real e imaginaria de las fases para multiplicación
    # psi = psi * phase equivale a:
    #   new_real = psi_r * phase_r - psi_i * phase_i
    #   new_imag = psi_r * phase_i + psi_i * phase_r
    
    nn_r = phase_nn.real.copy()
    nn_i = phase_nn.imag.copy()
    if phase_nnn is not None:
        nnn_r = phase_nnn.real.copy()
        nnn_i = phase_nnn.imag.copy()
    
    del phase_nn, phase_nnn
    
    rx_angle = h * dt * (-1 if inverse else 1)
    
    for _ in range(n_steps):
        if inverse:
            # Orden inverso: NNN → RX → NN
            if use_nnn:
                _apply_diagonal_batch(psi_real, psi_imag, nnn_r, nnn_i, batch_size)
            apply_rx_all_gpu(psi_real, psi_imag, rx_angle, batch_size)
            _apply_diagonal_batch(psi_real, psi_imag, nn_r, nn_i, batch_size)
        else:
            # Orden normal: NN → RX → NNN
            _apply_diagonal_batch(psi_real, psi_imag, nn_r, nn_i, batch_size)
            apply_rx_all_gpu(psi_real, psi_imag, rx_angle, batch_size)
            if use_nnn:
                _apply_diagonal_batch(psi_real, psi_imag, nnn_r, nnn_i, batch_size)


def _apply_diagonal_batch(psi_r, psi_i, diag_r, diag_i, batch_size):
    """Multiplicar batch de vectores por fase diagonal. Broadcast sobre batch."""
    for b in range(batch_size):
        offset = b * DIM
        pr = psi_r[offset:offset+DIM]
        pi = psi_i[offset:offset+DIM]
        
        new_r = pr * diag_r - pi * diag_i
        new_i = pr * diag_i + pi * diag_r
        
        psi_r[offset:offset+DIM] = new_r
        psi_i[offset:offset+DIM] = new_i


# ============================================================================
# KERNEL OPTIMIZADO PARA DIAGONAL (elimina loop Python sobre batch)
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

diag_kernel = cp.RawKernel(_diag_kernel_code, 'apply_diagonal_batch')


def _apply_diagonal_batch(psi_r, psi_i, diag_r, diag_i, batch_size):
    """Multiplicar batch por diagonal — un solo kernel launch."""
    threads = 256
    total = DIM * batch_size
    blocks = (total + threads - 1) // threads
    diag_kernel(
        (blocks,), (threads,),
        (psi_r, psi_i, diag_r, diag_i, np.int64(DIM), np.int32(batch_size))
    )


# ============================================================================
# INNER PRODUCT BATCH
# ============================================================================

def batch_fidelity(psi0_r, psi0_i, psi_r, psi_i, batch_size):
    """Calcular |⟨psi0|psi⟩|² para cada elemento del batch."""
    fidelities = []
    for b in range(batch_size):
        off = b * DIM
        # ⟨psi0|psi⟩ = sum(psi0* · psi) = sum((psi0_r - i*psi0_i)(psi_r + i*psi_i))
        # real part = sum(psi0_r*psi_r + psi0_i*psi_i)
        # imag part = sum(psi0_r*psi_i - psi0_i*psi_r)
        re = float(cp.sum(psi0_r[off:off+DIM] * psi_r[off:off+DIM] + 
                          psi0_i[off:off+DIM] * psi_i[off:off+DIM]))
        im = float(cp.sum(psi0_r[off:off+DIM] * psi_i[off:off+DIM] - 
                          psi0_i[off:off+DIM] * psi_r[off:off+DIM]))
        fidelities.append(re*re + im*im)
    return fidelities


# ============================================================================
# CORRER UN UNIVERSO (BATCHED)
# ============================================================================

def run_universe_batched(J1, h, J2, lam, n_seeds, times, label=""):
    """Correr n_seeds trayectorias con batching. Retorna C_means, C_stds, etc."""
    
    n_times = len(times)
    results_by_time = [[] for _ in range(n_times)]
    
    # Procesar en batches
    n_batches = (n_seeds + BATCH_SIZE - 1) // BATCH_SIZE
    seeds_done = 0
    t_start = time.time()
    
    for bi in range(n_batches):
        actual_batch = min(BATCH_SIZE, n_seeds - seeds_done)
        
        # Generar batch de estados iniciales (split real/imag para kernels)
        psi0_complex = cp.random.standard_normal((actual_batch, DIM), dtype=cp.float64) + \
                       1j * cp.random.standard_normal((actual_batch, DIM), dtype=cp.float64)
        
        # Normalizar cada vector
        norms = cp.linalg.norm(psi0_complex, axis=1, keepdims=True)
        psi0_complex /= norms
        
        # Split en real/imag contiguos
        psi0_r = cp.ascontiguousarray(psi0_complex.real.reshape(-1))
        psi0_i = cp.ascontiguousarray(psi0_complex.imag.reshape(-1))
        del psi0_complex, norms
        
        for ti, t in enumerate(times):
            gamma = 1 - np.exp(-K_VALUE * t * TEMPERATURE)
            
            # Forward: copiar psi0, evolucionar
            fwd_r = psi0_r.copy()
            fwd_i = psi0_i.copy()
            evolve_batch(fwd_r, fwd_i, J1, h, J2, lam, t, TROTTER_STEPS, actual_batch, inverse=False)
            
            # Inverse: copiar forward, evolucionar inverso
            inv_r = fwd_r.copy()
            inv_i = fwd_i.copy()
            evolve_batch(inv_r, inv_i, J1, h, J2, lam, t, TROTTER_STEPS, actual_batch, inverse=True)
            
            # Fidelidad: |⟨psi0|U†U|psi0⟩|²
            fids = batch_fidelity(psi0_r, psi0_i, inv_r, inv_i, actual_batch)
            
            for fid in fids:
                C = fid * (1 - gamma) + gamma / DIM
                results_by_time[ti].append(C)
            
            del fwd_r, fwd_i, inv_r, inv_i
        
        del psi0_r, psi0_i
        cp.get_default_memory_pool().free_all_blocks()
        
        seeds_done += actual_batch
        elapsed = time.time() - t_start
        rate = seeds_done / elapsed
        eta = (n_seeds - seeds_done) / rate if rate > 0 else 0
        
        if label:
            print(f"    {label}: {seeds_done}/{n_seeds} seeds | "
                  f"{rate:.2f} seeds/s | ETA: {eta:.0f}s")
    
    elapsed_total = time.time() - t_start
    
    # Estadísticas
    C_means = [float(np.mean(results_by_time[i])) for i in range(n_times)]
    C_stds = [float(np.std(results_by_time[i])) for i in range(n_times)]
    
    C_asintotico = C_means[-1]
    gamma_final = 1 - np.exp(-K_VALUE * times[-1] * TEMPERATURE)
    C_plus_gamma = C_asintotico + gamma_final
    
    # t_page
    t_page = None
    for i in range(len(C_means) - 1):
        if C_means[i] >= 0.5 and C_means[i+1] < 0.5:
            t1, t2 = times[i], times[i+1]
            c1, c2 = C_means[i], C_means[i+1]
            t_page = float(t1 + (0.5 - c1) * (t2 - t1) / (c2 - c1))
            break
    
    return {
        'C_means': C_means,
        'C_stds': C_stds,
        'C_asintotico': float(C_asintotico),
        'gamma_final': float(gamma_final),
        'C_plus_gamma': float(C_plus_gamma),
        't_page': t_page,
        'elapsed': elapsed_total,
        'seeds_per_second': n_seeds / elapsed_total,
        'times': [float(t) for t in times]
    }


# ============================================================================
# VALIDACIÓN TROTTER (batched)
# ============================================================================

def validate_trotter_batched(J1, h, J2, lam, t_test=10.0, n_tests=5):
    """U†U = I check con un mini-batch."""
    fids = []
    for _ in range(n_tests):
        psi_c = cp.random.standard_normal(DIM, dtype=cp.float64) + \
                1j * cp.random.standard_normal(DIM, dtype=cp.float64)
        psi_c /= cp.linalg.norm(psi_c)
        
        psi0_r = cp.ascontiguousarray(psi_c.real.copy())
        psi0_i = cp.ascontiguousarray(psi_c.imag.copy())
        psi_r = psi0_r.copy()
        psi_i = psi0_i.copy()
        del psi_c
        
        evolve_batch(psi_r, psi_i, J1, h, J2, lam, t_test, TROTTER_STEPS, 1, inverse=False)
        evolve_batch(psi_r, psi_i, J1, h, J2, lam, t_test, TROTTER_STEPS, 1, inverse=True)
        
        re = float(cp.sum(psi0_r * psi_r + psi0_i * psi_i))
        im = float(cp.sum(psi0_r * psi_i - psi0_i * psi_r))
        F = re*re + im*im
        fids.append(F)
        
        del psi0_r, psi0_i, psi_r, psi_i
    
    cp.get_default_memory_pool().free_all_blocks()
    return float(np.mean(fids))


# ============================================================================
# MAIN — SMOKE TEST EN 3 FASES
# ============================================================================

def main():
    print(f"\n{'█'*70}")
    print(f"  JADE SMOKE TEST — B200 OPTIMIZADO")
    print(f"{'█'*70}")
    print(f"  GPU:           {GPU_NAME} ({GPU_MEM:.0f} GB)")
    print(f"  Qubits:        {N_QUBITS} ({DIM:,} dimensiones)")
    print(f"  Vector:        {GB_PER_VECTOR:.2f} GB")
    print(f"  Batch size:    {BATCH_SIZE}")
    print(f"  Trotter:       {TROTTER_STEPS} pasos")
    print(f"  Optimizado:    Diag ZZ + CUDA RX + Batch seeds")
    print(f"  H = -J₁·ZZ_nn - h·X - λ·J₂·ZZ_nnn")
    print(f"{'█'*70}\n")
    
    times = np.logspace(-2, 1, N_TIMES)  # 0.01 a 10
    all_results = {}
    benchmarks = {}
    
    # ==================================================================
    # PASO 0: Inicializar diagonales
    # ==================================================================
    print("=" * 60)
    print("  PASO 0: Inicialización")
    print("=" * 60)
    t0 = time.time()
    init_diagonals()
    t_init = time.time() - t0
    print(f"  ✓ Diagonales listas en {t_init:.1f}s\n")
    
    # Warm-up: compilar kernels
    print("  Warm-up: compilando CUDA kernels...")
    psi_test_r = cp.random.standard_normal(DIM, dtype=cp.float64)
    psi_test_i = cp.random.standard_normal(DIM, dtype=cp.float64)
    norm = float(cp.sqrt(cp.sum(psi_test_r**2 + psi_test_i**2)))
    psi_test_r /= norm
    psi_test_i /= norm
    
    # Compilar RX kernel
    apply_rx_all_gpu(psi_test_r, psi_test_i, 0.1, 1)
    # Compilar diagonal kernel
    diag_r = cp.ones(DIM, dtype=cp.float64)
    diag_i = cp.zeros(DIM, dtype=cp.float64)
    _apply_diagonal_batch(psi_test_r, psi_test_i, diag_r, diag_i, 1)
    del psi_test_r, psi_test_i, diag_r, diag_i
    cp.get_default_memory_pool().free_all_blocks()
    print("  ✓ Kernels compilados\n")
    
    start_global = time.time()
    
    # ==================================================================
    # FASE 1: BENCHMARK — 1 universo × 10 seeds
    # ==================================================================
    print("=" * 60)
    print("  FASE 1: BENCHMARK (1 universo × 10 seeds)")
    print("=" * 60)
    
    u = SMOKE_PHASE1[0]
    print(f"\n  Universo: {u['nombre']} (J₁={u['J1']}, h={u['h']}, λ={u['lam']})")
    
    # Trotter validation
    F = validate_trotter_batched(u['J1'], u['h'], u['J2'], u['lam'])
    print(f"  Trotter F = {F:.15f}")
    
    t1 = time.time()
    result = run_universe_batched(
        u['J1'], u['h'], u['J2'], u['lam'],
        SMOKE_PHASE1_SEEDS, times, label="F1"
    )
    t_phase1 = time.time() - t1
    result['trotter_fidelity'] = float(F)
    result['J1'] = u['J1']
    result['h'] = u['h']
    result['J2'] = u['J2']
    result['lambda'] = u['lam']
    
    all_results[u['nombre']] = result
    
    seeds_per_sec = SMOKE_PHASE1_SEEDS / t_phase1
    benchmarks['phase1'] = {
        'seeds': SMOKE_PHASE1_SEEDS,
        'elapsed': t_phase1,
        'seeds_per_second': seeds_per_sec,
        'C_plus_gamma': result['C_plus_gamma'],
        'trotter_F': float(F)
    }
    
    print(f"\n  ┌──────────────────────────────────────┐")
    print(f"  │ BENCHMARK RESULTADO                    │")
    print(f"  │ Seeds/s:      {seeds_per_sec:>8.3f}               │")
    print(f"  │ C+γ:          {result['C_plus_gamma']:>12.10f}      │")
    print(f"  │ Trotter F:    {F:>15.12f}       │")
    print(f"  │ Tiempo total: {t_phase1:>8.1f}s               │")
    print(f"  └──────────────────────────────────────┘")
    
    # ==================================================================
    # FASE 2: MUESTREO — 6 universos × 20 seeds
    # ==================================================================
    print(f"\n{'='*60}")
    print(f"  FASE 2: MUESTREO (6 universos × 20 seeds)")
    print(f"{'='*60}")
    
    t2_start = time.time()
    
    for ui, u in enumerate(SMOKE_PHASE2):
        print(f"\n  [{ui+1}/6] {u['nombre']} (J₁={u['J1']}, h={u['h']}, λ={u['lam']})")
        
        F = validate_trotter_batched(u['J1'], u['h'], u['J2'], u['lam'], n_tests=3)
        print(f"  Trotter F = {F:.15f}")
        
        result = run_universe_batched(
            u['J1'], u['h'], u['J2'], u['lam'],
            SMOKE_PHASE2_SEEDS, times, label=f"F2-{ui+1}"
        )
        result['trotter_fidelity'] = float(F)
        result['J1'] = u['J1']
        result['h'] = u['h']
        result['J2'] = u['J2']
        result['lambda'] = u['lam']
        
        all_results[u['nombre']] = result
        
        print(f"  ✓ C∞={result['C_asintotico']:.10f} | "
              f"C+γ={result['C_plus_gamma']:.10f} | "
              f"{result['elapsed']:.1f}s")
    
    t_phase2 = time.time() - t2_start
    
    # Resumen fase 2
    Cpg_arr = np.array([r['C_plus_gamma'] for r in all_results.values()])
    Ft_arr = np.array([r['trotter_fidelity'] for r in all_results.values()])
    
    benchmarks['phase2'] = {
        'universes': len(SMOKE_PHASE2),
        'seeds_per_universe': SMOKE_PHASE2_SEEDS,
        'elapsed': t_phase2,
        'C_plus_gamma_mean': float(np.mean(Cpg_arr)),
        'C_plus_gamma_std': float(np.std(Cpg_arr)),
        'trotter_F_mean': float(np.mean(Ft_arr))
    }
    
    print(f"\n  ┌──────────────────────────────────────┐")
    print(f"  │ FASE 2 RESUMEN                         │")
    print(f"  │ C+γ promedio: {np.mean(Cpg_arr):>12.10f}      │")
    print(f"  │ σ(C+γ):       {np.std(Cpg_arr):>12.2e}      │")
    print(f"  │ Trotter F:    {np.mean(Ft_arr):>15.12f}       │")
    print(f"  │ Tiempo:       {t_phase2:>8.1f}s               │")
    print(f"  └──────────────────────────────────────┘")
    
    # ==================================================================
    # FASE 3: EXTRAPOLACIÓN Y PLAN DE CLUSTER
    # ==================================================================
    elapsed_total = time.time() - start_global
    
    # Tomar la tasa más representativa (fase 2, que tiene más datos)
    total_seeds_phase2 = len(SMOKE_PHASE2) * SMOKE_PHASE2_SEEDS
    rate_per_universe = total_seeds_phase2 / t_phase2  # seeds/s promedio
    rate_per_seed = t_phase2 / total_seeds_phase2      # s/seed
    
    # Estimaciones
    seeds_full = 500
    universes_full = 96
    tray_full = universes_full * seeds_full      # 48,000
    tray_340k = 340_000
    
    time_1_univ_500s = seeds_full * rate_per_seed
    time_96_univ_500s = universes_full * time_1_univ_500s
    
    # Para 340K: necesitamos más universos o más seeds
    # Opción A: 96 univ × 3542 seeds = 340,032
    seeds_for_340k = int(np.ceil(tray_340k / universes_full))
    time_340k_1gpu = universes_full * seeds_for_340k * rate_per_seed
    
    # Opción B: más GPUs
    n_gpus_4h = max(1, int(np.ceil(time_340k_1gpu / (4 * 3600))))
    n_gpus_2h = max(1, int(np.ceil(time_340k_1gpu / (2 * 3600))))
    n_gpus_1h = max(1, int(np.ceil(time_340k_1gpu / (1 * 3600))))
    
    print(f"\n{'='*60}")
    print(f"  FASE 3: EXTRAPOLACIÓN Y PLAN DE CLUSTER")
    print(f"{'='*60}")
    print(f"\n  Métricas base (B200 optimizado):")
    print(f"    Rate:                {1/rate_per_seed:.3f} seeds/s")
    print(f"    Tiempo/seed:         {rate_per_seed:.2f}s")
    print(f"    Tiempo/universo×500: {time_1_univ_500s/60:.1f} min")
    
    print(f"\n  Plan A: 96 univ × 500 seeds = 48,000 tray (1×B200)")
    print(f"    Tiempo estimado:     {time_96_univ_500s/3600:.2f} horas")
    
    print(f"\n  Plan B: 340,000 trayectorias (1×B200)")
    print(f"    Seeds/universo:      {seeds_for_340k}")
    print(f"    Tiempo estimado:     {time_340k_1gpu/3600:.2f} horas")
    
    print(f"\n  Plan C: 340,000 trayectorias (cluster B200)")
    print(f"    En 4 horas:          {n_gpus_4h} × B200")
    print(f"    En 2 horas:          {n_gpus_2h} × B200")
    print(f"    En 1 hora:           {n_gpus_1h} × B200")
    
    # Comparativa v11.2 vs optimizado
    # v11.2 en B200: ~6 min por universo (estimación del usuario)
    v11_rate = 500 / (6 * 60)  # seeds/s en v11.2
    speedup = (1/rate_per_seed) / v11_rate if v11_rate > 0 else 0
    
    print(f"\n  Comparativa v11.2 vs B200-optimizado:")
    print(f"    v11.2 estimado:      {v11_rate:.3f} seeds/s")
    print(f"    B200-opt:            {1/rate_per_seed:.3f} seeds/s")
    print(f"    Speedup:             {speedup:.1f}×")
    
    benchmarks['phase3'] = {
        'rate_seeds_per_second': float(1/rate_per_seed),
        'time_per_seed_seconds': float(rate_per_seed),
        'estimate_48k_hours': float(time_96_univ_500s / 3600),
        'estimate_340k_1gpu_hours': float(time_340k_1gpu / 3600),
        'gpus_for_340k_in_4h': n_gpus_4h,
        'gpus_for_340k_in_2h': n_gpus_2h,
        'gpus_for_340k_in_1h': n_gpus_1h,
        'speedup_vs_v11': float(speedup)
    }
    
    # ==================================================================
    # VEREDICTO
    # ==================================================================
    print(f"\n  {'█'*60}")
    sigma = np.std(Cpg_arr)
    if sigma < 1e-3 and np.mean(Ft_arr) > 0.8:
        print(f"  ✓ SMOKE TEST PASSED — C+γ = {np.mean(Cpg_arr):.10f}")
        print(f"    σ(C+γ) = {sigma:.2e}")
        print(f"    F(Trotter) = {np.mean(Ft_arr):.10f}")
        print(f"    Listo para corrida completa")
    else:
        print(f"  ⚠ SMOKE TEST: revisar resultados")
        print(f"    C+γ = {np.mean(Cpg_arr):.10f} ± {sigma:.2e}")
        print(f"    F = {np.mean(Ft_arr):.10f}")
    print(f"  {'█'*60}")
    
    # ==================================================================
    # JSON
    # ==================================================================
    output = {
        'metadata': {
            'experiment': f'JADE Smoke Test — B200 Optimizado',
            'type': 'smoke_test',
            'qubits': N_QUBITS,
            'dimensions': DIM,
            'gpu': f'1× {GPU_NAME} ({GPU_MEM:.0f} GB)',
            'optimizations': [
                'Pre-computed ZZ diagonals (1 multiply vs 47 exp+multiply/step)',
                'CUDA RawKernel for RX (0% CPU in gates)',
                'Batched seed processing (batch_size={})'.format(BATCH_SIZE),
                'Pre-computed Trotter phases per (universe, time)'
            ],
            'batch_size': BATCH_SIZE,
            'trotter_steps': TROTTER_STEPS,
            'n_times': N_TIMES,
            'k': K_VALUE,
            'temperature': TEMPERATURE,
            'hamiltoniano': 'H = -J1·ZZ_nn - h·X - λ·J2·ZZ_nnn',
            'timestamp': datetime.now().isoformat(),
            'elapsed_seconds': elapsed_total,
            'elapsed_minutes': elapsed_total / 60,
        },
        'benchmarks': benchmarks,
        'estadisticas': {
            'C_plus_gamma_promedio': float(np.mean(Cpg_arr)),
            'C_plus_gamma_std': float(np.std(Cpg_arr)),
            'trotter_fidelity_promedio': float(np.mean(Ft_arr)),
            'trotter_fidelity_min': float(np.min(Ft_arr)),
        },
        'universos': {
            nombre: {
                'J1': r['J1'], 'h': r['h'], 'lambda': r['lambda'],
                'C_asintotico': r['C_asintotico'],
                'C_plus_gamma': r['C_plus_gamma'],
                't_page': r['t_page'],
                'trotter_fidelity': r['trotter_fidelity'],
                'C_means': r['C_means'],
                'C_stds': r['C_stds'],
                'times': r['times'],
                'seeds_per_second': r.get('seeds_per_second', None)
            }
            for nombre, r in sorted(all_results.items())
        }
    }
    
    filename = f"jade_smoke_B200_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    with open(filename, 'rb') as f:
        sha = hashlib.sha512(f.read()).hexdigest().upper()
    
    print(f"\n  Archivo: {filename}")
    print(f"  SHA-512: {sha[:64]}...")
    print(f"  Tiempo total smoke test: {elapsed_total/60:.1f} min\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
