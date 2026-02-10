#!/usr/bin/env python3
"""
JADE - THE EDGE TEST v2.0 (GPU OPTIMIZADO H200)
=================================================

Versión optimizada que USA la H200 de verdad.

Cambios vs v1.0:
  - Hamiltoniano construido directamente en GPU (bitwise ops, sin kron_op)
  - Diamond distance: todo en GPU (expm via eigendecomp, traza parcial, norma)
  - Level spacing: eigvalsh en GPU (cupy.linalg.eigvalsh)
  - n_E: 6 → 8 (mejor baño térmico, total 10q para diamond)
  - n_qubits_r: 14 → 15 (mejor estadística, H200 lo maneja fácil)
  - Samples: 200 → 500 (GPU es rápida, más estadística)
  - Ratios vectorizados (sin loop Python)

Estimación: ~5-10 min en H200 (vs ~60 min en CPU)

Jocsan Laguna - Quantum Forensics Lab
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

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

LAMBDAS = np.linspace(0.0, 1.0, 11)

J1 = 1.0
J2 = 1.0
H_FIELD = 0.5

# Experimento 1: Diamond distance
N_S = 2
N_E = 8          # ↑ de 6 (mejor baño térmico)
N_TOTAL_D = N_S + N_E  # 10 qubits (1024 dim — trivial en GPU)
SAMPLES_D = 500  # ↑ de 200 (GPU es rápida)
T_EVOLVE = 10.0

# Experimento 2: Level spacing
N_QUBITS_R = 15  # ↑ de 14 (32,768 dim — ~16 GB, cabe en H200)


# ============================================================================
# HAMILTONIANO EN GPU (sin kron_op, construido con operaciones bitwise)
# ============================================================================

def build_hamiltonian_gpu(n_qubits, J1, h, J2, lam):
    """
    Construye H(λ) directamente en GPU usando operaciones bitwise.
    
    H(λ) = -J1 Σ σz_i·σz_{i+1} - h Σ σx_i - λ·J2 Σ σz_i·σz_{i+2}
    
    Diagonal:  ZZ terms (σz es diagonal en base computacional)
    Off-diag:  X terms  (σx flipea un bit)
    
    Mucho más rápido que kron_op (sin matrices intermedias).
    """
    import cupy as cp

    dim = 2**n_qubits
    indices = cp.arange(dim, dtype=cp.int64)

    # ── DIAGONAL: ZZ interactions ──
    diag = cp.zeros(dim, dtype=cp.float64)

    # Nearest-neighbor: -J1 * σz_i · σz_{i+1}
    for i in range(n_qubits - 1):
        sz_i = 1.0 - 2.0 * ((indices >> i) & 1).astype(cp.float64)
        sz_j = 1.0 - 2.0 * ((indices >> (i + 1)) & 1).astype(cp.float64)
        diag -= J1 * sz_i * sz_j

    # Next-nearest-neighbor: -λ·J2 * σz_i · σz_{i+2}
    if lam > 0:
        for i in range(n_qubits - 2):
            sz_i = 1.0 - 2.0 * ((indices >> i) & 1).astype(cp.float64)
            sz_k = 1.0 - 2.0 * ((indices >> (i + 2)) & 1).astype(cp.float64)
            diag -= lam * J2 * sz_i * sz_k

    # Inicializar H con diagonal
    H = cp.diag(diag.astype(cp.complex128))

    # ── OFF-DIAGONAL: Transverse field -h·σx_i ──
    # σx_i flips bit i: |k⟩ → |k ⊕ 2^i⟩
    for i in range(n_qubits):
        flipped = indices ^ (1 << i)
        H[indices, flipped] -= h

    return H


# ============================================================================
# EXPERIMENTO 1: DIAMOND DISTANCE D(λ) — 100% GPU
# ============================================================================

def compute_diamond_distance_gpu(n_S, n_E, J1, h, J2, lam, t, n_samples):
    """
    Distancia diamante entre canal físico y despolarización.
    Todo en GPU: Hamiltoniano, expm (via eigendecomp), evolución, traza parcial.
    """
    import cupy as cp

    n_total = n_S + n_E
    dim_total = 2**n_total
    dim_S = 2**n_S
    dim_E = 2**n_E

    # ── Construir H y calcular U = exp(-iHt) via eigendecomposición ──
    H = build_hamiltonian_gpu(n_total, J1, h, J2, lam)

    # H = V D V†  →  U = V exp(-iDt) V†
    eigenvalues, V = cp.linalg.eigh(H)
    exp_diag = cp.exp(-1j * eigenvalues * t)
    U = (V * exp_diag[None, :]) @ V.conj().T

    del H, eigenvalues, V, exp_diag
    cp.get_default_memory_pool().free_all_blocks()

    Ds = []
    C_values = []

    for s in range(n_samples):
        # Estado inicial aleatorio del SISTEMA (Haar random en d_S)
        psi_S = cp.random.standard_normal(dim_S) + 1j * cp.random.standard_normal(dim_S)
        psi_S = psi_S.astype(cp.complex128)
        psi_S /= cp.linalg.norm(psi_S)

        # Ambiente en |0⟩
        psi_E = cp.zeros(dim_E, dtype=cp.complex128)
        psi_E[0] = 1.0

        # Estado total y evolución unitaria
        psi_total = cp.kron(psi_S, psi_E)
        psi_evolved = U @ psi_total

        # Traza parcial sobre ambiente → ρ_S
        psi_matrix = psi_evolved.reshape(dim_S, dim_E)
        rho_S = psi_matrix @ psi_matrix.conj().T

        # Fidelidad con estado inicial
        rho_0 = cp.outer(psi_S, psi_S.conj())
        fidelity = float(cp.real(cp.trace(rho_0 @ rho_S)))
        C_values.append(fidelity)

        # γ efectivo: F = (1-γ) + γ/d  →  γ = (1-F)/(1-1/d)
        gamma_eff = (1 - fidelity) / (1 - 1 / dim_S) if fidelity < 1 else 0.0

        # Despolarización ideal con este γ
        rho_depol = ((1 - gamma_eff) * rho_0
                     + gamma_eff * cp.eye(dim_S, dtype=cp.complex128) / dim_S)

        # Distancia de traza
        diff = rho_S - rho_depol
        eigenvals = cp.linalg.eigvalsh(diff)
        D_trace = float(0.5 * cp.sum(cp.abs(eigenvals)))
        Ds.append(D_trace)

    del U
    cp.get_default_memory_pool().free_all_blocks()

    return {
        'D_mean': float(np.mean(Ds)),
        'D_std': float(np.std(Ds)),
        'C_mean': float(np.mean(C_values)),
        'C_std': float(np.std(C_values)),
        'gamma_eff': float(np.mean([(1 - c) / (1 - 1 / dim_S) for c in C_values])),
        'C_plus_gamma': float(np.mean([c + (1 - c) / (1 - 1 / dim_S) * (1 - 1 / dim_S)
                                        for c in C_values]))
    }


# ============================================================================
# EXPERIMENTO 2: LEVEL SPACING r(λ) — eigenvalores en GPU
# ============================================================================

def compute_level_spacing_gpu(n_qubits, J1, h, J2, lam):
    """
    Level spacing ratio r (firma de caos cuántico).
    r ≈ 0.386 → Poisson (integrable)
    r ≈ 0.530 → GOE (caótico)
    
    Eigenvalores calculados en GPU con cupy.linalg.eigvalsh.
    """
    import cupy as cp

    H = build_hamiltonian_gpu(n_qubits, J1, h, J2, lam)

    # Eigenvalores en GPU
    eigenvalues = cp.linalg.eigvalsh(H)
    eigenvalues = cp.sort(eigenvalues)

    # Transferir a CPU para cálculo de ratios
    eigs = eigenvalues.get()

    del H, eigenvalues
    cp.get_default_memory_pool().free_all_blocks()

    # Spacings (vectorizado, sin loop Python)
    spacings = np.diff(eigs)
    spacings = spacings[spacings > 1e-12]

    s1 = spacings[:-1]
    s2 = spacings[1:]
    valid = np.maximum(s1, s2) > 1e-15
    s1_v = s1[valid]
    s2_v = s2[valid]
    ratios = np.minimum(s1_v, s2_v) / np.maximum(s1_v, s2_v)

    r_mean = float(np.mean(ratios)) if len(ratios) > 0 else 0.0
    r_std = float(np.std(ratios)) if len(ratios) > 0 else 0.0

    return {
        'r_mean': r_mean,
        'r_std': r_std,
        'n_ratios': int(len(ratios)),
        'n_eigenvalues': int(len(eigs)),
        'poisson_ref': 0.386,
        'goe_ref': 0.530
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    import cupy as cp
    global N_QUBITS_R

    print("\n" + "#" * 70)
    print("  JADE - THE EDGE TEST v2.0 (GPU)")
    print("  ¿Cuándo se activa JADE? ¿Hay transición de fase?")
    print("#" * 70)
    print(f"\n  λ: {len(LAMBDAS)} valores de {LAMBDAS[0]:.1f} a {LAMBDAS[-1]:.1f}")
    print(f"  H(λ) = -J1·ZZ_nn - h·X - λ·J2·ZZ_nnn")
    print(f"  J1={J1}, J2={J2}, h={H_FIELD}")
    print(f"\n  Exp 1: Diamond distance D(λ)")
    print(f"    n_S={N_S}, n_E={N_E}, total={N_TOTAL_D}q ({2**N_TOTAL_D:,} dim)")
    print(f"    samples={SAMPLES_D}, t={T_EVOLVE}")
    print(f"  Exp 2: Level spacing r(λ)")
    print(f"    {N_QUBITS_R} qubits ({2**N_QUBITS_R:,} dimensiones)")
    print(f"    Memoria H: {(2**N_QUBITS_R)**2 * 16 / (1024**3):.1f} GB")

    # ── GPU INFO ──
    try:
        props = cp.cuda.runtime.getDeviceProperties(0)
        gpu_name = props['name'].decode()
        mem_gb = props['totalGlobalMem'] / (1024**3)
        free, total = cp.cuda.Device(0).mem_info
        free_gb = free / (1024**3)
        print(f"\n  GPU: {gpu_name} ({mem_gb:.0f} GB, {free_gb:.0f} GB libres)")
    except Exception as e:
        print(f"\n  ✗ GPU ERROR: {e}")
        print(f"  Este script requiere GPU con CuPy.")
        return 1

    # ── WARMUP ──
    print(f"\n  Warmup GPU...", end=" ", flush=True)
    t_warm = time.time()
    _ = cp.linalg.eigvalsh(cp.random.standard_normal((256, 256), dtype=cp.float64))
    cp.cuda.Stream.null.synchronize()
    print(f"OK ({time.time()-t_warm:.1f}s)")

    # ── PRE-TEST: verificar que eigvalsh funciona al tamaño target ──
    print(f"  Pre-test eigvalsh {N_QUBITS_R}q ({2**N_QUBITS_R:,} dim)...", end=" ", flush=True)
    t_pre = time.time()
    try:
        dim_test = 2**N_QUBITS_R
        H_test = cp.random.standard_normal((dim_test, dim_test)).astype(cp.float64)
        H_test = (H_test + H_test.T) / 2  # Make symmetric
        eigs_test = cp.linalg.eigvalsh(H_test)
        cp.cuda.Stream.null.synchronize()
        del H_test, eigs_test
        cp.get_default_memory_pool().free_all_blocks()
        print(f"OK ({time.time()-t_pre:.1f}s)")
    except Exception as e:
        print(f"\n  ✗ eigvalsh falló a {N_QUBITS_R}q: {e}")
        print(f"  Reduciendo a 14 qubits...")
        N_QUBITS_R = 14

    # ══════════════════════════════════════════════════════════════
    # EXPERIMENTO
    # ══════════════════════════════════════════════════════════════

    results_by_lambda = {}
    start_total = time.time()

    for li, lam in enumerate(LAMBDAS):
        t_start = time.time()

        # Label
        if lam == 0:
            label = "INTEGRABLE (sin segundo vecino)"
        elif lam < 0.3:
            label = "Perturbación débil"
        elif lam < 0.7:
            label = "Zona de transición"
        else:
            label = "CAÓTICO (scrambling activo)"

        print(f"\n  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"  [{li+1}/{len(LAMBDAS)}] λ = {lam:.2f} → {label}")
        print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        # ── Exp 1: Diamond distance ──
        print(f"    [D] Diamond distance ({SAMPLES_D} samples, {N_TOTAL_D}q)...",
              end=" ", flush=True)
        t1 = time.time()
        d_result = compute_diamond_distance_gpu(
            N_S, N_E, J1, H_FIELD, J2, lam, T_EVOLVE, SAMPLES_D
        )
        dt1 = time.time() - t1
        print(f"D = {d_result['D_mean']:.6f} ± {d_result['D_std']:.6f} "
              f"(C = {d_result['C_mean']:.4f}) [{dt1:.1f}s]")

        # ── Exp 2: Level spacing ──
        print(f"    [r] Level spacing ({N_QUBITS_R}q, {2**N_QUBITS_R:,} dim)...",
              end=" ", flush=True)
        t2 = time.time()
        r_result = compute_level_spacing_gpu(N_QUBITS_R, J1, H_FIELD, J2, lam)
        dt2 = time.time() - t2

        r_val = r_result['r_mean']
        if r_val < 0.42:
            r_class = "POISSON (integrable)"
        elif r_val > 0.50:
            r_class = "GOE (caótico)"
        else:
            r_class = "TRANSICIÓN"

        print(f"r = {r_val:.4f} → {r_class} [{dt2:.1f}s]")

        elapsed_lambda = time.time() - t_start
        elapsed_total = time.time() - start_total
        remaining = len(LAMBDAS) - li - 1
        eta = elapsed_lambda * remaining

        results_by_lambda[f"{lam:.2f}"] = {
            'lambda': float(lam),
            'diamond': d_result,
            'level_spacing': r_result,
            'elapsed_seconds': elapsed_lambda
        }

        # GPU memory status
        free, total = cp.cuda.Device(0).mem_info
        used_gb = (total - free) / (1024**3)
        print(f"    Tiempo: {elapsed_lambda:.1f}s (D:{dt1:.1f}s + r:{dt2:.1f}s) | "
              f"GPU: {used_gb:.1f} GB | "
              f"ETA: {eta/60:.1f}min")

    elapsed_total = time.time() - start_total

    # ══════════════════════════════════════════════════════════════
    # RESULTADOS
    # ══════════════════════════════════════════════════════════════

    print(f"\n{'=' * 70}")
    print(f"  THE EDGE TEST v2.0 - RESULTADOS")
    print(f"  GPU: {gpu_name}")
    print(f"  Tiempo total: {elapsed_total/60:.1f} min ({elapsed_total:.0f}s)")
    print(f"{'=' * 70}")

    header = (f"  {'λ':>5} │ {'D(λ)':>10} │ {'C∞':>8} │ {'γ_eff':>8} │ "
              f"{'r':>7} │ {'Estado'}")
    print(f"\n{header}")
    print(f"  {'─'*5}─┼─{'─'*10}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*7}─┼─{'─'*20}")

    D_values = []
    r_values = []
    C_values = []
    gamma_values = []

    for lam_str, res in results_by_lambda.items():
        lam = res['lambda']
        D = res['diamond']['D_mean']
        C = res['diamond']['C_mean']
        g = res['diamond']['gamma_eff']
        r = res['level_spacing']['r_mean']

        D_values.append(D)
        r_values.append(r)
        C_values.append(C)
        gamma_values.append(g)

        if r < 0.42:
            estado = "Integrable"
        elif r > 0.50:
            estado = "Caótico"
        else:
            estado = "→ TRANSICIÓN ←"

        print(f"  {lam:5.2f} │ {D:10.6f} │ {C:8.4f} │ {g:8.4f} │ {r:7.4f} │ {estado}")

    # ── ANÁLISIS DE CONSERVACIÓN ──
    print(f"\n  {'─'*60}")
    print(f"  CONSERVACIÓN C + γ:")
    for i, lam_str in enumerate(results_by_lambda.keys()):
        lam = float(lam_str)
        C = C_values[i]
        g = gamma_values[i]
        cpg = C + g
        print(f"    λ={lam:.2f}: C={C:.6f} + γ={g:.6f} = {cpg:.6f}")

    # ── DETECTAR TRANSICIÓN ──
    print(f"\n  {'─'*60}")

    lambda_c = None
    for i in range(len(r_values) - 1):
        if r_values[i] < 0.46 and r_values[i + 1] >= 0.46:
            l1, l2 = LAMBDAS[i], LAMBDAS[i + 1]
            r1, r2 = r_values[i], r_values[i + 1]
            lambda_c = l1 + (0.46 - r1) * (l2 - l1) / (r2 - r1)
            break

    if lambda_c is not None:
        print(f"\n  ★ TRANSICIÓN DETECTADA en λ_c ≈ {lambda_c:.3f}")
        print(f"    r cruza de Poisson (0.386) a GOE (0.530)")
    else:
        # Check if always integrable or always chaotic
        if all(r < 0.42 for r in r_values):
            print(f"\n  → Todos los λ son integrables (r < 0.42)")
        elif all(r > 0.50 for r in r_values):
            print(f"\n  → Todos los λ son caóticos (r > 0.50)")
        else:
            print(f"\n  → Transición presente pero no cruza umbral 0.46 entre puntos")

    # ── CORRELACIÓN D vs r ──
    corr = np.corrcoef(D_values, r_values)[0, 1]
    print(f"\n  Correlación D vs r: {corr:.4f}")
    if abs(corr) > 0.7:
        print(f"  → FUERTE: D y r están correlacionados")
    elif abs(corr) > 0.4:
        print(f"  → MODERADA")
    else:
        print(f"  → DÉBIL (D no depende del caos)")

    # ── VEREDICTO ──
    D_range = max(D_values) - min(D_values)
    r_range = max(r_values) - min(r_values)

    print(f"\n  {'═'*60}")
    print(f"  VEREDICTO:")
    print(f"  D_range = {D_range:.6f}")
    print(f"  r_range = {r_range:.4f}")

    if D_range > 0.1 and r_range > 0.08:
        dD = np.diff(D_values)
        max_jump = np.max(np.abs(dD))
        if max_jump > D_range * 0.4:
            print(f"  → Escenario B: TRANSICIÓN ABRUPTA")
            print(f"    JADE es un ESTADO que se activa en λ_c")
            print(f"    Salto máximo en D: {max_jump:.6f}")
        else:
            print(f"  → Escenario A: Transición SUAVE")
            print(f"    JADE es propiedad emergente gradual")
    else:
        if D_range < 0.05:
            print(f"  → Escenario C: D es INVARIANTE respecto a λ")
            print(f"    La despolarización NO depende del tipo de dinámica")
            print(f"    (Consistente con universalidad algebraica de JADE)")
        else:
            print(f"  → Resultado mixto. D_range={D_range:.6f}, r_range={r_range:.4f}")

    print(f"  {'═'*60}")

    # ── GUARDAR JSON ──
    output = {
        'metadata': {
            'experiment': 'JADE Edge Test v2.0 (GPU)',
            'description': 'Phase transition sweep: integrable → chaotic',
            'hamiltonian': 'H = -J1·ZZ_nn - h·X - λ·J2·ZZ_nnn',
            'J1': J1, 'J2': J2, 'h': H_FIELD,
            'lambdas': [float(l) for l in LAMBDAS],
            'diamond_config': {
                'n_S': N_S, 'n_E': N_E,
                'n_total': N_TOTAL_D,
                'dimensions': 2**N_TOTAL_D,
                'samples': SAMPLES_D,
                't_evolve': T_EVOLVE
            },
            'level_spacing_config': {
                'n_qubits': N_QUBITS_R,
                'dimensions': 2**N_QUBITS_R,
                'matrix_size_gb': (2**N_QUBITS_R)**2 * 16 / (1024**3)
            },
            'timestamp': datetime.now().isoformat(),
            'elapsed_seconds': elapsed_total,
            'elapsed_minutes': elapsed_total / 60,
            'gpu': gpu_name,
            'version': '2.0-GPU'
        },
        'results': results_by_lambda,
        'summary': {
            'D_values': D_values,
            'r_values': r_values,
            'C_values': C_values,
            'gamma_values': gamma_values,
            'lambda_c': float(lambda_c) if lambda_c else None,
            'correlation_D_r': float(corr),
            'D_range': float(D_range),
            'r_range': float(r_range)
        }
    }

    filename = f"jade_edge_test_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n  Archivo: {filename}")
    print(f"\n{'=' * 70}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
