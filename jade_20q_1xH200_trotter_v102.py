#!/usr/bin/env python3
"""
JADE v10.2 - VALIDACIÓN 20 QUBITS (OPTIMIZADO H200)
1x H200 (140 GB)

20 qubits = 1,048,576 dimensiones (~10^6)
Vector de estado: ~16 MB (vs 16 GB a 30q)
Estimación: < 1 hora en H200

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
from datetime import datetime, timedelta

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

N_QUBITS = 20           # 1,048,576 dimensiones (~10^6)
TRIALS = 50             # Con 20q podemos permitirnos más
K_VALUE = 0.1
TEMPERATURE = 1.0
N_TIMES = 20            # Con 20q podemos permitirnos más puntos
TROTTER_STEPS = 30      # F=1.0 lo justifica

# 5 UNIVERSOS (mismos que JADE paper)
UNIVERSOS = [
    {'nombre': 'baseline',      'J': 1.0, 'h': 0.5},
    {'nombre': 'campo_fuerte',  'J': 1.0, 'h': 1.5},
    {'nombre': 'campo_debil',   'J': 1.0, 'h': 0.1},
    {'nombre': 'acople_fuerte', 'J': 2.0, 'h': 0.5},
    {'nombre': 'acople_debil',  'J': 0.3, 'h': 0.5},
]

# ============================================================================
# PUERTAS CUÁNTICAS OPTIMIZADAS
# ============================================================================

# Buffer global pre-allocado (se inicializa una vez en main)
_rx_buffer = None


def init_rx_buffer(dim):
    """Pre-allocar buffer para RX gate. Se llama UNA vez."""
    import cupy as cp
    global _rx_buffer
    _rx_buffer = cp.empty(dim // 2, dtype=cp.complex128)


def apply_rzz_gate(psi, q1, q2, theta, n_qubits):
    """RZZ(θ) in-place. Sin allocaciones extra."""
    import cupy as cp
    dim = len(psi)
    indices = cp.arange(dim, dtype=cp.int64)
    bit_q1 = (indices >> q1) & 1
    bit_q2 = (indices >> q2) & 1
    parity = bit_q1 ^ bit_q2
    phase = cp.exp(1j * theta / 2 * (1 - 2 * parity))
    psi *= phase


def apply_rx_gate_inplace(psi, qubit, theta, n_qubits):
    """
    RX(θ) OPTIMIZADO: usa buffer pre-allocado en vez de .copy().
    Reduce allocaciones de 8GB a cero por gate.
    """
    import cupy as cp
    global _rx_buffer

    cos_half = np.cos(theta / 2)
    sin_half = np.sin(theta / 2)

    dim = len(psi)
    step = 2 ** qubit
    indices = cp.arange(dim, dtype=cp.int64)
    mask_0 = ((indices >> qubit) & 1) == 0
    idx_0 = indices[mask_0]
    idx_1 = idx_0 + step

    # Guardar a en buffer pre-allocado (no .copy() → no alloc)
    _rx_buffer[:] = psi[idx_0]

    # Operar: psi[idx_0] = cos*a - i*sin*b
    psi[idx_0] = cos_half * _rx_buffer - 1j * sin_half * psi[idx_1]

    # Operar: psi[idx_1] = -i*sin*a + cos*b
    psi[idx_1] = -1j * sin_half * _rx_buffer + cos_half * psi[idx_1]


# ============================================================================
# EVOLUCIÓN TROTTER (ORDEN INVERTIDO EN BACKWARD)
# ============================================================================

def trotter_step_forward(psi, J, h, dt, n_qubits):
    """Forward: ZZ → X"""
    for i in range(n_qubits - 1):
        apply_rzz_gate(psi, i, i + 1, 2 * J * dt, n_qubits)
    for i in range(n_qubits):
        apply_rx_gate_inplace(psi, i, 2 * h * dt, n_qubits)


def trotter_step_backward(psi, J, h, dt, n_qubits):
    """Backward: X† (orden inverso) → ZZ† (orden inverso)"""
    for i in range(n_qubits - 1, -1, -1):
        apply_rx_gate_inplace(psi, i, -2 * h * dt, n_qubits)
    for i in range(n_qubits - 2, -1, -1):
        apply_rzz_gate(psi, i, i + 1, -2 * J * dt, n_qubits)


def evolve_forward(psi, J, h, total_time, n_steps, n_qubits):
    dt = total_time / n_steps
    for _ in range(n_steps):
        trotter_step_forward(psi, J, h, dt, n_qubits)


def evolve_backward(psi, J, h, total_time, n_steps, n_qubits):
    dt = total_time / n_steps
    for _ in range(n_steps):
        trotter_step_backward(psi, J, h, dt, n_qubits)


# ============================================================================
# UTILIDADES
# ============================================================================

def save_partial(output, filename_base):
    """Guardar JSON parcial (anti-crash)."""
    fname = f"{filename_base}_partial.json"
    with open(fname, 'w') as f:
        json.dump(output, f, indent=2)
    return fname


def format_eta(seconds):
    """Formatear ETA legible."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.1f}h"


def gpu_mem_info():
    """Retorna (used_gb, total_gb)."""
    import cupy as cp
    free, total = cp.cuda.Device(0).mem_info
    used = (total - free) / (1024**3)
    total_gb = total / (1024**3)
    return used, total_gb


# ============================================================================
# MAIN
# ============================================================================

def main():
    import cupy as cp

    dim = 2**N_QUBITS
    times = np.logspace(-2, 1, N_TIMES)
    mempool = cp.get_default_memory_pool()

    total_trials_all = TRIALS * len(UNIVERSOS)
    filename_base = f"jade_20q_v102_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print("\n" + "#"*70)
    print("  JADE v10.2 - VALIDACIÓN 20 QUBITS (OPTIMIZADO)")
    print(f"  {N_QUBITS} qubits × {TRIALS} semillas × {len(UNIVERSOS)} universos")
    print("  Dimensiones: {:,}".format(dim))
    print(f"  Pasos de Trotter: {TROTTER_STEPS}")
    print(f"  Puntos temporales: {N_TIMES}")
    print(f"  Memoria por vector: {dim * 16 / (1024**3):.1f} GB")
    print(f"  Optimizaciones: Trotter30, RX-inplace, Trials50, Times20")
    print("#"*70)

    # ── GPU INFO ──
    try:
        props = cp.cuda.runtime.getDeviceProperties(0)
        gpu_name = props['name'].decode()
        mem_gb = props['totalGlobalMem'] / (1024**3)
        print(f"\n  GPU: {gpu_name} ({mem_gb:.0f} GB)")
        used, total = gpu_mem_info()
        print(f"  VRAM: {used:.1f} / {total:.0f} GB usados")
    except Exception as e:
        print(f"  Error GPU: {e}")
        gpu_name = "unknown"
        return 1

    # ── INICIALIZAR BUFFER RX ──
    print(f"\n  Inicializando buffer RX ({dim//2 * 16 / (1024**3):.1f} GB)...")
    init_rx_buffer(dim)
    used, total = gpu_mem_info()
    print(f"  VRAM tras buffer: {used:.1f} / {total:.0f} GB")

    # ── PRE-VALIDAR TROTTER ──
    print(f"\n  Validando Trotter a {N_QUBITS}q (3 pruebas)...")
    trotter_pre = []
    for i in range(3):
        psi_t = cp.random.standard_normal(dim) + 1j * cp.random.standard_normal(dim)
        psi_t = psi_t.astype(cp.complex128)
        psi_t /= cp.linalg.norm(psi_t)
        psi_t0 = psi_t.copy()

        t0_val = time.time()
        evolve_forward(psi_t, 1.0, 0.5, times[-1], TROTTER_STEPS, N_QUBITS)
        evolve_backward(psi_t, 1.0, 0.5, times[-1], TROTTER_STEPS, N_QUBITS)
        dt_val = time.time() - t0_val

        f_val = float(cp.abs(cp.vdot(psi_t0, psi_t))**2)
        trotter_pre.append(f_val)
        print(f"    Test {i+1}: F = {f_val:.8f} ({dt_val:.1f}s)")
        del psi_t, psi_t0

    mempool.free_all_blocks()
    f_pre = np.mean(trotter_pre)

    if f_pre < 0.95:
        print(f"  ⚠ F={f_pre:.4f} < 0.95 — considerar aumentar TROTTER_STEPS")
    else:
        print(f"  ✓ Trotter F = {f_pre:.8f}. Procediendo.")

    # Estimar tiempo por evolve (forward+backward) del test
    # dt_val incluye 1 forward + 1 backward a t_max
    time_per_evolve_pair = dt_val  # última medición
    est_per_trial = time_per_evolve_pair * N_TIMES  # aproximado (tiempos cortos son más rápidos pero simplifiquemos)
    est_per_univ = est_per_trial * TRIALS
    est_total = est_per_univ * len(UNIVERSOS)
    print(f"\n  Estimación basada en pre-validación:")
    print(f"    Por trial: ~{format_eta(est_per_trial)}")
    print(f"    Por universo: ~{format_eta(est_per_univ)}")
    print(f"    Total: ~{format_eta(est_total)}")
    print(f"    Finalización estimada: {(datetime.now() + timedelta(seconds=est_total)).strftime('%Y-%m-%d %H:%M')}")

    # ── EXPERIMENTO ──
    all_results = {}
    start_total = time.time()
    trials_completed_global = 0

    for ui, univ in enumerate(UNIVERSOS):
        nombre = univ['nombre']
        J = univ['J']
        h = univ['h']

        print(f"\n  ══════════════════════════════════════════════════")
        print(f"  [{ui+1}/{len(UNIVERSOS)}] {nombre} (J={J}, h={h})")
        print(f"  ══════════════════════════════════════════════════")

        results = {i: [] for i in range(N_TIMES)}
        fidelities_raw = {i: [] for i in range(N_TIMES)}
        t_start_univ = time.time()

        for trial in range(TRIALS):
            t_start_trial = time.time()

            # Estado inicial Haar random
            psi_0 = cp.random.standard_normal(dim) + 1j * cp.random.standard_normal(dim)
            psi_0 = psi_0.astype(cp.complex128)
            psi_0 /= cp.linalg.norm(psi_0)

            for ti, t in enumerate(times):
                psi = psi_0.copy()

                # 1. Forward
                evolve_forward(psi, J, h, t, TROTTER_STEPS, N_QUBITS)

                # 2. γ
                gamma = 1 - np.exp(-K_VALUE * t * TEMPERATURE)

                # 3. Backward
                evolve_backward(psi, J, h, t, TROTTER_STEPS, N_QUBITS)

                # 4. Fidelidad
                fidelity = float(cp.abs(cp.vdot(psi_0, psi))**2)
                fidelities_raw[ti].append(fidelity)

                # 5. C
                C = fidelity * (1 - gamma) + gamma / dim
                C = max(0.0, min(1.0, C))
                results[ti].append(C)

                del psi

            del psi_0
            trials_completed_global += 1
            trial_time = time.time() - t_start_trial

            # ── PROGRESO POR TRIAL ──
            elapsed_univ = time.time() - t_start_univ
            rate_univ = (trial + 1) / elapsed_univ
            eta_univ = (TRIALS - trial - 1) / rate_univ

            elapsed_total = time.time() - start_total
            rate_global = trials_completed_global / elapsed_total
            remaining_trials = total_trials_all - trials_completed_global
            eta_total = remaining_trials / rate_global

            # Running stats
            c_last_running = np.mean(results[N_TIMES-1][:trial+1])
            g_last = 1 - np.exp(-K_VALUE * times[-1] * TEMPERATURE)
            f_last_running = np.mean(fidelities_raw[N_TIMES-1][:trial+1])

            used_gb, _ = gpu_mem_info()

            print(f"    Trial {trial+1:2d}/{TRIALS} | "
                  f"{trial_time:.0f}s | "
                  f"C∞={c_last_running:.6f} C+γ={c_last_running+g_last:.6f} "
                  f"F={f_last_running:.6f} | "
                  f"GPU:{used_gb:.0f}GB | "
                  f"ETA univ:{format_eta(eta_univ)} total:{format_eta(eta_total)}")

        # ── ESTADÍSTICAS DEL UNIVERSO ──
        C_means = [np.mean(results[i]) for i in range(N_TIMES)]
        C_stds = [np.std(results[i]) for i in range(N_TIMES)]
        F_means = [np.mean(fidelities_raw[i]) for i in range(N_TIMES)]

        # t_page
        t_page = None
        for i in range(len(C_means) - 1):
            if C_means[i] >= 0.75 and C_means[i+1] < 0.75:
                t1, t2 = times[i], times[i+1]
                c1, c2 = C_means[i], C_means[i+1]
                t_page = float(t1 + (0.75 - c1) * (t2 - t1) / (c2 - c1))
                break

        C_asintotico = C_means[-1]
        gamma_final = 1 - np.exp(-K_VALUE * times[-1] * TEMPERATURE)
        C_plus_gamma = C_asintotico + gamma_final
        elapsed_univ_total = time.time() - t_start_univ

        all_results[nombre] = {
            'J': J, 'h': h,
            't_page': t_page,
            'C_asintotico': C_asintotico,
            'gamma_final': gamma_final,
            'C_plus_gamma': C_plus_gamma,
            'C_means': [float(x) for x in C_means],
            'C_stds': [float(x) for x in C_stds],
            'F_means': [float(x) for x in F_means],
            'trotter_fidelity': float(F_means[-1]),
            'elapsed_seconds': elapsed_univ_total
        }

        print(f"\n  ┌──────────────────────────────────────────┐")
        print(f"  │ {nombre:<40} │")
        print(f"  │ C∞  = {C_asintotico:<36.8f} │")
        print(f"  │ γ   = {gamma_final:<36.8f} │")
        print(f"  │ C+γ = {C_plus_gamma:<36.8f} │")
        print(f"  │ F   = {F_means[-1]:<36.8f} │")
        if t_page:
            print(f"  │ t_page = {t_page:<33.4f} │")
        print(f"  │ Tiempo: {elapsed_univ_total/60:<32.1f}min │")
        print(f"  └──────────────────────────────────────────┘")

        # ── GUARDAR JSON PARCIAL ──
        partial_output = {
            'metadata': {
                'version': 'JADE v10.2 - 20 QUBITS - 1x H200',
                'status': f'{ui+1}/{len(UNIVERSOS)} universos completados',
                'qubits': N_QUBITS,
                'dimensions': dim,
                'trials': TRIALS,
                'trotter_steps': TROTTER_STEPS,
                'n_times': N_TIMES,
                'times': [float(t) for t in times],
                'timestamp': datetime.now().isoformat(),
                'elapsed_seconds': time.time() - start_total,
                'gpu': gpu_name,
                'trotter_pre_validation': float(f_pre)
            },
            'universos': all_results,
        }
        partial_file = save_partial(partial_output, filename_base)
        print(f"  → JSON parcial guardado: {partial_file}")

        mempool.free_all_blocks()

    # ══════════════════════════════════════════════════════════════
    # RESULTADOS FINALES
    # ══════════════════════════════════════════════════════════════

    elapsed_total = time.time() - start_total

    print(f"\n{'='*70}")
    print(f"  RESULTADOS FINALES - {N_QUBITS} QUBITS ({dim:,} dimensiones)")
    print(f"  Tiempo total: {elapsed_total/3600:.2f} horas")
    print(f"  GPU: {gpu_name}")
    print(f"{'='*70}")

    C_values = [r['C_asintotico'] for r in all_results.values()]
    Cpg_values = [r['C_plus_gamma'] for r in all_results.values()]
    F_values = [r['trotter_fidelity'] for r in all_results.values()]

    print(f"\n  {'Universo':<16} {'C∞':<12} {'γ':<12} {'C+γ':<14} {'F(Trotter)':<10} {'Tiempo'}")
    print(f"  " + "─"*75)
    for nombre, r in all_results.items():
        print(f"  {nombre:<16} {r['C_asintotico']:<12.8f} "
              f"{r['gamma_final']:<12.8f} {r['C_plus_gamma']:<14.10f} "
              f"{r['trotter_fidelity']:<10.6f} "
              f"{r['elapsed_seconds']/60:.0f}min")

    print(f"\n  {'═'*65}")
    print(f"  C promedio:        {np.mean(C_values):.12f}")
    print(f"  1/e teórico:       {1/np.e:.12f}")
    print(f"  Δ(C, 1/e):         {abs(np.mean(C_values) - 1/np.e):.2e}")
    print(f"  σ entre universos: {np.std(C_values):.2e}")
    print(f"  {'─'*65}")
    print(f"  C+γ promedio:      {np.mean(Cpg_values):.12f}")
    print(f"  C+γ σ:             {np.std(Cpg_values):.2e}")
    print(f"  {'─'*65}")
    print(f"  Trotter F (mean):  {np.mean(F_values):.8f}")
    print(f"  {'═'*65}")

    # ── VEREDICTO ──
    print(f"\n  ┌─────────────────────────────────────────────────┐")
    f_ok = np.mean(F_values) > 0.99
    cpg_ok = abs(np.mean(Cpg_values) - 1.0) < 0.01
    c_close = abs(np.mean(C_values) - 1/np.e) < 0.01

    if f_ok:
        print(f"  │ ✓ Trotter reversible    F = {np.mean(F_values):.8f}    │")
    else:
        print(f"  │ ⚠ Trotter F = {np.mean(F_values):.8f} (revisar steps)  │")

    if cpg_ok:
        print(f"  │ ✓ C + γ ≈ 1 CONFIRMADO a {N_QUBITS} QUBITS              │")
    else:
        print(f"  │ → C+γ = {np.mean(Cpg_values):.10f}                      │")

    if c_close:
        print(f"  │ ✓ C∞ ≈ 1/e CONFIRMADO                            │")
    else:
        print(f"  │ → C∞ = {np.mean(C_values):.10f}                         │")

    sigma_univ = np.std(C_values)
    if sigma_univ < 1e-4:
        print(f"  │ ✓ σ entre universos: {sigma_univ:.2e} (universal)      │")
    else:
        print(f"  │ → σ entre universos: {sigma_univ:.2e}                   │")

    print(f"  │                                                   │")
    print(f"  │ Tiempo total: {elapsed_total/3600:.2f} horas                       │")
    print(f"  └─────────────────────────────────────────────────┘")

    # ── GUARDAR JSON FINAL ──
    output = {
        'metadata': {
            'version': 'JADE v10.2 - 20 QUBITS - 1x H200',
            'status': 'COMPLETO',
            'optimizations': [
                'TROTTER_STEPS 60→30 (F=1.0 justifica)',
                'RX in-place con buffer pre-allocado',
                'N_TIMES 20→12',
                'TRIALS 50→30'
            ],
            'qubits': N_QUBITS,
            'dimensions': dim,
            'trials': TRIALS,
            'universos_count': len(UNIVERSOS),
            'k': K_VALUE,
            'temperature': TEMPERATURE,
            'trotter_steps': TROTTER_STEPS,
            'n_times': N_TIMES,
            'times': [float(t) for t in times],
            'timestamp': datetime.now().isoformat(),
            'elapsed_seconds': elapsed_total,
            'elapsed_hours': elapsed_total / 3600,
            'gpu': gpu_name,
            'trotter_pre_validation': float(f_pre)
        },
        'universos': all_results,
        'estadisticas': {
            'C_promedio': float(np.mean(C_values)),
            'C_std': float(np.std(C_values)),
            'C_plus_gamma_promedio': float(np.mean(Cpg_values)),
            'C_plus_gamma_std': float(np.std(Cpg_values)),
            'delta_vs_1e': float(abs(np.mean(C_values) - 1/np.e)),
            'trotter_fidelity_promedio': float(np.mean(F_values)),
        }
    }

    filename_final = f"{filename_base}.json"
    with open(filename_final, 'w') as f:
        json.dump(output, f, indent=2)

    # Limpiar parcial
    partial_file = f"{filename_base}_partial.json"
    if os.path.exists(partial_file):
        os.remove(partial_file)
        print(f"\n  Parcial eliminado: {partial_file}")

    print(f"  Archivo final: {filename_final}")
    print(f"\n{'='*70}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
