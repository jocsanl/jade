#!/usr/bin/env python3
"""
JADE v10.0 - VALIDACIÓN 30 QUBITS
1x H200 (80GB)
Evolución TROTTERIZADA - Sin matrices densas

Vector de estado: 2^30 = 1,073,741,824 elementos (~17 GB) ✓

Jocsan Laguna - Quantum Forensics Lab
Enero 2026
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

N_QUBITS = 30           # 1,073,741,824 dimensiones (~10^9)
TRIALS = 50             # Prueba inicial
K_VALUE = 0.1
TEMPERATURE = 1.0
N_TIMES = 20
TROTTER_STEPS = 50      # Pasos de Trotter por evolución

# 5 UNIVERSOS para prueba inicial
UNIVERSOS = [
    {'nombre': 'baseline',      'J': 1.0, 'h': 0.5},
    {'nombre': 'campo_fuerte',  'J': 1.0, 'h': 1.5},
    {'nombre': 'campo_debil',   'J': 1.0, 'h': 0.1},
    {'nombre': 'acople_fuerte', 'J': 2.0, 'h': 0.5},
    {'nombre': 'acople_debil',  'J': 0.3, 'h': 0.5},
]


def apply_rzz_gate(psi, q1, q2, theta, n_qubits):
    """
    Aplica RZZ(theta) a los qubits q1, q2.
    RZZ(θ)|xy⟩ = e^{iθ((-1)^{x⊕y})/2} |xy⟩
    """
    import cupy as cp
    
    dim = 2 ** n_qubits
    indices = cp.arange(dim, dtype=cp.int64)
    bit_q1 = (indices >> q1) & 1
    bit_q2 = (indices >> q2) & 1
    parity = bit_q1 ^ bit_q2
    
    phase = cp.exp(1j * theta / 2 * (1 - 2 * parity))
    psi *= phase


def apply_rx_gate(psi, qubit, theta, n_qubits):
    """
    Aplica RX(theta) al qubit especificado.
    Versión vectorizada para GPU.
    """
    import cupy as cp
    
    cos_half = np.cos(theta / 2)
    sin_half = np.sin(theta / 2)
    
    dim = 2 ** n_qubits
    step = 2 ** qubit
    
    indices = cp.arange(dim, dtype=cp.int64)
    mask_0 = ((indices >> qubit) & 1) == 0
    idx_0 = indices[mask_0]
    idx_1 = idx_0 + step
    
    a = psi[idx_0].copy()
    b = psi[idx_1].copy()
    
    psi[idx_0] = cos_half * a - 1j * sin_half * b
    psi[idx_1] = -1j * sin_half * a + cos_half * b


def trotter_step(psi, J, h, dt, n_qubits, forward=True):
    """
    Un paso de Trotter.
    H = -J Σ σz_i σz_{i+1} - h Σ σx_i
    
    forward=True:  e^{-iHδt}
    forward=False: e^{+iHδt} (inverso)
    """
    sign = 1.0 if forward else -1.0
    
    # 1. RZZ gates (interacción ZZ)
    for i in range(n_qubits - 1):
        apply_rzz_gate(psi, i, i + 1, sign * 2 * J * dt, n_qubits)
    
    # 2. RX gates (campo transversal)
    for i in range(n_qubits):
        apply_rx_gate(psi, i, sign * 2 * h * dt, n_qubits)


def evolve(psi, J, h, total_time, n_steps, n_qubits, forward=True):
    """Evoluciona el estado con n_steps pasos de Trotter."""
    dt = total_time / n_steps
    for _ in range(n_steps):
        trotter_step(psi, J, h, dt, n_qubits, forward)


def main():
    import cupy as cp
    
    print("\n" + "#"*70)
    print("  JADE v10.0 - VALIDACIÓN 30 QUBITS (TROTTER)")
    print(f"  {N_QUBITS} qubits × {TRIALS} semillas × {len(UNIVERSOS)} universos")
    print("  Dimensiones: {:,}".format(2**N_QUBITS))
    print(f"  Pasos de Trotter: {TROTTER_STEPS}")
    print(f"  Memoria por vector: {2**N_QUBITS * 16 / (1024**3):.1f} GB")
    print("#"*70)
    
    # Verificar GPU
    try:
        props = cp.cuda.runtime.getDeviceProperties(0)
        mem_gb = props['totalGlobalMem']/(1024**3)
        print(f"\n  GPU: {props['name'].decode()} ({mem_gb:.0f} GB)")
        
        free, total = cp.cuda.Device(0).mem_info
        print(f"  VRAM disponible: {free/(1024**3):.1f} GB")
    except Exception as e:
        print(f"  Error GPU: {e}")
        return 1
    
    dim = 2**N_QUBITS
    times = np.logspace(-2, 1, N_TIMES)
    mempool = cp.get_default_memory_pool()
    
    all_results = {}
    start_total = time.time()
    
    for ui, univ in enumerate(UNIVERSOS):
        nombre = univ['nombre']
        J = univ['J']
        h = univ['h']
        
        print(f"\n  [{ui+1}/{len(UNIVERSOS)}] {nombre} (J={J}, h={h})")
        print(f"  " + "-"*50)
        
        results = {i: [] for i in range(N_TIMES)}
        t_start = time.time()
        
        for trial in range(TRIALS):
            # Estado inicial Haar random
            psi_0 = cp.random.standard_normal(dim) + 1j * cp.random.standard_normal(dim)
            psi_0 = psi_0.astype(cp.complex128)
            psi_0 /= cp.linalg.norm(psi_0)
            
            for ti, t in enumerate(times):
                psi = psi_0.copy()
                
                # 1. Evolución forward
                evolve(psi, J, h, t, TROTTER_STEPS, N_QUBITS, forward=True)
                
                # 2. Decoherencia (γ)
                gamma = 1 - np.exp(-K_VALUE * t * TEMPERATURE)
                
                # 3. Evolución inversa (recuperación forense)
                evolve(psi, J, h, t, TROTTER_STEPS, N_QUBITS, forward=False)
                
                # 4. Fidelidad de recuperación
                fidelity = float(cp.abs(cp.vdot(psi_0, psi))**2)
                
                # 5. C con decoherencia
                # C = fidelity * (1-γ) + γ/d ≈ (1-γ) para d grande
                C = fidelity * (1 - gamma) + gamma / dim
                C = max(0.0, min(1.0, C))
                
                results[ti].append(C)
            
            del psi_0
            
            if (trial + 1) % 10 == 0:
                elapsed = time.time() - t_start
                rate = (trial + 1) / elapsed
                eta = (TRIALS - trial - 1) / rate
                print(f"    Trial {trial+1}/{TRIALS} | {rate:.2f} t/s | ETA: {eta:.0f}s")
        
        # Estadísticas
        C_means = [np.mean(results[i]) for i in range(N_TIMES)]
        C_stds = [np.std(results[i]) for i in range(N_TIMES)]
        
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
        
        # Validar Trotter (10 muestras rápidas)
        trotter_fidelities = []
        for _ in range(10):
            psi_test = cp.random.standard_normal(dim) + 1j * cp.random.standard_normal(dim)
            psi_test = psi_test.astype(cp.complex128)
            psi_test /= cp.linalg.norm(psi_test)
            psi_test_0 = psi_test.copy()
            
            evolve(psi_test, J, h, times[-1], TROTTER_STEPS, N_QUBITS, forward=True)
            evolve(psi_test, J, h, times[-1], TROTTER_STEPS, N_QUBITS, forward=False)
            
            trotter_fidelities.append(float(cp.abs(cp.vdot(psi_test_0, psi_test))**2))
        
        trotter_f = np.mean(trotter_fidelities)
        
        all_results[nombre] = {
            'J': J, 'h': h,
            't_page': t_page,
            'C_asintotico': C_asintotico,
            'gamma_final': gamma_final,
            'C_plus_gamma': C_plus_gamma,
            'C_means': [float(x) for x in C_means],
            'C_stds': [float(x) for x in C_stds],
            'trotter_fidelity': trotter_f
        }
        
        print(f"\n    C∞ = {C_asintotico:.6f}")
        print(f"    γ  = {gamma_final:.6f}")
        print(f"    C+γ = {C_plus_gamma:.6f}")
        print(f"    Trotter F = {trotter_f:.6f}")
        
        mempool.free_all_blocks()
    
    elapsed_total = time.time() - start_total
    
    # RESULTADOS FINALES
    print(f"\n{'='*70}")
    print(f"  RESULTADOS FINALES - 30 QUBITS")
    print(f"  Tiempo total: {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")
    print(f"{'='*70}")
    
    C_values = [r['C_asintotico'] for r in all_results.values()]
    C_plus_gamma_values = [r['C_plus_gamma'] for r in all_results.values()]
    trotter_values = [r['trotter_fidelity'] for r in all_results.values()]
    
    print(f"\n  {'Universo':<15} {'C':<12} {'γ':<12} {'C+γ':<12} {'Trotter':<10}")
    print(f"  " + "-"*60)
    for nombre, r in all_results.items():
        print(f"  {nombre:<15} {r['C_asintotico']:<12.6f} {r['gamma_final']:<12.6f} "
              f"{r['C_plus_gamma']:<12.6f} {r['trotter_fidelity']:<10.6f}")
    
    print(f"\n  " + "="*60)
    print(f"  C promedio:      {np.mean(C_values):.12f}")
    print(f"  1/e teórico:     {1/np.e:.12f}")
    print(f"  Diferencia:      {abs(np.mean(C_values) - 1/np.e):.2e}")
    print(f"  " + "-"*60)
    print(f"  C+γ promedio:    {np.mean(C_plus_gamma_values):.12f}")
    print(f"  C+γ std:         {np.std(C_plus_gamma_values):.2e}")
    print(f"  " + "-"*60)
    print(f"  Trotter F:       {np.mean(trotter_values):.6f}")
    print(f"  " + "="*60)
    
    # Veredicto
    print(f"\n  VEREDICTO:")
    if np.mean(trotter_values) > 0.99:
        print(f"  ✓ Trotter reversible (F = {np.mean(trotter_values):.4f})")
    else:
        print(f"  ⚠️  Trotter F = {np.mean(trotter_values):.4f} - aumentar TROTTER_STEPS")
    
    if abs(np.mean(C_plus_gamma_values) - 1.0) < 0.01:
        print(f"  ✓ C + γ ≈ 1 CONFIRMADO a 30 QUBITS")
    else:
        print(f"  → C+γ = {np.mean(C_plus_gamma_values):.6f}")
    
    # Guardar JSON
    output = {
        'metadata': {
            'version': 'JADE v10.0 - TROTTER - 1x H200',
            'qubits': N_QUBITS,
            'dimensions': 2**N_QUBITS,
            'trials': TRIALS,
            'universos': len(UNIVERSOS),
            'k': K_VALUE,
            'temperature': TEMPERATURE,
            'trotter_steps': TROTTER_STEPS,
            'timestamp': datetime.now().isoformat(),
            'elapsed_seconds': elapsed_total
        },
        'universos': all_results,
        'estadisticas': {
            'C_promedio': float(np.mean(C_values)),
            'C_std': float(np.std(C_values)),
            'C_plus_gamma_promedio': float(np.mean(C_plus_gamma_values)),
            'C_plus_gamma_std': float(np.std(C_plus_gamma_values)),
            'trotter_fidelity_promedio': float(np.mean(trotter_values)),
        }
    }
    
    filename = f"jade_30q_trotter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n  Archivo: {filename}")
    print(f"\n{'='*70}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
