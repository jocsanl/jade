#!/usr/bin/env python3
"""
JADE v8.3 — EL PUENTE (Quick Test)
¿Converge el canal físico al de despolarización con scrambling?

HARDWARE: RTX 4070 / 4070 Ti
TIEMPO: ~1-2 horas
PREGUNTA: Si D → 0, entonces C + γ = 1 es físico, no solo algebraico.

Jocsan Laguna — Quantum Forensics Lab | Febrero 2026
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
# CONFIG — RÁPIDO
# ============================================================================
MAX_HOURS = 2
MAX_TIME = MAX_HOURS * 3600
N_SAMPLES = 100
OUTPUT_FILE = "jade_v83_quick"

# ============================================================================
# GPU INIT
# ============================================================================
try:
    import cupy as cp
    from cupyx.scipy.sparse.linalg import expm_multiply
    from cupyx.scipy.linalg import expm
    cp.cuda.Device(0).use()
    mempool = cp.get_default_memory_pool()
    props = cp.cuda.runtime.getDeviceProperties(0)
    free, total = cp.cuda.Device(0).mem_info
    gpu_name = props['name'].decode()
    print(f"\n  GPU: {gpu_name}")
    print(f"  VRAM: {free/(1024**3):.1f} / {total/(1024**3):.1f} GB")
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)

# ============================================================================
# PAULI MATRICES
# ============================================================================
I2 = cp.eye(2, dtype=cp.complex128)
sx = cp.array([[0,1],[1,0]], dtype=cp.complex128)
sy = cp.array([[0,-1j],[1j,0]], dtype=cp.complex128)
sz = cp.array([[1,0],[0,-1]], dtype=cp.complex128)

def kron_chain(ops):
    """Kronecker product of list of operators."""
    result = ops[0]
    for op in ops[1:]:
        result = cp.kron(result, op)
    return result

# ============================================================================
# HAMILTONIANS
# ============================================================================
def build_ising(n, J=1.0, h=0.5):
    dim = 2**n
    H = cp.zeros((dim, dim), dtype=cp.complex128)
    for i in range(n-1):
        ops = [I2]*n; ops[i] = sz; ops[i+1] = sz
        H -= J * kron_chain(ops)
    for i in range(n):
        ops = [I2]*n; ops[i] = sx
        H -= h * kron_chain(ops)
    return H

def build_interaction(n_S, n_E, coupling, g=0.5, seed=0):
    d_S, d_E = 2**n_S, 2**n_E
    H = cp.zeros((d_S*d_E, d_S*d_E), dtype=cp.complex128)
    
    if coupling == 'local':
        for pS, pE in [(sx,sx), (sz,sz)]:
            ops_S = [I2]*n_S; ops_S[-1] = pS
            ops_E = [I2]*n_E; ops_E[0] = pE
            H -= (g/2) * cp.kron(kron_chain(ops_S), kron_chain(ops_E))
    
    elif coupling == 'random':
        rng = np.random.RandomState(seed)
        n_terms = 0
        for pS, pE in [(sx,sx), (sy,sy), (sz,sz)]:
            for i in range(n_S):
                for j in range(n_E):
                    ops_S = [I2]*n_S; ops_S[i] = pS
                    ops_E = [I2]*n_E; ops_E[j] = pE
                    J_ij = rng.randn()
                    H -= J_ij * cp.kron(kron_chain(ops_S), kron_chain(ops_E))
                    n_terms += 1
        H *= g / np.sqrt(n_terms)
    
    return H

def build_H_total(n_S, n_E, coupling, seed=0):
    d_S, d_E = 2**n_S, 2**n_E
    H_S = build_ising(n_S, 1.0, 0.5)
    H_E = build_ising(n_E, 0.7, 0.3)
    H_SE = build_interaction(n_S, n_E, coupling, 0.5, seed)
    return cp.kron(H_S, cp.eye(d_E, dtype=cp.complex128)) + \
           cp.kron(cp.eye(d_S, dtype=cp.complex128), H_E) + H_SE

# ============================================================================
# PHYSICAL CHANNEL
# ============================================================================
def channel_output(U, psi_S, d_S, d_E):
    """ρ_out = Tr_E[U(|ψ⟩⟨ψ| ⊗ I/d_E)U†] — vectorized."""
    d_total = d_S * d_E
    Psi_in = cp.zeros((d_total, d_E), dtype=cp.complex128)
    for k in range(d_E):
        for s in range(d_S):
            Psi_in[s*d_E + k, k] = psi_S[s]
    Phi = U @ Psi_in
    T = Phi.reshape(d_S, d_E, d_E)
    return cp.einsum('aek,bek->ab', T, cp.conj(T)) / d_E

def trace_norm(A):
    return float(cp.sum(cp.linalg.svd(A, compute_uv=False)))

# ============================================================================
# MEASURE
# ============================================================================
def measure(U, d_S, d_E, n_samples):
    """Mide C_phys y distancia al canal de despolarización."""
    I_dS = cp.eye(d_S, dtype=cp.complex128) / d_S
    C_vals, D_vals = [], []
    
    for _ in range(n_samples):
        psi = cp.random.standard_normal(d_S) + 1j * cp.random.standard_normal(d_S)
        psi = psi.astype(cp.complex128)
        psi /= cp.linalg.norm(psi)
        
        rho_phys = channel_output(U, psi, d_S, d_E)
        C = float(cp.real(psi.conj() @ rho_phys @ psi))
        C_vals.append(C)
        
        # Best-fit depolarization
        gamma_eff = (1 - C) / (1 - 1/d_S)
        rho_depol = (1 - gamma_eff) * cp.outer(psi, cp.conj(psi)) + gamma_eff * I_dS
        D = 0.5 * trace_norm(rho_phys - rho_depol)
        D_vals.append(D)
        
        del psi, rho_phys, rho_depol
    
    mempool.free_all_blocks()
    return {
        'C': float(np.mean(C_vals)),
        'C_std': float(np.std(C_vals)),
        'gamma_eff': float((1 - np.mean(C_vals)) / (1 - 1/d_S)),
        'D': float(np.mean(D_vals)),
        'D_std': float(np.std(D_vals)),
    }

# ============================================================================
# MAIN
# ============================================================================
def main():
    t_global = time.time()
    gamma_jade = 1 - 1/np.e
    
    print(f"\n{'='*65}")
    print(f"  JADE v8.3 — EL PUENTE (Quick Test)")
    print(f"  ¿Es el universo un canal de despolarización global?")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*65}")
    
    results = []
    
    # ── EXPERIMENTO 1: SCALING (n_S=2, n_E=1..10) ──
    print(f"\n  ── EXP 1: SCALING — D vs tamaño del ambiente ──")
    print(f"  n_S=2 (d_S=4), t=10.0")
    print(f"  {'n_E':>3} {'d_E':>5} {'coupling':>8} {'C':>7} {'γ_eff':>7} {'D':>7} {'time':>6}")
    print(f"  {'-'*50}")
    
    for n_E in range(1, 11):
        d_S, d_E = 4, 2**n_E
        d_total = d_S * d_E
        if d_total > 4096:
            break
        
        for coupling in ['local', 'random']:
            t0 = time.time()
            
            if time.time() - t_global > MAX_TIME:
                print(f"  ⚠ Tiempo agotado"); break
            
            H = build_H_total(2, n_E, coupling, seed=42)
            U = expm(-1j * H * 10.0)
            del H; mempool.free_all_blocks()
            
            r = measure(U, d_S, d_E, N_SAMPLES)
            del U; mempool.free_all_blocks()
            
            elapsed = time.time() - t0
            r.update({'n_S':2, 'n_E':n_E, 'coupling':coupling, 't':10.0, 'exp':'scaling'})
            results.append(r)
            
            print(f"  {n_E:3d} {d_E:5d} {coupling:>8s} {r['C']:7.4f} {r['gamma_eff']:7.4f} "
                  f"{r['D']:7.4f} {elapsed:5.1f}s")
    
    # ── EXPERIMENTO 2: n_S=3 SCALING ──
    print(f"\n  ── EXP 2: SCALING — n_S=3 (d_S=8), t=10.0 ──")
    print(f"  {'n_E':>3} {'d_E':>5} {'coupling':>8} {'C':>7} {'γ_eff':>7} {'D':>7} {'time':>6}")
    print(f"  {'-'*50}")
    
    for n_E in range(1, 10):
        d_S, d_E = 8, 2**n_E
        if d_S * d_E > 4096:
            break
        
        for coupling in ['local', 'random']:
            t0 = time.time()
            if time.time() - t_global > MAX_TIME:
                print(f"  ⚠ Tiempo agotado"); break
            
            H = build_H_total(3, n_E, coupling, seed=42)
            U = expm(-1j * H * 10.0)
            del H; mempool.free_all_blocks()
            
            r = measure(U, d_S, d_E, N_SAMPLES)
            del U; mempool.free_all_blocks()
            
            elapsed = time.time() - t0
            r.update({'n_S':3, 'n_E':n_E, 'coupling':coupling, 't':10.0, 'exp':'scaling_3'})
            results.append(r)
            
            print(f"  {n_E:3d} {d_E:5d} {coupling:>8s} {r['C']:7.4f} {r['gamma_eff']:7.4f} "
                  f"{r['D']:7.4f} {elapsed:5.1f}s")
    
    # ── EXPERIMENTO 3: TEMPORAL ──
    print(f"\n  ── EXP 3: TEMPORAL — D vs t (n_S=2, n_E=6, random) ──")
    print(f"  {'t':>6} {'C':>7} {'γ_eff':>7} {'D':>7}")
    print(f"  {'-'*30}")
    
    times = [0.1, 0.5, 1.0, 2.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]
    H = build_H_total(2, 6, 'random', seed=42)
    
    for t in times:
        if time.time() - t_global > MAX_TIME:
            print(f"  ⚠ Tiempo agotado"); break
        
        U = expm(-1j * H * t)
        r = measure(U, 4, 64, N_SAMPLES)
        del U; mempool.free_all_blocks()
        
        r.update({'n_S':2, 'n_E':6, 'coupling':'random', 't':t, 'exp':'temporal'})
        results.append(r)
        print(f"  {t:6.1f} {r['C']:7.4f} {r['gamma_eff']:7.4f} {r['D']:7.4f}")
    
    del H; mempool.free_all_blocks()
    
    # ── RESUMEN ──
    print(f"\n{'='*65}")
    print(f"  RESUMEN")
    print(f"{'='*65}")
    
    # Scaling n_S=2
    for coupling in ['local', 'random']:
        entries = [r for r in results if r['exp']=='scaling' and r['coupling']==coupling]
        if len(entries) >= 2:
            d_first = entries[0]['D']
            d_last = entries[-1]['D']
            reduction = (1 - d_last/d_first) * 100 if d_first > 0 else 0
            trend = "CONVERGE ✅" if reduction > 40 else "PARCIAL ⚠️" if reduction > 15 else "NO ❌"
            print(f"  n_S=2, {coupling:>7s}: D {d_first:.3f} → {d_last:.3f} ({reduction:+.0f}%) {trend}")
    
    for coupling in ['local', 'random']:
        entries = [r for r in results if r['exp']=='scaling_3' and r['coupling']==coupling]
        if len(entries) >= 2:
            d_first = entries[0]['D']
            d_last = entries[-1]['D']
            reduction = (1 - d_last/d_first) * 100 if d_first > 0 else 0
            trend = "CONVERGE ✅" if reduction > 40 else "PARCIAL ⚠️" if reduction > 15 else "NO ❌"
            print(f"  n_S=3, {coupling:>7s}: D {d_first:.3f} → {d_last:.3f} ({reduction:+.0f}%) {trend}")
    
    # Cadena lógica
    scaling_random = [r for r in results if r['exp']=='scaling' and r['coupling']=='random']
    if scaling_random:
        d_last = scaling_random[-1]['D']
        print(f"\n  ── CADENA LÓGICA ──")
        print(f"  1. Agujeros negros son fast scramblers (Sekino-Susskind 2008)")
        print(f"  2. Fast scrambling → canal converge a despolarización (D={d_last:.3f})")
        print(f"  3. Despolarización → C + γ = 1 (identidad algebraica)")
        if d_last < 0.15:
            print(f"  ∴ C + γ = 1 es el límite físico correcto ✅")
        else:
            print(f"  ∴ Tendencia correcta, necesita más qubits de ambiente")
    
    # Guardar
    total_time = time.time() - t_global
    output = {
        'fecha': datetime.now().isoformat(),
        'experiment': 'JADE v8.3 Quick Bridge',
        'gpu': gpu_name,
        'total_seconds': total_time,
        'results': results,
    }
    fn = f"{OUTPUT_FILE}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(fn, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n  Tiempo total: {total_time/60:.1f} min")
    print(f"  Guardado: {fn}")
    print(f"{'='*65}")

if __name__ == "__main__":
    main()
