#!/usr/bin/env python3
"""
JADE v8.2 — EL PUENTE (4070 Edition)
¿Es el universo un canal de despolarización global?

OBJETIVO:
  Medir la distancia entre el canal FÍSICO (sistema acoplado a ambiente)
  y el canal de despolarización, escalando el tamaño del ambiente.

  Si D → 0 con scrambling fuerte y ambiente grande, entonces:
  Agujeros negros (fast scramblers) → despolarización → C + γ = 1

HARDWARE: NVIDIA RTX 4070 / 4070 Ti (12 GB VRAM)
TIEMPO ESTIMADO: ~4-6 horas
CHECKPOINTING: Guarda resultados cada 5 minutos

FASES:
  1. Scaling: D vs n_E para n_S = {2, 3, 4}, coupling = {local, random}
  2. Choi exacto: ||J_phys - J_depol||₁ para sistemas pequeños
  3. Temporal: D vs t para una configuración fija
  4. Resumen ejecutivo

Jocsan Laguna — Quantum Forensics Lab
Febrero 2026
"""

import os
import sys
import time
import json
import signal
import numpy as np
from datetime import datetime, timedelta

# ============================================================================
# CONFIGURACIÓN GLOBAL
# ============================================================================

MAX_HOURS = 6
MAX_TIME = MAX_HOURS * 3600
N_SAMPLES = 200          # Muestras Haar-random por configuración
CHECKPOINT_INTERVAL = 300 # Guardar cada 5 min
OUTPUT_FILE = "jade_v82_bridge"

# Límites de memoria (4070 = 12 GB)
# expm de d×d necesita ~5d² bytes de workspace
# Para d=4096: ~320 MB → OK
# Para d=8192: ~2.5 GB → posible pero ajustado
MAX_D_TOTAL_EXPM = 4096   # 2^12: seguro en 4070
MAX_D_TOTAL_PUSH = 8192   # 2^13: intentar si hay tiempo

# Configuraciones de escalamiento
# n_S: qubits del sistema (horizonte)
# n_E: qubits del ambiente (radiación/baño)
SCALING_CONFIGS = [
    # (n_S, max_n_E)
    (2, 10),  # d_S=4,   d_total max = 4096
    (3, 9),   # d_S=8,   d_total max = 4096
    (4, 8),   # d_S=16,  d_total max = 4096
]

COUPLINGS = ['local', 'random']
TIMES = [5.0, 10.0, 20.0]
N_REALIZATIONS = 3  # Para coupling aleatorio

# Para fase temporal
TEMPORAL_TIMES = [0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0]

# ============================================================================
# INICIALIZACIÓN GPU
# ============================================================================

def init_gpu():
    """Inicializa CuPy y verifica la GPU."""
    try:
        import cupy as cp
        cp.cuda.Device(0).use()
        mempool = cp.get_default_memory_pool()
        props = cp.cuda.runtime.getDeviceProperties(0)
        free, total = cp.cuda.Device(0).mem_info
        
        gpu_name = props['name'].decode()
        vram_total = total / (1024**3)
        vram_free = free / (1024**3)
        
        print(f"  GPU: {gpu_name}")
        print(f"  VRAM: {vram_free:.1f} / {vram_total:.1f} GB libre")
        
        return cp, mempool
    except Exception as e:
        print(f"  ERROR: GPU no disponible: {e}")
        print(f"  Este script requiere NVIDIA GPU con CUDA + CuPy")
        sys.exit(1)

# ============================================================================
# MATRICES DE PAULI (GPU)
# ============================================================================

def get_paulis(cp):
    """Retorna matrices de Pauli en GPU."""
    I2 = cp.eye(2, dtype=cp.complex128)
    sx = cp.array([[0, 1], [1, 0]], dtype=cp.complex128)
    sy = cp.array([[0, -1j], [1j, 0]], dtype=cp.complex128)
    sz = cp.array([[1, 0], [0, -1]], dtype=cp.complex128)
    return I2, sx, sy, sz

# ============================================================================
# HAMILTONIANOS
# ============================================================================

def build_ising(n_qubits, J, h, cp):
    """Hamiltoniano Ising 1D con campo transversal."""
    I2, sx, sy, sz = get_paulis(cp)
    dim = 2**n_qubits
    H = cp.zeros((dim, dim), dtype=cp.complex128)
    
    for i in range(n_qubits - 1):
        ops = [I2] * n_qubits
        ops[i] = sz; ops[i+1] = sz
        term = ops[0]
        for op in ops[1:]:
            term = cp.kron(term, op)
        H -= J * term
        del term
    
    for i in range(n_qubits):
        ops = [I2] * n_qubits
        ops[i] = sx
        term = ops[0]
        for op in ops[1:]:
            term = cp.kron(term, op)
        H -= h * term
        del term
    
    return H

def build_interaction(n_S, n_E, coupling, g, cp, seed=0):
    """
    Hamiltoniano de interacción sistema-ambiente.
    
    'local':  σ^S_último ⊗ σ^E_primero (XX + ZZ)
    'random': Σ_{i∈S, j∈E} J_ij σ_i ⊗ σ_j (scrambling)
    """
    I2, sx, sy, sz = get_paulis(cp)
    d_S = 2**n_S
    d_E = 2**n_E
    d_total = d_S * d_E
    H_SE = cp.zeros((d_total, d_total), dtype=cp.complex128)
    
    if coupling == 'local':
        for pauli_S, pauli_E in [(sx, sx), (sz, sz)]:
            ops_S = [I2]*n_S; ops_S[-1] = pauli_S
            ops_E = [I2]*n_E; ops_E[0] = pauli_E
            
            term_S = ops_S[0]
            for op in ops_S[1:]: term_S = cp.kron(term_S, op)
            term_E = ops_E[0]
            for op in ops_E[1:]: term_E = cp.kron(term_E, op)
            
            H_SE -= (g / 2) * cp.kron(term_S, term_E)
            del term_S, term_E
            
    elif coupling == 'random':
        rng = np.random.RandomState(seed)
        n_terms = 0
        
        for pauli_S, pauli_E in [(sx, sx), (sy, sy), (sz, sz)]:
            for i in range(n_S):
                for j in range(n_E):
                    ops_S = [I2]*n_S; ops_S[i] = pauli_S
                    ops_E = [I2]*n_E; ops_E[j] = pauli_E
                    
                    term_S = ops_S[0]
                    for op in ops_S[1:]: term_S = cp.kron(term_S, op)
                    term_E = ops_E[0]
                    for op in ops_E[1:]: term_E = cp.kron(term_E, op)
                    
                    J_ij = rng.randn()
                    H_SE -= J_ij * cp.kron(term_S, term_E)
                    n_terms += 1
                    
                    del term_S, term_E
        
        # Normalizar
        H_SE *= g / np.sqrt(n_terms)
    
    return H_SE

def build_full_hamiltonian(n_S, n_E, coupling, g, cp, seed=0):
    """Construye H_total = H_S ⊗ I_E + I_S ⊗ H_E + H_SE."""
    d_S = 2**n_S
    d_E = 2**n_E
    
    H_S = build_ising(n_S, J=1.0, h=0.5, cp=cp)
    H_E = build_ising(n_E, J=0.7, h=0.3, cp=cp)
    H_SE = build_interaction(n_S, n_E, coupling, g, cp, seed)
    
    I_S = cp.eye(d_S, dtype=cp.complex128)
    I_E = cp.eye(d_E, dtype=cp.complex128)
    
    H_total = cp.kron(H_S, I_E) + cp.kron(I_S, H_E) + H_SE
    
    del H_S, H_E, H_SE, I_S, I_E
    return H_total

# ============================================================================
# CANAL FÍSICO: MÉTODO VECTORIAL EFICIENTE
# ============================================================================

def physical_channel_output(U, psi_S, d_S, d_E, cp):
    """
    Calcula ρ_out = Tr_E[U(|ψ_S⟩⟨ψ_S| ⊗ I/d_E)U†]
    
    Método vectorial: O(d_E) mat-vec products, sin formar ρ_total.
    Memoria: O(d_S² + d_total) en lugar de O(d_total²).
    """
    d_total = d_S * d_E
    rho_out = cp.zeros((d_S, d_S), dtype=cp.complex128)
    
    for k in range(d_E):
        # |ψ_S⟩ ⊗ |k⟩
        psi_total = cp.zeros(d_total, dtype=cp.complex128)
        for s in range(d_S):
            psi_total[s * d_E + k] = psi_S[s]
        
        # U|ψ_S, k⟩
        phi_k = U @ psi_total
        
        # Reshape a (d_S, d_E) y acumular Tr_E
        M_k = phi_k.reshape(d_S, d_E)
        rho_out += M_k @ M_k.conj().T
        
        del psi_total, phi_k, M_k
    
    rho_out /= d_E
    return rho_out

def physical_channel_output_fast(U, psi_S, d_S, d_E, cp):
    """
    Versión batch: construye la matriz de vectores de una vez.
    Más rápido pero usa más memoria.
    Usar para d_E ≤ 256.
    """
    d_total = d_S * d_E
    
    # Construir todas las entradas batch: Ψ[:, k] = |ψ_S⟩ ⊗ |k⟩
    # En lugar de loop, usar indexing inteligente
    Psi_in = cp.zeros((d_total, d_E), dtype=cp.complex128)
    for k in range(d_E):
        for s in range(d_S):
            Psi_in[s * d_E + k, k] = psi_S[s]
    
    # Batch U @ Ψ
    Phi = U @ Psi_in  # (d_total, d_E)
    
    # Reshape cada columna a (d_S, d_E) y hacer partial trace
    # Phi[:, k] reshaped → M_k de (d_S, d_E)
    # Σ_k M_k @ M_k† = reshape(Phi, (d_S, d_E, d_E)) operations
    
    # Forma eficiente: reorganizar Phi
    # Phi[s*d_E + e, k] → tensor[s, e, k]
    Phi_tensor = Phi.reshape(d_S, d_E, d_E)
    
    # rho_out[a, b] = (1/d_E) Σ_k Σ_e Phi_tensor[a, e, k] * conj(Phi_tensor[b, e, k])
    # = (1/d_E) Σ_k (M_k @ M_k†)[a,b]
    # = (1/d_E) * einsum('aek,bek->ab', Phi_tensor, conj(Phi_tensor))
    
    rho_out = cp.einsum('aek,bek->ab', Phi_tensor, cp.conj(Phi_tensor)) / d_E
    
    del Psi_in, Phi, Phi_tensor
    return rho_out

# ============================================================================
# MÉTRICAS
# ============================================================================

def trace_norm_gpu(A, cp):
    """||A||₁ = Σ σ_i."""
    svd_vals = cp.linalg.svd(A, compute_uv=False)
    return float(cp.sum(svd_vals))

def measure_C_and_distance(U, d_S, d_E, gamma, n_samples, cp, mempool):
    """
    Para n_samples estados Haar-random:
    1. Mide C = ⟨ψ|ρ_phys|ψ⟩
    2. Mide D = 0.5 ||ρ_phys - ρ_depol(γ)||₁
    
    Retorna C_mean, C_std, D_mean, D_std
    """
    d_total = d_S * d_E
    use_fast = (d_E <= 256 and d_total * d_E * 16 < 2e9)  # <2GB para batch
    
    I_dS = cp.eye(d_S, dtype=cp.complex128) / d_S
    
    C_vals = []
    D_vals = []
    
    for i in range(n_samples):
        # Estado Haar-random
        psi_S = cp.random.standard_normal(d_S) + 1j * cp.random.standard_normal(d_S)
        psi_S = psi_S.astype(cp.complex128)
        psi_S /= cp.linalg.norm(psi_S)
        
        # Canal físico
        if use_fast:
            rho_phys = physical_channel_output_fast(U, psi_S, d_S, d_E, cp)
        else:
            rho_phys = physical_channel_output(U, psi_S, d_S, d_E, cp)
        
        # Fidelidad
        C = float(cp.real(psi_S.conj() @ rho_phys @ psi_S))
        C_vals.append(C)
        
        # Canal de despolarización con el γ dado
        rho_S = cp.outer(psi_S, cp.conj(psi_S))
        rho_depol = (1 - gamma) * rho_S + gamma * I_dS
        
        # Distancia traza
        diff = rho_phys - rho_depol
        D = 0.5 * trace_norm_gpu(diff, cp)
        D_vals.append(D)
        
        del psi_S, rho_phys, rho_S, rho_depol, diff
        
        # Liberar memoria periódicamente
        if (i + 1) % 50 == 0:
            mempool.free_all_blocks()
    
    mempool.free_all_blocks()
    
    C_arr = np.array(C_vals)
    D_arr = np.array(D_vals)
    
    return {
        'C_mean': float(np.mean(C_arr)),
        'C_std': float(np.std(C_arr)),
        'D_mean': float(np.mean(D_arr)),
        'D_std': float(np.std(D_arr)),
        'D_max': float(np.max(D_arr)),
    }

def compute_gamma_eff(C_mean, d_S):
    """γ_eff = (1 - C) / (1 - 1/d)"""
    return (1 - C_mean) / (1 - 1/d_S)

# ============================================================================
# CHOI MATRIX (para sistemas pequeños)
# ============================================================================

def compute_choi_physical(U, d_S, d_E, cp, mempool):
    """
    Matriz de Choi del canal físico.
    Solo factible para d_S pequeño (2-4 qubits de sistema).
    """
    d_S2 = d_S * d_S
    choi = cp.zeros((d_S2, d_S2), dtype=cp.complex128)
    
    for i in range(d_S):
        # Preparar base |i⟩
        psi_i = cp.zeros(d_S, dtype=cp.complex128)
        psi_i[i] = 1.0
        
        for j in range(d_S):
            psi_j = cp.zeros(d_S, dtype=cp.complex128)
            psi_j[j] = 1.0
            
            # Necesitamos ℰ(|i⟩⟨j|)
            # = (1/d_E) Σ_k Tr_E[ U(|i⟩⊗|k⟩)(⟨j|⊗⟨k|)U† ]
            rho_out = cp.zeros((d_S, d_S), dtype=cp.complex128)
            
            for k in range(d_E):
                # |i⟩ ⊗ |k⟩
                psi_ik = cp.zeros(d_S * d_E, dtype=cp.complex128)
                for s_idx in range(d_S):
                    if s_idx == i:
                        psi_ik[s_idx * d_E + k] = 1.0
                
                # |j⟩ ⊗ |k⟩
                psi_jk = cp.zeros(d_S * d_E, dtype=cp.complex128)
                for s_idx in range(d_S):
                    if s_idx == j:
                        psi_jk[s_idx * d_E + k] = 1.0
                
                phi_ik = U @ psi_ik
                phi_jk = U @ psi_jk
                
                # Tr_E[|φ_ik⟩⟨φ_jk|]
                M_ik = phi_ik.reshape(d_S, d_E)
                M_jk = phi_jk.reshape(d_S, d_E)
                
                rho_out += M_ik @ M_jk.conj().T
                
                del psi_ik, psi_jk, phi_ik, phi_jk, M_ik, M_jk
            
            rho_out /= d_E
            
            # Llenar Choi: J[i*d+a, j*d+b] = ℰ(|i⟩⟨j|)[a,b]
            for a in range(d_S):
                for b in range(d_S):
                    choi[i * d_S + a, j * d_S + b] = rho_out[a, b]
            
            del rho_out
    
    mempool.free_all_blocks()
    return choi

def compute_choi_depol(d_S, gamma, cp):
    """Choi del canal de despolarización."""
    d_S2 = d_S * d_S
    
    # |Φ⁺⟩ (sin normalizar)
    phi_plus = cp.zeros(d_S2, dtype=cp.complex128)
    for i in range(d_S):
        phi_plus[i * d_S + i] = 1.0
    
    rho_phi = cp.outer(phi_plus, cp.conj(phi_plus))  # trace = d_S
    I_d2 = cp.eye(d_S2, dtype=cp.complex128)
    
    choi = (1 - gamma) * rho_phi / d_S + gamma * I_d2 / d_S
    
    del phi_plus, rho_phi, I_d2
    return choi

# ============================================================================
# TIMER Y CHECKPOINT
# ============================================================================

class Timer:
    def __init__(self, max_seconds):
        self.start = time.time()
        self.max_seconds = max_seconds
        self.last_checkpoint = self.start
    
    def elapsed(self):
        return time.time() - self.start
    
    def remaining(self):
        return max(0, self.max_seconds - self.elapsed())
    
    def should_stop(self):
        return self.elapsed() >= self.max_seconds
    
    def should_checkpoint(self):
        now = time.time()
        if now - self.last_checkpoint >= CHECKPOINT_INTERVAL:
            self.last_checkpoint = now
            return True
        return False
    
    def eta_str(self):
        r = self.remaining()
        return str(timedelta(seconds=int(r)))
    
    def elapsed_str(self):
        return str(timedelta(seconds=int(self.elapsed())))

def save_checkpoint(results, phase, filename=None):
    """Guarda resultados parciales."""
    if filename is None:
        filename = f"{OUTPUT_FILE}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    output = {
        'fecha': datetime.now().isoformat(),
        'experiment': 'JADE v8.2 — El Puente (4070 Edition)',
        'question': '¿Es el universo un canal de despolarización global?',
        'phase_completed': phase,
        'gamma_jade': 1 - 1/np.e,
        'results': results,
    }
    
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    return filename

# ============================================================================
# FASE 1: ESCALAMIENTO
# ============================================================================

def run_phase1(cp, mempool, timer, results):
    """
    Escalamiento: D vs n_E para n_S = {2, 3, 4}.
    Coupling: local vs random.
    """
    print(f"\n{'='*70}")
    print(f"  FASE 1: ESCALAMIENTO — D vs n_E")
    print(f"  ¿Converge el canal físico al de despolarización?")
    print(f"{'='*70}")
    
    from cupyx.scipy.linalg import expm
    
    gamma_jade = 1 - 1/np.e
    results['phase1'] = []
    
    total_configs = sum(
        max_nE * len(COUPLINGS) * len(TIMES) * 
        (N_REALIZATIONS if 'random' in COUPLINGS else 1)
        for _, max_nE in SCALING_CONFIGS
    )
    config_count = 0
    
    for n_S, max_nE in SCALING_CONFIGS:
        d_S = 2**n_S
        
        print(f"\n  ── n_S = {n_S} (d_S = {d_S}) ──")
        print(f"  {'n_E':>3} {'d_E':>5} {'coupling':>8} {'t':>5} "
              f"{'C_phys':>8} {'γ_eff':>8} {'D_best':>8} {'D_jade':>8} "
              f"{'tiempo':>7}")
        print(f"  {'-'*68}")
        
        for n_E in range(1, max_nE + 1):
            d_E = 2**n_E
            d_total = d_S * d_E
            
            if d_total > MAX_D_TOTAL_EXPM:
                print(f"  {n_E:3d} {d_E:5d} SALTANDO (d_total={d_total} > {MAX_D_TOTAL_EXPM})")
                continue
            
            if timer.should_stop():
                print(f"\n  ⚠ TIEMPO AGOTADO — saltando resto de Fase 1")
                return
            
            for coupling in COUPLINGS:
                n_real = N_REALIZATIONS if coupling == 'random' else 1
                
                for real_idx in range(n_real):
                    seed = 1000 * n_S + 100 * n_E + real_idx
                    
                    # Construir H_total
                    try:
                        H_total = build_full_hamiltonian(
                            n_S, n_E, coupling, g=0.5, cp=cp, seed=seed
                        )
                    except Exception as e:
                        print(f"  ERROR construyendo H: {e}")
                        continue
                    
                    for t in TIMES:
                        if timer.should_stop():
                            del H_total; mempool.free_all_blocks()
                            return
                        
                        t_start = time.time()
                        config_count += 1
                        
                        try:
                            # Evolución
                            U = expm(-1j * H_total * t)
                            
                            # Medir C con γ_eff (mejor ajuste)
                            quick = measure_C_and_distance(
                                U, d_S, d_E, gamma_jade, 
                                min(50, N_SAMPLES), cp, mempool
                            )
                            gamma_eff = compute_gamma_eff(quick['C_mean'], d_S)
                            
                            # Medir distancia con γ_eff (best fit)
                            result_best = measure_C_and_distance(
                                U, d_S, d_E, gamma_eff,
                                N_SAMPLES, cp, mempool
                            )
                            
                            # Medir distancia con γ_jade
                            result_jade = measure_C_and_distance(
                                U, d_S, d_E, gamma_jade,
                                N_SAMPLES, cp, mempool
                            )
                            
                            elapsed = time.time() - t_start
                            
                            entry = {
                                'n_S': n_S, 'n_E': n_E,
                                'd_S': d_S, 'd_E': d_E, 'd_total': d_total,
                                'coupling': coupling, 'realization': real_idx,
                                't': t, 'seed': seed,
                                'C_mean': result_best['C_mean'],
                                'C_std': result_best['C_std'],
                                'gamma_eff': gamma_eff,
                                'D_best': result_best['D_mean'],
                                'D_best_std': result_best['D_std'],
                                'D_jade': result_jade['D_mean'],
                                'D_jade_std': result_jade['D_std'],
                                'elapsed': elapsed,
                            }
                            results['phase1'].append(entry)
                            
                            # Solo imprimir primera realización y t=10
                            if real_idx == 0 and t == 10.0:
                                print(f"  {n_E:3d} {d_E:5d} {coupling:>8s} {t:5.1f} "
                                      f"{result_best['C_mean']:8.4f} {gamma_eff:8.4f} "
                                      f"{result_best['D_mean']:8.4f} {result_jade['D_mean']:8.4f} "
                                      f"{elapsed:6.1f}s")
                            
                            del U
                            
                        except Exception as e:
                            print(f"  ERROR en n_E={n_E}, {coupling}, t={t}: {e}")
                        
                        mempool.free_all_blocks()
                    
                    del H_total
                    mempool.free_all_blocks()
                    
                    # Checkpoint
                    if timer.should_checkpoint():
                        fn = save_checkpoint(results, 'phase1_partial')
                        print(f"  💾 Checkpoint: {fn} [{timer.elapsed_str()} / {timer.eta_str()} restante]")

# ============================================================================
# FASE 2: CHOI EXACTO
# ============================================================================

def run_phase2(cp, mempool, timer, results):
    """Norma de Choi exacta para sistemas pequeños."""
    print(f"\n{'='*70}")
    print(f"  FASE 2: NORMA DE CHOI EXACTA")
    print(f"  ||J_phys - J_depol||₁ — Gold standard")
    print(f"{'='*70}")
    
    if timer.should_stop():
        print("  ⚠ Sin tiempo para Fase 2")
        return
    
    from cupyx.scipy.linalg import expm
    
    gamma_jade = 1 - 1/np.e
    results['phase2'] = []
    
    # Solo sistemas pequeños: Choi es d_S² × d_S²
    # n_S=2: Choi 16×16, n_S=3: Choi 64×64 → ambos triviales
    configs = [
        (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6),
        (3, 1), (3, 2), (3, 3), (3, 4), (3, 5),
    ]
    
    t_choi = 10.0
    
    print(f"  {'n_S':>3} {'n_E':>3} {'coupling':>8} {'γ_eff':>8} {'||ΔJ||₁ best':>14} {'||ΔJ||₁ jade':>14}")
    print(f"  {'-'*56}")
    
    for n_S, n_E in configs:
        d_S = 2**n_S
        d_E = 2**n_E
        d_total = d_S * d_E
        
        if d_total > 1024 or timer.should_stop():
            continue
        
        for coupling in COUPLINGS:
            n_real = 3 if coupling == 'random' else 1
            norms_best = []
            norms_jade = []
            gammas = []
            
            for r in range(n_real):
                seed = 5000 + 100 * n_S + 10 * n_E + r
                
                H_total = build_full_hamiltonian(n_S, n_E, coupling, 0.5, cp, seed)
                U = expm(-1j * H_total * t_choi)
                
                # Medir γ_eff
                quick = measure_C_and_distance(
                    U, d_S, d_E, gamma_jade, 100, cp, mempool
                )
                gamma_eff = compute_gamma_eff(quick['C_mean'], d_S)
                gammas.append(gamma_eff)
                
                # Choi matrices
                J_phys = compute_choi_physical(U, d_S, d_E, cp, mempool)
                J_depol_best = compute_choi_depol(d_S, gamma_eff, cp)
                J_depol_jade = compute_choi_depol(d_S, gamma_jade, cp)
                
                tn_best = trace_norm_gpu(J_phys - J_depol_best, cp)
                tn_jade = trace_norm_gpu(J_phys - J_depol_jade, cp)
                
                norms_best.append(tn_best)
                norms_jade.append(tn_jade)
                
                del H_total, U, J_phys, J_depol_best, J_depol_jade
                mempool.free_all_blocks()
            
            entry = {
                'n_S': n_S, 'n_E': n_E, 'coupling': coupling,
                't': t_choi,
                'gamma_eff_mean': float(np.mean(gammas)),
                'choi_norm_best': float(np.mean(norms_best)),
                'choi_norm_best_std': float(np.std(norms_best)),
                'choi_norm_jade': float(np.mean(norms_jade)),
                'choi_norm_jade_std': float(np.std(norms_jade)),
            }
            results['phase2'].append(entry)
            
            print(f"  {n_S:3d} {n_E:3d} {coupling:>8s} {np.mean(gammas):8.4f} "
                  f"{np.mean(norms_best):14.6f} {np.mean(norms_jade):14.6f}")

# ============================================================================
# FASE 3: CONVERGENCIA TEMPORAL
# ============================================================================

def run_phase3(cp, mempool, timer, results):
    """D vs t para una configuración fija."""
    print(f"\n{'='*70}")
    print(f"  FASE 3: CONVERGENCIA TEMPORAL")
    print(f"  n_S=2, n_E=5, coupling=random")
    print(f"{'='*70}")
    
    if timer.should_stop():
        print("  ⚠ Sin tiempo para Fase 3")
        return
    
    from cupyx.scipy.linalg import expm
    
    gamma_jade = 1 - 1/np.e
    results['phase3'] = []
    
    n_S, n_E = 2, 5
    d_S, d_E = 2**n_S, 2**n_E
    
    H_total = build_full_hamiltonian(n_S, n_E, 'random', 0.5, cp, seed=42)
    
    print(f"  {'t':>6} {'C_phys':>8} {'γ_eff':>8} {'D_best':>8} {'D_jade':>8}")
    print(f"  {'-'*44}")
    
    for t in TEMPORAL_TIMES:
        if timer.should_stop():
            break
        
        U = expm(-1j * H_total * t)
        
        quick = measure_C_and_distance(U, d_S, d_E, gamma_jade, 50, cp, mempool)
        gamma_eff = compute_gamma_eff(quick['C_mean'], d_S)
        
        result_best = measure_C_and_distance(U, d_S, d_E, gamma_eff, N_SAMPLES, cp, mempool)
        result_jade = measure_C_and_distance(U, d_S, d_E, gamma_jade, N_SAMPLES, cp, mempool)
        
        entry = {
            't': t,
            'C_mean': result_best['C_mean'],
            'gamma_eff': gamma_eff,
            'D_best': result_best['D_mean'],
            'D_best_std': result_best['D_std'],
            'D_jade': result_jade['D_mean'],
        }
        results['phase3'].append(entry)
        
        print(f"  {t:6.1f} {result_best['C_mean']:8.4f} {gamma_eff:8.4f} "
              f"{result_best['D_mean']:8.4f} {result_jade['D_mean']:8.4f}")
        
        del U
        mempool.free_all_blocks()
    
    del H_total
    mempool.free_all_blocks()

# ============================================================================
# RESUMEN EJECUTIVO
# ============================================================================

def print_summary(results):
    """Análisis final de los resultados."""
    print(f"\n{'='*70}")
    print(f"  RESUMEN EJECUTIVO — JADE v8.2")
    print(f"{'='*70}")
    
    # Fase 1: Scaling
    if 'phase1' in results and results['phase1']:
        print(f"\n  ── SCALING (Fase 1) ──")
        
        for n_S in [2, 3, 4]:
            d_S = 2**n_S
            print(f"\n  n_S = {n_S} (d_S = {d_S}):")
            
            for coupling in ['local', 'random']:
                # Filtrar t=10, promediar sobre realizaciones
                entries = [e for e in results['phase1'] 
                          if e['n_S'] == n_S and e['coupling'] == coupling and e['t'] == 10.0]
                
                if not entries:
                    continue
                
                # Agrupar por n_E
                by_nE = {}
                for e in entries:
                    nE = e['n_E']
                    if nE not in by_nE:
                        by_nE[nE] = []
                    by_nE[nE].append(e['D_best'])
                
                print(f"    {coupling:>8s}: ", end="")
                vals = []
                for nE in sorted(by_nE.keys()):
                    d_mean = np.mean(by_nE[nE])
                    vals.append((nE, d_mean))
                    print(f"E{nE}={d_mean:.3f} ", end="")
                print()
                
                if len(vals) >= 2:
                    reduction = (1 - vals[-1][1] / vals[0][1]) * 100
                    trend = "↘ CONVERGE" if reduction > 30 else "→ ESTABLE" if reduction > -10 else "↗ DIVERGE"
                    print(f"             Reducción: {reduction:.1f}% {trend}")
    
    # Fase 2: Choi
    if 'phase2' in results and results['phase2']:
        print(f"\n  ── CHOI EXACTO (Fase 2) ──")
        for coupling in ['local', 'random']:
            entries = [e for e in results['phase2'] if e['coupling'] == coupling]
            if entries:
                print(f"    {coupling}:")
                for e in entries:
                    print(f"      S={e['n_S']}, E={e['n_E']} | ||ΔJ||₁ = {e['choi_norm_best']:.4f}")
    
    # Conclusión
    print(f"\n  ── CONCLUSIÓN ──")
    
    # Verificar si hay convergencia en random
    random_entries = [e for e in results.get('phase1', []) 
                     if e['coupling'] == 'random' and e['t'] == 10.0 and e['n_S'] == 2]
    
    if random_entries:
        by_nE = {}
        for e in random_entries:
            nE = e['n_E']
            if nE not in by_nE:
                by_nE[nE] = []
            by_nE[nE].append(e['D_best'])
        
        nE_sorted = sorted(by_nE.keys())
        if len(nE_sorted) >= 2:
            d_first = np.mean(by_nE[nE_sorted[0]])
            d_last = np.mean(by_nE[nE_sorted[-1]])
            reduction = (1 - d_last / d_first) * 100
            
            if reduction > 50:
                print(f"  ★ CONVERGENCIA FUERTE: D se reduce {reduction:.0f}% con scrambling")
                print(f"    El canal físico CONVERGE al de despolarización.")
                print(f"    → C + γ = 1 es el límite termodinámico correcto.")
            elif reduction > 20:
                print(f"  ✓ CONVERGENCIA MODERADA: D se reduce {reduction:.0f}%")
                print(f"    Tendencia clara pero necesita más qubits de ambiente.")
            else:
                print(f"  ⚠ CONVERGENCIA DÉBIL: D se reduce solo {reduction:.0f}%")
                print(f"    Resultados no concluyentes a esta escala.")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print(f"\n{'='*70}")
    print(f"  JADE v8.2 — EL PUENTE")
    print(f"  ¿Es el universo un canal de despolarización global?")
    print(f"")
    print(f"  Agujero Negro = Fast Scrambler → Despolarización → C + γ = 1")
    print(f"")
    print(f"  Tiempo máximo: {MAX_HOURS} horas")
    print(f"  Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    
    # GPU
    cp, mempool = init_gpu()
    
    # Timer
    timer = Timer(MAX_TIME)
    
    # Resultados
    results = {
        'config': {
            'max_hours': MAX_HOURS,
            'n_samples': N_SAMPLES,
            'scaling_configs': SCALING_CONFIGS,
            'couplings': COUPLINGS,
            'times': TIMES,
            'n_realizations': N_REALIZATIONS,
        }
    }
    
    # Fase 1
    run_phase1(cp, mempool, timer, results)
    fn = save_checkpoint(results, 'phase1_complete')
    print(f"\n  💾 Fase 1 completa: {fn} [{timer.elapsed_str()}]")
    
    # Fase 2
    run_phase2(cp, mempool, timer, results)
    fn = save_checkpoint(results, 'phase2_complete')
    print(f"\n  💾 Fase 2 completa: {fn} [{timer.elapsed_str()}]")
    
    # Fase 3
    run_phase3(cp, mempool, timer, results)
    fn = save_checkpoint(results, 'phase3_complete')
    print(f"\n  💾 Fase 3 completa: {fn} [{timer.elapsed_str()}]")
    
    # Resumen
    print_summary(results)
    
    # Guardar final
    final_fn = save_checkpoint(results, 'COMPLETE')
    
    print(f"\n{'='*70}")
    print(f"  COMPLETADO en {timer.elapsed_str()}")
    print(f"  Resultados: {final_fn}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
