#!/usr/bin/env python3
"""
JADE — THE EDGE TEST v4.2 (PTM + GPU Trotter)
==============================================
Target: NVIDIA B200 (192 GB HBM3e, Blackwell)

Filosofía: NO reescribir el motor. Mismo Trotter probado en H200 y
RTX 6000 Ada. Solo extender el rango de n_E aprovechando 192 GB.
Resultados DIRECTAMENTE comparables con corridas anteriores.

Zona de confort B200:
  28q (n_E=27) → 268M dim | ~110 GB / 192 GB (59%) | 75 GB margen

Sistema: n_S = 1 (Bloch sphere → PTM 3×3)
Ambiente: n_E = [8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 27]
Tiempos: t = [1, 2, 5, 10, 20, 50] × τ_scramble por n_E

Métrica D: distancia Frobenius de PTM a mejor canal despolarizante
  D = ||M - f·I||_F + ||t_vec||  (anisotropía + no-unitalidad)
  Si D → 0: canal físico → despolarización (JADE confirmado)

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

# ── GPU ──
try:
    import cupy as cp
    cp.cuda.Device(0).use()
    props = cp.cuda.runtime.getDeviceProperties(0)
    GPU_NAME = props['name'].decode()
    GPU_MEM = props['totalGlobalMem'] / (1024**3)
    USE_GPU = True
    print(f"  GPU: {GPU_NAME} ({GPU_MEM:.0f} GB)")
except Exception as e:
    print(f"  GPU no disponible ({e}). Usando CPU fallback.")
    USE_GPU = False
    GPU_NAME = "CPU"
    GPU_MEM = 0

if not USE_GPU:
    import scipy.sparse as sps
    from scipy.sparse.linalg import expm_multiply

# ============================================================================
# CONFIGURACIÓN — B200 (192 GB) zona de confort
# ============================================================================

N_S = 1

# Barrido completo hasta 28q (n_E=27)
# De 8 a 18: solapan con H200/RTX6000 → validación cruzada
# De 20 a 27: territorio nuevo exclusivo B200
N_E_VALUES = [8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 27]

# Tiempos de evolución (incluye ×50 para convergencia profunda)
T_MULTIPLIERS = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]

# Hamiltoniano (IDÉNTICO a v4.0 para comparabilidad directa)
J1 = 1.0
J2 = 1.0
H_FIELD = 0.5
LAMBDA = 0.5

# Trotter
TROTTER_STEPS = 80

# ── Samples adaptativos (generosos — hay VRAM de sobra) ──
def get_n_samples(n_E):
    """Más samples que RTX 6000 porque B200 tiene margen."""
    n_total = N_S + n_E
    if n_total <= 19:
        return 20   # Rápido, máxima estadística
    elif n_total <= 23:
        return 15   # Más que RTX 6000 (que usaba 10)
    elif n_total <= 26:
        return 10   # Buen balance
    else:
        return 5    # 27-28q: ~110 GB, conservador pero suficiente

# ── VRAM estimation (pre-stored parities, motor actual) ──
def estimate_vram_gb(n_total):
    """VRAM con motor actual: state + pre-stored parities."""
    dim = 2**n_total
    n_pars = (n_total - 1) + max(0, n_total - 2)  # nn + nnn
    state = dim * 16            # complex128
    parities = n_pars * dim * 8 # float64 arrays
    rx_overhead = dim * 16      # copies en _apply_rx
    return (state + parities + rx_overhead) / (1024**3)

def get_max_qubits():
    if not USE_GPU:
        return 16
    usable = GPU_MEM - 7.0  # Margen generoso
    for n in range(32, 10, -1):
        if estimate_vram_gb(n) < usable:
            return n
    return 16


# ============================================================================
# ESTADOS BASE PARA PTM (Tomografía de Proceso)
# ============================================================================

BASIS_STATES = [
    np.array([1, 0], dtype=complex),              # |0⟩
    np.array([0, 1], dtype=complex),              # |1⟩
    np.array([1, 1], dtype=complex) / np.sqrt(2), # |+⟩
    np.array([1, 1j], dtype=complex) / np.sqrt(2) # |+i⟩
]
BASIS_LABELS = ["|0⟩", "|1⟩", "|+⟩", "|+i⟩"]


# ============================================================================
# TROTTER GPU ENGINE (idéntico a v4.0 — probado en H200/RTX6000)
# ============================================================================

class TrotterGPU:
    """Evolución Trotter en GPU para H = -J1·ZZ_nn - h·X - λ·J2·ZZ_nnn
    
    NOTA: Motor idéntico al usado en H200. No se modifica para B200.
    Solo se extiende el rango de n_E que alimenta.
    int64 en indices para soportar >2^31 dimensiones.
    """
    
    def __init__(self, n_qubits, J1, h, J2, lam):
        self.n = n_qubits
        self.dim = 2**n_qubits
        
        xp = cp if USE_GPU else np
        # int64 necesario para n >= 32 (2^31 overflow en int32)
        indices = xp.arange(self.dim, dtype=xp.int64)
        
        # Paridades ZZ nearest-neighbor
        self.nn_pars = []
        for i in range(n_qubits - 1):
            bi = (indices >> (n_qubits - 1 - i)) & 1
            bj = (indices >> (n_qubits - 1 - (i + 1))) & 1
            self.nn_pars.append((bi ^ bj).astype(xp.float64))
        
        # Paridades ZZ next-nearest-neighbor
        self.nnn_pars = []
        if lam > 0:
            for i in range(n_qubits - 2):
                bi = (indices >> (n_qubits - 1 - i)) & 1
                bj = (indices >> (n_qubits - 1 - (i + 2))) & 1
                self.nnn_pars.append((bi ^ bj).astype(xp.float64))
        
        self.J1 = J1
        self.h = h
        self.J2 = J2
        self.lam = lam
        
        # Liberar array de indices (ya no se necesita)
        del indices
        if USE_GPU:
            cp.get_default_memory_pool().free_all_blocks()
    
    def _apply_zz(self, psi, parities, angle):
        xp = cp if USE_GPU else np
        for par in parities:
            psi *= xp.exp(1j * angle * (1.0 - 2.0 * par))
        return psi
    
    def _apply_rx(self, psi, angle):
        xp = cp if USE_GPU else np
        c = float(np.cos(angle))
        s = float(np.sin(angle))
        shape = [2] * self.n
        psi_t = psi.reshape(shape)
        
        for qi in range(self.n):
            idx0 = [slice(None)] * self.n
            idx1 = [slice(None)] * self.n
            idx0[qi] = 0
            idx1[qi] = 1
            
            p0 = psi_t[tuple(idx0)].copy()
            p1 = psi_t[tuple(idx1)].copy()
            
            psi_t[tuple(idx0)] = c * p0 + 1j * s * p1
            psi_t[tuple(idx1)] = 1j * s * p0 + c * p1
        
        return psi_t.reshape(self.dim)
    
    def evolve(self, psi, t_total, n_steps):
        dt = t_total / n_steps
        for _ in range(n_steps):
            psi = self._apply_zz(psi, self.nn_pars, self.J1 * dt)
            psi = self._apply_rx(psi, self.h * dt)
            if self.lam > 0 and self.nnn_pars:
                psi = self._apply_zz(psi, self.nnn_pars, self.lam * self.J2 * dt)
        return psi
    
    def cleanup(self):
        """Liberar memoria GPU explícitamente."""
        for attr in ['nn_pars', 'nnn_pars']:
            if hasattr(self, attr):
                lst = getattr(self, attr)
                for item in lst:
                    del item
                lst.clear()
        if USE_GPU:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()


# ============================================================================
# CPU FALLBACK (scipy sparse)
# ============================================================================

def evolve_cpu_krylov(n_qubits, J1, h, J2, lam, psi, t):
    """Fallback CPU usando expm_multiply."""
    sx = sps.csr_matrix([[0, 1], [1, 0]])
    sz = sps.csr_matrix([[1, 0], [0, -1]])
    id2 = sps.eye(2, format='csr')
    
    dim = 2**n_qubits
    H = sps.csr_matrix((dim, dim), dtype=complex)
    
    for i in range(n_qubits):
        ops = [id2]*n_qubits; ops[i] = sx
        term = ops[0]
        for op in ops[1:]: term = sps.kron(term, op)
        H -= h * term
    
    for i in range(n_qubits - 1):
        ops = [id2]*n_qubits; ops[i] = sz; ops[i+1] = sz
        term = ops[0]
        for op in ops[1:]: term = sps.kron(term, op)
        H -= J1 * term
    
    if lam > 0:
        for i in range(n_qubits - 2):
            ops = [id2]*n_qubits; ops[i] = sz; ops[i+2] = sz
            term = ops[0]
            for op in ops[1:]: term = sps.kron(term, op)
            H -= lam * J2 * term
    
    return expm_multiply(-1j * H * t, psi)


# ============================================================================
# PTM: RECONSTRUCCIÓN DE CANAL Y DISTANCIA
# ============================================================================

def bloch_vector(rho):
    """ρ (2×2) → (rx, ry, rz)"""
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    return np.real(np.array([np.trace(rho @ X), np.trace(rho @ Y), np.trace(rho @ Z)]))


def compute_ptm_distance(bloch_out):
    """
    Dado los 4 vectores de Bloch de salida,
    reconstruye la PTM y calcula distancia al canal despolarizante.
    """
    v_0, v_1, v_plus, v_plusi = bloch_out
    
    t_vec = (v_0 + v_1) / 2
    m_x = v_plus - t_vec
    m_y = v_plusi - t_vec
    m_z = (v_0 - v_1) / 2
    
    M = np.column_stack((m_x, m_y, m_z))
    f_fit = np.trace(M) / 3.0
    anisotropy = np.linalg.norm(M - f_fit * np.eye(3))
    non_unitality = np.linalg.norm(t_vec)
    D_total = np.sqrt(anisotropy**2 + non_unitality**2)
    
    return {
        'D': float(D_total),
        'f_fit': float(f_fit),
        'anisotropy': float(anisotropy),
        'non_unitality': float(non_unitality),
        'M_diagonal': [float(M[0,0]), float(M[1,1]), float(M[2,2])],
        'M_offdiag_norm': float(np.linalg.norm(M - np.diag(np.diag(M))))
    }


# ============================================================================
# EXPERIMENTO CENTRAL
# ============================================================================

def run_single_point(n_E, t_evolve, env_seed):
    """Un punto: (n_E, t, seed) → distancia PTM al despolarizante."""
    xp = cp if USE_GPU else np
    
    n_total = N_S + n_E
    dim_E = 2**n_E
    dim_total = 2**n_total
    
    if USE_GPU:
        evolver = TrotterGPU(n_total, J1, H_FIELD, J2, LAMBDA)
    
    np.random.seed(env_seed)
    psi_E_np = np.random.randn(dim_E) + 1j * np.random.randn(dim_E)
    psi_E_np /= np.linalg.norm(psi_E_np)
    
    bloch_out = []
    
    for psi_S in BASIS_STATES:
        psi_total_np = np.kron(psi_S, psi_E_np)
        
        if USE_GPU:
            psi_total = cp.asarray(psi_total_np)
            psi_evolved = evolver.evolve(psi_total, t_evolve, TROTTER_STEPS)
            psi_mat = psi_evolved.reshape(2, dim_E)
            rho_S = cp.asnumpy(psi_mat @ psi_mat.conj().T)
            del psi_total, psi_evolved, psi_mat
        else:
            psi_evolved = evolve_cpu_krylov(
                n_total, J1, H_FIELD, J2, LAMBDA, psi_total_np, t_evolve
            )
            psi_mat = psi_evolved.reshape(2, dim_E)
            rho_S = psi_mat @ psi_mat.conj().T
        
        bloch_out.append(bloch_vector(rho_S))
    
    if USE_GPU:
        evolver.cleanup()
        del evolver
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    
    return compute_ptm_distance(bloch_out)


# ============================================================================
# MAIN
# ============================================================================

def main():
    max_q = get_max_qubits()
    
    print(f"\n{'█'*70}")
    print(f"  JADE — THE EDGE TEST v4.2 (PTM + Trotter {'GPU' if USE_GPU else 'CPU'})")
    print(f"  Target: B200 (192 GB) — Zona de Confort")
    print(f"  ¿D → 0 conforme crece el ambiente?")
    print(f"{'█'*70}")
    
    print(f"\n  Engine: {'Trotter GPU (CuPy)' if USE_GPU else 'Krylov CPU (scipy)'}")
    if USE_GPU:
        print(f"  GPU: {GPU_NAME} ({GPU_MEM:.0f} GB)")
    print(f"  n_S = {N_S}, n_E = {N_E_VALUES}")
    print(f"  λ = {LAMBDA} (caótico)")
    print(f"  t_multipliers = {T_MULTIPLIERS}")
    print(f"  Trotter steps = {TROTTER_STEPS}")
    print(f"  Max qubits = {max_q}")
    print(f"  H = -J1·ZZ_nn - h·X - λ·J2·ZZ_nnn")
    
    # Filtrar factibles
    feasible = [nE for nE in N_E_VALUES if (N_S + nE) <= max_q]
    print(f"\n  Factibles: {feasible}")
    
    # Plan de ejecución
    print(f"\n  {'n_E':>4} → {'n_tot':>5} | {'dim':>14} | {'VRAM est':>10} | {'samples':>7}")
    print(f"  {'─'*62}")
    total_evolutions = 0
    for nE in feasible:
        n_tot = N_S + nE
        dim = 2**n_tot
        vram = estimate_vram_gb(n_tot)
        samples = get_n_samples(nE)
        total_evolutions += len(T_MULTIPLIERS) * samples * 4
        print(f"  {nE:4d} → {n_tot:5d} | {dim:14,} | ~{vram:7.1f} GB | {samples:7d}")
    
    print(f"\n  Total evoluciones: {total_evolutions:,}")
    
    # ═══════════════════════════════════════════════════════════════════
    # BARRIDO
    # ═══════════════════════════════════════════════════════════════════
    
    results = {}
    start_total = time.time()
    
    for nE in feasible:
        n_total = N_S + nE
        n_samples = get_n_samples(nE)
        vram_est = estimate_vram_gb(n_total)
        results[nE] = {}
        
        print(f"\n  ╔{'═'*55}╗")
        print(f"  ║  n_E = {nE}  →  {n_total}q  ({2**n_total:,} dim)")
        print(f"  ║  VRAM est: ~{vram_est:.1f} GB | samples: {n_samples}")
        print(f"  ╚{'═'*55}╝")
        
        for t_mult in T_MULTIPLIERS:
            t_evolve = t_mult * n_total
            
            Ds, fs, anis, nunis = [], [], [], []
            t0 = time.time()
            
            for env_idx in range(n_samples):
                seed = 1000 * nE + 100 * int(t_mult) + env_idx
                try:
                    res = run_single_point(nE, t_evolve, seed)
                    Ds.append(res['D'])
                    fs.append(res['f_fit'])
                    anis.append(res['anisotropy'])
                    nunis.append(res['non_unitality'])
                except Exception as e:
                    if 'out of memory' in str(e).lower() or 'OutOfMemory' in type(e).__name__:
                        print(f"    ⚠ OOM en n_E={nE}, t×{t_mult}, "
                              f"sample {env_idx}. Limpiando...")
                        if USE_GPU:
                            cp.get_default_memory_pool().free_all_blocks()
                            cp.get_default_pinned_memory_pool().free_all_blocks()
                    else:
                        print(f"    ⚠ Error sample {env_idx}: {e}")
                    continue
            
            if not Ds:
                print(f"    t={t_evolve:7.1f} (×{t_mult:4.1f}) │ SKIP (sin datos)")
                continue
            
            elapsed = time.time() - t0
            
            result = {
                'D_mean': float(np.mean(Ds)),
                'D_std': float(np.std(Ds)),
                'D_sem': float(np.std(Ds) / np.sqrt(len(Ds))),
                'f_mean': float(np.mean(fs)),
                'anisotropy_mean': float(np.mean(anis)),
                'non_unitality_mean': float(np.mean(nunis)),
                't_evolve': t_evolve,
                't_mult': t_mult,
                'n_samples_actual': len(Ds),
                'elapsed': elapsed
            }
            results[nE][t_mult] = result
            
            print(f"    t={t_evolve:7.1f} (×{t_mult:4.1f}) │ "
                  f"D={result['D_mean']:.6f}±{result['D_sem']:.6f} │ "
                  f"f={result['f_mean']:.4f} │ "
                  f"anis={result['anisotropy_mean']:.5f} │ "
                  f"nunit={result['non_unitality_mean']:.5f} │ "
                  f"[{elapsed:.1f}s]")
            
            # ETA
            elapsed_total = time.time() - start_total
            total_expected = len(feasible) * len(T_MULTIPLIERS)
            done = sum(1 for nE2 in results for _ in results[nE2])
            if done > 0:
                eta = elapsed_total * (total_expected / done - 1)
                print(f"    ({done}/{total_expected} puntos, "
                      f"ETA: {eta/60:.1f}min)", flush=True)
    
    elapsed_total = time.time() - start_total
    
    # ═══════════════════════════════════════════════════════════════════
    # ANÁLISIS
    # ═══════════════════════════════════════════════════════════════════
    
    print(f"\n{'═'*70}")
    print(f"  RESULTADOS — B200 Edge Test")
    print(f"{'═'*70}")
    
    # ── 1. D vs n_E para cada t_mult ──
    print(f"\n  D(n_E) por tiempo de evolución:")
    print(f"  {'n_E':>4} │ {'dim_E':>12}", end="")
    for tm in T_MULTIPLIERS:
        print(f" │ {'t×'+str(tm):>10}", end="")
    print()
    
    for nE in feasible:
        print(f"  {nE:4d} │ {2**nE:12,}", end="")
        for tm in T_MULTIPLIERS:
            if tm in results.get(nE, {}):
                D = results[nE][tm]['D_mean']
                print(f" │ {D:10.6f}", end="")
            else:
                print(f" │ {'---':>10}", end="")
        print()
    
    # ── 2. Convergencia temporal ──
    print(f"\n  Convergencia temporal:")
    for nE in feasible:
        Ds_t = [results[nE][tm]['D_mean'] for tm in T_MULTIPLIERS if tm in results.get(nE, {})]
        if len(Ds_t) >= 2:
            delta = abs(Ds_t[-1] - Ds_t[-2]) / max(Ds_t[-2], 1e-10)
            converged = "✓ CONVERGIDO" if delta < 0.05 else "⟳ aún cambiando"
            print(f"    n_E={nE:2d}: D_final={Ds_t[-1]:.6f}, "
                  f"Δ_rel={delta:.3f} {converged}")
    
    # ── 3. Escalamiento D vs n_E ──
    t_max = max(T_MULTIPLIERS)
    D_scaling = []
    nE_scaling = []
    for nE in feasible:
        if t_max in results.get(nE, {}):
            D_scaling.append(results[nE][t_max]['D_mean'])
            nE_scaling.append(nE)
    
    if len(D_scaling) >= 3:
        D_arr = np.array(D_scaling)
        nE_arr = np.array(nE_scaling, dtype=float)
        
        valid = D_arr > 1e-15
        if np.sum(valid) >= 3:
            coeffs = np.polyfit(nE_arr[valid], np.log(D_arr[valid]), 1)
            slope = coeffs[0]
            
            print(f"\n  Escalamiento (t_mult={t_max}):")
            print(f"    log(D) vs n_E: pendiente = {slope:.4f}")
            
            if slope < -0.1:
                print(f"    → D DECAE EXPONENCIALMENTE: D ~ exp({slope:.3f}·n_E)")
                nE_001 = (np.log(0.01) - coeffs[1]) / slope
                print(f"    → D < 0.01 estimado en n_E ≈ {nE_001:.0f}")
            elif slope < -0.01:
                print(f"    → D decae, posiblemente power law")
            else:
                print(f"    → D NO decae (pendiente ≥ 0)")
    
    # ── 4. Reducción total ──
    if len(D_scaling) >= 2:
        red = (1 - D_scaling[-1] / D_scaling[0]) * 100
        print(f"\n  Reducción total: {D_scaling[0]:.6f} → {D_scaling[-1]:.6f} "
              f"({red:.1f}%)")
    
    # ── 5. D PROMEDIO por n_E ──
    print(f"\n  D PROMEDIO (sobre todos los tiempos):")
    for nE in feasible:
        all_D = [results[nE][tm]['D_mean'] for tm in T_MULTIPLIERS 
                 if tm in results.get(nE, {})]
        if all_D:
            D_prom = np.mean(all_D)
            t_total = sum(results[nE][tm]['elapsed'] for tm in results[nE])
            print(f"    n_E={nE:2d} ({N_S+nE:2d}q): D_PROMEDIO = {D_prom:.6f} "
                  f"(Tiempo: {t_total:.1f}s)")
    
    # ── 6. Comparación con H200 (si hay solapamiento) ──
    h200_overlap = [nE for nE in feasible if nE <= 18]
    if h200_overlap:
        print(f"\n  Zona de solapamiento con H200 (n_E ≤ 18):")
        print(f"  → {len(h200_overlap)} puntos compartidos para validación cruzada")
    
    # ── VEREDICTO ──
    print(f"\n  {'█'*60}")
    if len(D_scaling) >= 2:
        red = (1 - D_scaling[-1] / D_scaling[0]) * 100
        if red > 60:
            print(f"  VEREDICTO: D → 0 CONFIRMADO ({red:.0f}% reducción)")
            print(f"  → Canal físico CONVERGE a despolarización")
            print(f"  → JADE: propiedad física demostrada")
        elif red > 30:
            print(f"  VEREDICTO: D DECRECE ({red:.0f}%)")
            print(f"  → Tendencia positiva, escalar más")
        elif red > 10:
            print(f"  VEREDICTO: D DECRECE LENTAMENTE ({red:.0f}%)")
        else:
            print(f"  VEREDICTO: D ESTABLE o PLATEAU ({red:.1f}%)")
            print(f"  → Canal tiene estructura residual no-despolarizante")
    print(f"  {'█'*60}")
    
    # ═══════════════════════════════════════════════════════════════════
    # JSON
    # ═══════════════════════════════════════════════════════════════════
    
    output = {
        'metadata': {
            'experiment': 'JADE Edge Test v4.2 — PTM + Trotter (B200)',
            'question': 'D → 0 as n_E grows?',
            'method': 'PTM reconstruction (4 basis states) + Frobenius distance',
            'evolution': f"Trotter {'GPU' if USE_GPU else 'CPU'} ({TROTTER_STEPS} steps)",
            'hamiltonian': 'H = -J1·ZZ_nn - h·X - λ·J2·ZZ_nnn',
            'J1': J1, 'J2': J2, 'h': H_FIELD, 'lambda': LAMBDA,
            'n_S': N_S,
            'n_E_values': feasible,
            't_multipliers': T_MULTIPLIERS,
            'trotter_steps': TROTTER_STEPS,
            'samples_policy': 'adaptive (20/15/10/5 by n_total)',
            'timestamp': datetime.now().isoformat(),
            'elapsed_seconds': elapsed_total,
            'gpu': GPU_NAME,
            'gpu_mem_gb': GPU_MEM,
            'max_qubits': max_q,
            'note': 'Same Trotter engine as H200/RTX6000 runs. Extended range only.'
        },
        'results': {
            str(nE): {
                str(tm): results[nE][tm] for tm in results[nE]
            } for nE in results
        },
        'scaling': {
            'n_E': nE_scaling,
            'D_at_max_t': D_scaling,
            't_mult_used': t_max
        }
    }
    
    if len(D_scaling) >= 3:
        valid = np.array(D_scaling) > 1e-15
        if np.sum(valid) >= 3:
            coeffs = np.polyfit(
                np.array(nE_scaling, dtype=float)[valid],
                np.log(np.array(D_scaling)[valid]), 1
            )
            output['scaling']['log_fit_slope'] = float(coeffs[0])
            output['scaling']['model'] = f"D ~ exp({coeffs[0]:.4f} * n_E)"
    
    output['d_promedio'] = {}
    for nE in feasible:
        all_D = [results[nE][tm]['D_mean'] for tm in T_MULTIPLIERS 
                 if tm in results.get(nE, {})]
        if all_D:
            output['d_promedio'][str(nE)] = float(np.mean(all_D))
    
    filename = f"jade_edge_v4.2_B200_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n  Archivo: {filename}")
    print(f"  Tiempo total: {elapsed_total/60:.1f} min")
    print(f"\n{'═'*70}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
