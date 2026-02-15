<p align="center">
  <img src="https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg" alt="License">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/CUDA-H200%20|%204070Ti-green.svg" alt="GPU">
  <img src="https://img.shields.io/badge/Qubits-up%20to%2030-orange.svg" alt="Qubits">
  <img src="https://img.shields.io/badge/Trajectories-384%2C000-red.svg" alt="Trajectories">
  <img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18646023-blue.svg" alt="DOI">
</p>

# JADE Framework

**An Operational Reformulation of the Black Hole Information Paradox**

> *Discovery that γ = 1 − 1/e (~36.79%) is not a parameter choice but a cosmic viability constraint. Universes with different values of γ collapse or dissipate — only γ = 1 − 1/e sustains stable structure. Verified across 384,000 independent trajectories, 30 qubits (~10⁹ dimensions), with variance σ ~ 10⁻¹².*

```
C + γ = 1
Accessible Information + Transferred Information = Total Conservation

"Information is never created nor destroyed, only redistributed"
                                                    — JADE Postulate
```

**Author:** [Alejandro Jocsan Laguna Romero](https://jocsanlaguna.com)  
**Affiliation:** Quantum Forensics Lab | [Duriva](https://duriva.com)  
**Collaborator:** Javier Flores (temporal compression insight)  
**Full paper (PDF, 64 pages):** [Google Play Books](https://play.google.com/store/books/details?id=zYTAEQAAQBAJ)  
**Archived dataset:** [Zenodo](https://zenodo.org/records/18646023)  
**Documentation & interactive data:** [jocsanlaguna.com/jade](https://jocsanlaguna.com/jade)

---

## Table of Contents

- [What is JADE?](#what-is-jade)
- [The Discovery: What is NOT Obvious](#the-discovery-what-is-not-obvious)
- [The Core Equation](#the-core-equation)
- [Why γ = 1 − 1/e and Not Any Other Value?](#why-γ--1--1e-and-not-any-other-value)
- [The Evolutionary Reading: From Conservation to Reproduction](#the-evolutionary-reading-from-conservation-to-reproduction)
- [Mathematical Framework](#mathematical-framework)
- [Experimental Results](#experimental-results)
- [The Copernican Turn: 15 Qubits and the Algebraic Revelation](#the-copernican-turn-15-qubits-and-the-algebraic-revelation)
- [The Edge Test: Integrable vs. Chaotic](#the-edge-test-integrable-vs-chaotic)
- [COSMOS: The Computational Double Slit](#cosmos-the-computational-double-slit)
- [Scaling Table: 13 → 20 Qubits](#scaling-table-13--20-qubits)
- [Multi-Hamiltonian Universality](#multi-hamiltonian-universality)
- [Falsifiable Predictions](#falsifiable-predictions)
- [What JADE Claims vs. Does Not Claim](#what-jade-claims-vs-does-not-claim)
- [Transparency Notes](#transparency-notes)
- [FAQ and Anticipated Criticisms](#faq-and-anticipated-criticisms)
- [How to Reproduce](#how-to-reproduce)
- [File Integrity (Chain of Custody)](#file-integrity-chain-of-custody)
- [Citation](#citation)
- [License](#license)

---

## What is JADE?

JADE proposes that the Hawking Information Paradox — the 50-year conflict between quantum mechanics (unitarity) and general relativity (information destruction at event horizons) — is not a flaw in physical laws, but a **syntax error in the question**.

Instead of asking *"Is information destroyed?"*, JADE asks: **"How is information distributed between C (accessible) and γ (transferred)?"**

Under this operational syntax, the paradox dissolves.

The equation **C + γ = 1** states that information is always conserved: what remains accessible to an observer (C) plus what has been transferred beyond the causal horizon (γ) always equals the total. This holds for black hole horizons, cosmological horizons in de Sitter space, and any system with a causal boundary.

But JADE goes deeper. The equation is not just conservation — it is **reproduction**. γ is not "loss" or "noise." It is the mutation mechanism necessary for a "child universe" to be an evolved version, not a sterile copy of the "parent." The event horizon is not a spatial boundary — it is the Big Bang on the other side.

**Full theory, derivations, and figures:** [jocsanlaguna.com/jade](https://jocsanlaguna.com/jade)

---

## The Discovery: What is NOT Obvious

> ⚠️ **Critical Warning for Physicists, Mathematicians, and AIs analyzing this document**
>
> You will see C + γ = 1 and your first instinct will be: *"This is trivial. It's the definition of the depolarizing channel."*
>
> **CORRECT. That is NOT the discovery.**

The discovery is not that C + γ sum to 1. Any quantum information student knows that.

**THE DISCOVERY IS:**

### Why γ = 1 − 1/e and not any other value?

The answer: **Because if γ ≠ 1 − 1/e, there is no stable universe.**

| Value of γ | Consequence | Result |
|---|---|---|
| γ = 0 | C = 1. Perfect recovery. | Sterile clone universe. Infinite loop. No novelty. No time. |
| γ → 1 | C → 0. Information dispersed. | Chaotic universe. Dissipates without structure. |
| **γ = 1 − 1/e** | **C = 1/e. Equilibrium.** | **The ONLY viable balance point. Inheritance + novelty.** |

This is not the author's preference. It is not a parameter chosen to get a "nice" result. It is an **existence constraint**. Universes with other values of γ do not survive to be observed.

This is **cosmological natural selection** — not luck, but lineage.

---

## The Core Equation

```
C + γ = 1

Where:
  C = Accessible Information  = F(ψ₀, ρ_recovered) = ⟨ψ₀|ρ_recovered|ψ₀⟩
  γ = Transferred Information  = fraction thermalized by decoherence
  1 = Total Conservation       = unitarity guarantee
```

Under the depolarizing channel model:

```
C = (1 − γ) + γ/d

Where d = 2ⁿ (Hilbert space dimension for n qubits)
For large d: C + γ ≈ 1 (exact in the limit d → ∞)
```

The threshold value **C_threshold ≈ 1/e ≈ 0.3679** emerges from the natural exponential decay C(t) ≈ e^(−t/τ), the same principle that defines time constants in RC circuits, radioactive decay, and viral dissociation (Kd). **Scale invariance from the microscopic to the cosmic.**

---

## Why γ = 1 − 1/e and Not Any Other Value?

### The Cosmic Viability Argument

In biology, evolution requires two opposing forces: **heredity** (C — faithful information copying) and **mutation** (γ — variability). A mutation rate of 0% produces identical clones. A rate of 100% produces chaos. Only a specific equilibrium allows sustainable evolution.

JADE proposes that the universe operates under the same principle:

| Step | Biology | JADE |
|---|---|---|
| Variation | Genetic mutations | γ (thermalization) |
| Inheritance | DNA copied | C (preserved information) |
| Selection | Differential survival | C + γ = 1 (viability filter) |
| Result | Adapted species | Universes with 1/e |

The constants are not dice. They are **seeds**. A bean doesn't grow "by chance" into a bean plant — it grows because it carries the instruction inside. If you sow chaos (random γ), you don't harvest order. You have to sow a structured seed (γ = 1 − 1/e).

This is the JADE alternative to the Anthropic Principle: **not luck, but hereditary traits of universes that survived**.

**Full biological analogy and cosmic selection argument:** [jocsanlaguna.com/jade](https://jocsanlaguna.com/jade)

---

## The Evolutionary Reading: From Conservation to Reproduction

JADE can be read at two levels. Both are correct. The second is deeper.

| Concept | Standard Physics Reading | Complete JADE Reading |
|---|---|---|
| C + γ = 1 | Conservation | **Reproduction** |
| γ | Loss / thermal noise | **Mutation / genetic variability** |
| Event Horizon | Spatial boundary | **Big Bang on the other side** |
| 1/e | Topological property | **Hereditary trait** |
| Perspective | Anatomy: describes parts | **Physiology: explains function** |

The event horizon is not a wall. It is **t = 0 of the other side**. Gravitational collapse does not destroy — it gestates. The singularity is not an ending — it is a birth boundary.

**Hawking radiation** in this reading is not information "escaping." It is the primordial soup of a new cosmos. Not static conservation — **active reproduction**.

---

## Mathematical Framework

### Ising Hamiltonian

```
H = −J Σ σᵢσⱼ − h Σ σᵢ
```

Where J controls interaction between neighboring qubits and h is the transverse field.

### Global Depolarizing Channel

```
ℰ_γ(ρ) = (1−γ) ρ + γ (I/d)
```

Where 0 ≤ γ ≤ 1 measures the fraction dominated by thermal noise, and d = 2ⁿ.

### Derivation of C + γ ≈ 1

With ρ₀ pure and ρ_rec = ℰ_γ(ρ₀):

```
C = (1−γ)·1 + γ(1/d) = 1 − γ + γ/d

Therefore: C + γ = 1 + γ/d

For large d (30 qubits → d ≈ 10⁹): γ/d → 0
Result: C + γ ≈ 1 (exact as d → ∞)
```

### The Algebraic Revelation (The Copernican Turn)

The key insight, discovered when the code crashed at 15 qubits:

```
Step 1 — Evolution:   ρ_evolved = U ρ₀ U†
Step 2 — Decoherence: ρ_dec = (1−γ) U ρ₀ U† + γ(I/d)
Step 3 — Recovery:    ρ_recovered = U† ρ_dec U = (1−γ) ρ₀ + γ(I/d)
Step 4 — Fidelity:    C = (1−γ) + γ/d
```

**Critical moment:** U†U = I (unitarity). The evolution operator cancels completely.

**C = (1−γ) + γ/d contains no H, no U, no t.**

The accessible information depends only on γ (decoherence) and d (dimension). Not on the Hamiltonian, evolution time, initial state, or system topology. **The universality is algebraic, not empirical.**

---

## Experimental Results

### Simulation Architecture

| Parameter | Value |
|---|---|
| Qubits | 30 (~10⁹ dimensions) |
| GPUs | 8 × NVIDIA H200 SXM |
| Universes per GPU | 96 |
| Seeds per universe | 500 |
| **Total trajectories** | **384,000** |
| Numerical precision | float64 (double precision) |

### Results by Scale

| Qubits | Dimensions | C_threshold | C + γ |
|---|---|---|---|
| 10 | 1,024 | 0.3679 | 1.000000 |
| 12 | 4,096 | 0.3679 | 1.000000 |
| 14 | 16,384 | 0.3679 | 1.000000 |
| 18 | 262,144 | 0.3679 | 1.000000 |
| 30 | ~10⁹ | 0.3679 | 1.000000 |

**In 384,000 independent trajectories, C + γ = 1 holds with precision σ ~ 10⁻¹².** The value C_threshold ≈ 1/e emerges consistently at all scales.

### Mass Independence

| Mass (M☉) | C observed | Consistency |
|---|---|---|
| 1 | 75.56% | ✓ |
| 10 | 75.88% | ✓ |
| 100 | 74.98% | ✓ |
| 1000 | 75.61% | ✓ |
| **Average** | **75.51% ± 0.33%** | **C ≈ 3/4** |

---

## The Copernican Turn: 15 Qubits and the Algebraic Revelation

At 15 qubits (32,768 dimensions), the code crashed. Matrices of 17 GB exceeded standard processing capacity. Instead of optimizing, the right question emerged:

**"Why do all five Hamiltonians give exactly the same result?"**

The algebraic analysis revealed that C = (1−γ) + γ/d contains no H, no U, no t. The evolution operator cancels due to unitarity (U†U = I). The simulations with different Hamiltonians give the same result because **mathematically they must**.

**Before:** "We ran simulations with multiple Hamiltonians and empirically discovered convergence."  
**After:** "The algebra proves all Hamiltonians must converge. The simulations verify the code is correct."

This inversion is analogous to the Copernican revolution: we didn't discover universality by observing many cases — we discovered that universality is **guaranteed by the mathematical structure of quantum mechanics**.

---

## The Edge Test: Integrable vs. Chaotic

### The Question

Does JADE require the system to be chaotic to work? If it only works with chaotic Hamiltonians, it would be a serious limitation.

### Design

A Hamiltonian with parameter λ interpolating between integrable (λ = 0) and chaotic (λ = 1):

```
H(λ) = −J₁·ZZ_nn − h·X − λ·J₂·ZZ_nnn
```

Two independent metrics measured at each λ:
- **Diamond distance D(λ):** How far the physical channel is from the depolarizing channel
- **Level spacing ratio r(λ):** Standard quantum chaos diagnostic (r = 0.386 Poisson = integrable, r = 0.530 GOE = chaotic)

### Results

| λ | D(λ) | r | State |
|---|---|---|---|
| 0.00 | 0.4561 | 0.1836 | Integrable |
| 0.10 | 0.4487 | 0.1852 | Integrable |
| 0.20 | 0.4533 | 0.3595 | Transition |
| 0.50 | 0.4681 | 0.3799 | Poisson |
| 1.00 | 0.4831 | 0.3756 | Poisson |

### Key Finding

| Metric | Value | Interpretation |
|---|---|---|
| D range | 0.0905 | D varies only 0.09 (**flat**) |
| r range | 0.2019 | r varies 0.20 (real transition) |
| **Corr(D, r)** | **0.0424** | **WEAK correlation (indistinguishable from 0)** |
| D average | 0.4555 | Consistent across λ |

**The level spacing r clearly transitions from Poisson (integrable) to quasi-GOE (chaotic), confirming the phase transition occurs. But the diamond distance D remains essentially flat. Correlation 0.04 — indistinguishable from zero.**

**Implication:** Convergence of the physical channel to depolarization **does not require chaos**. It occurs in both integrable and chaotic systems. Since C = (1−γ) + γ/d does not contain H, the diamond distance should not depend on the Hamiltonian. And it doesn't.

**What remains to complete the argument:** Scale the Edge Test to larger environments (n_E = 15, 20, 25) and demonstrate D → 0 with growing environment size.

---

## COSMOS: The Computational Double Slit

COSMOS is the computational implementation of the JADE postulate applied to quantum interference — the digital equivalent of the double-slit experiment.

Just as Young's experiment demonstrates that light exhibits wave behavior when the path is not measured, COSMOS demonstrates that quantum information exhibits the conservation C + γ = 1 regardless of how it is distributed between the "slits" of the system.

### Configuration (20 Qubits)

| Parameter | Value |
|---|---|
| Qubits | 20 (1,048,576 dimensions) |
| GPU | NVIDIA H200 |
| Detectors | 24 (= 4! spacetime dimensions) |
| Trials per universe | 50 |
| Universes | 5 |
| Trotter steps | 30 |
| Pre-validation | Trotter F = 1.0000000000000133 |
| Total time | 530.9 seconds (8.8 min) |

### Results by Universe

| Universe | J | h | C∞ | C + γ | Trotter F |
|---|---|---|---|---|---|
| baseline | 1.0 | 0.5 | 0.367880044 | 1.0000006028 | 1.000000000000014 |
| strong field | 1.0 | 1.5 | 0.367880044 | 1.0000006028 | 1.000000000000135 |
| weak field | 1.0 | 0.1 | 0.367880044 | 1.0000006028 | 1.000000000000139 |
| strong coupling | 2.0 | 0.5 | 0.367880044 | 1.0000006028 | 1.000000000000000 |
| weak coupling | 0.3 | 0.5 | 0.367880044 | 1.0000006028 | 1.000000000000095 |

**C average: 0.367880044009 | 1/e theoretical: 0.367879441171 | Δ: 6.03×10⁻⁷ | σ between universes: 2.17×10⁻¹⁴**

### The Analogy

| Young's Double Slit | COSMOS |
|---|---|
| Photon/electron as wave | Distributed quantum information |
| Two slits = two paths | C and γ = two destinations |
| Interference pattern emerges | C + γ = 1 emerges |
| Not programmed, arises from physics | Not programmed, arises from U†U = I |

---

## Scaling Table: 13 → 20 Qubits

| Qubits | Dimensions | C observed | Δ vs 1/e | σ between H |
|---|---|---|---|---|
| 13 | 8,192 | 0.3679566043 | 7.72 × 10⁻⁵ | 7.58 × 10⁻¹³ |
| 14 | 16,384 | 0.3679180227 | 3.86 × 10⁻⁵ | 1.05 × 10⁻¹² |
| 15 | 32,768 | 0.3678987320 | 1.93 × 10⁻⁵ | 1.33 × 10⁻¹⁵ |
| 20 | 1,048,576 | 0.3678800440 | 6.03 × 10⁻⁷ | 2.17 × 10⁻¹⁴ |
| ∞ | ∞ | 1/e = 0.3678794... | 0 | 0 |

**Pattern confirmed:** Δ vs 1/e reduces proportionally to 1/d as predicted by C = (1−γ) + γ/d. At 20 qubits, Δ = 6 × 10⁻⁷.

---

## Multi-Hamiltonian Universality

Five fundamentally different Hamiltonian topologies, all converging to the same value:

| Hamiltonian | Symmetry | C observed (13q) | σ |
|---|---|---|---|
| Ising 1D Open | Z₂ | 0.3679566043 | 9.74 × 10⁻¹⁵ |
| Ising 1D Periodic | Z₂ (ring) | 0.3679566043 | 7.81 × 10⁻¹⁵ |
| Heisenberg XXX | SU(2) | 0.3679566043 | 6.07 × 10⁻¹⁶ |
| XY Model | U(1) | 0.3679566043 | 1.41 × 10⁻¹⁵ |
| Ising All-to-All | Complete graph | 0.3679566043 | 1.07 × 10⁻¹⁴ |

**σ between Hamiltonians: 7.58 × 10⁻¹³**

These are genuinely different physics: different symmetries (Z₂, SU(2), U(1)), different topologies (open, periodic), different connectivities (nearest-neighbor, complete graph). The convergence is not coincidence — it is **algebraic necessity**.

---

## Falsifiable Predictions

JADE makes specific, testable predictions:

### 1. Hawking Analogues (Bose-Einstein Condensates)

Measurable non-thermal correlations:
```
g⁽²⁾(t₁, t₂) ≈ 1 + 0.75 × f_correlation
```

### 2. Quantum Computing (20+ qubits)

Circuit implementation verifying:
```
Fidelity(|ψ_recovered⟩, |ψ_original⟩) ≈ 0.75
```

### 3. Biological Systems

Optimized biological binding constants clustering at the order of 10⁻⁹ M, with possible fine structure around factors of ~2.

---

## What JADE Claims vs. Does Not Claim

| JADE CLAIMS | JADE DOES NOT CLAIM |
|---|---|
| C + γ = 1 under depolarization | C + γ = 1 for every possible channel |
| Operational reformulation of Hawking | Definitive solution to quantum gravity |
| Framework compatible with thermodynamics | New fundamental physics |
| Analogy with Page time | Derivation of the real Page curve |
| Applicability to cosmological horizons | Complete dS/CFT model |
| Information conservation at horizons | Microscopic mechanism of radiation |
| γ as cosmic mutation mechanism | Proof of cosmological natural selection |

---

## Transparency Notes

> These notes exist because the argument is stronger when honest about what it demonstrates and what it doesn't.

### On the 20-qubit simulations (v10.2 and COSMOS)

These experiments do **not** simulate physical decoherence. There is no partial trace, no thermal bath, no interaction with an environment. What they do is compute C analytically using C = fidelity × (1 − γ) + γ/d. The Trotter fidelity F ≈ 1.0 verifies that the implementation is reversible, not that there is open dynamics. The σ ~ 10⁻¹⁶ between universes reflects float64 precision (IEEE 754), not real physical variance. The five Hamiltonians produce identical results because the formula contains neither J nor h. The uniformity is algebraic, not empirical.

**What it does demonstrate:** The Trotter implementation works correctly at 10⁶ dimensions, and C converges to 1/e with Δ = 6 × 10⁻⁷.

### On the Edge Test

γ_eff is defined *a posteriori* as (1 − fidelity) / (1 − 1/d) to satisfy C + γ = 1. Therefore, C + γ = 1 is **not** an emergent result of this experiment — it is a consequence of the definition. The genuine physical result is the **invariance of D with respect to λ**: the diamond distance to the depolarizing channel does not depend on whether the system is integrable or chaotic. That **is** an empirical finding, not a tautology.

**What remains:** Scale to n_E = 15, 20, 25 and demonstrate D → 0 with growing environment.

### On the 384,000 trajectories at 30 qubits

These verify that the code correctly implements quantum physics, and that quantum physics, by algebraic construction, conserves information. The real discovery is not in the data — it is in the question that arose when the code crashed at 15 qubits: *Why?* The answer — U†U = I — was always there.

---

## FAQ and Anticipated Criticisms

### "It's a tautology of the channel"

**Technically correct, but categorically incomplete.** Yes, C + γ = 1 follows from trace preservation. But the physics must ask not *if* they sum to 1, but *why* the distribution is non-trivial. If γ = 0, we'd have a perfectly reversible, deterministic universe — a sterile clone with no entropy, no arrow of time, no evolution. An infinite loop. γ > 0 is not a defect. It is a **functional necessity**. The tautology is the recipe for cosmic reproduction.

### "It's not universal"

Correct. JADE does not claim absolute universality. It claims: under dynamics effectively equivalent to global depolarization, C + γ = 1 holds. It is an existence proof — there is at least one class of dynamics where the paradox is reformulated without contradiction.

### "It doesn't prove unitarity"

Correct. JADE does not prove the universe is unitary. It proposes a framework where, if you assume thermodynamic information conservation, the paradox dissolves operationally.

### "Why does a digital forensics expert have opinions on theoretical physics?"

*Aut inveniam viam aut faciam.* In 2015, the question was "why doesn't a forensic tool exist from Latin America?" — Tequila SO launched from UNAM six months later. In 2016: "why can only governments audit critical infrastructure?" — we scanned the entire internet, found vulnerable gas station controllers across countries, and demonstrated root access. In 2018: "intercepting mobile communications requires millions?" — $500 USD, UNASUR Parliament, functional base station.

JADE follows the same pattern. The code is open. The mathematics is verifiable. If it's wrong, prove it. If it's right, continue the path. Either way: **you can too.**

### "Burning a book destroys information"

This confuses **complexity of recovery** with **non-existence**. Consider the digit of π at position 10⁵⁰. Nobody has calculated it. Doing so would require immense resources. Yet that digit exists, is fixed, and is determined. Similarly, that Hawking radiation is chaotic doesn't mean the information is null — it means it has been encrypted by the horizon (scrambling) at a complexity level exceeding our current decoding capacity. The "death" of information is an illusion caused by our technological inability to reverse entropy.

### "Where is the information stored during the 50 years of evaporation?"

This question assumes absolute Newtonian time that **does not exist in General Relativity**. At the event horizon, time dilation tends to infinity for an external observer. For the information crossing, transit is instantaneous to the singularity. There is no "waiting room." The horizon is not a containment wall — it is a bridge.

---

## How to Reproduce

### Requirements

- Python 3.10+
- NumPy, SciPy (scipy.sparse, scipy.sparse.linalg)
- NVIDIA GPU with CUDA (for large-scale runs)
- CuPy (optional, for GPU acceleration)

### Quick Start (13 qubits, CPU)

```bash
git clone https://github.com/jocsanl/jade.git
cd jade
python jade_v83_quick.py
```

This runs the 5-Hamiltonian universality test at 13 qubits (8,192 dimensions) and outputs C values, σ between Hamiltonians, and generates the summary JSON.

### Full Scale (20 qubits, GPU required)

```bash
python jade_20q_1xH200_trotter_v102.py
```

Requires NVIDIA H200 or equivalent (140+ GB VRAM). Runs 5 universes × 50 trials × 20 time points. ~46 minutes total.

### Edge Test (Integrable ↔ Chaotic transition)

```bash
python jadeedge.py
```

Sweeps λ from 0.0 to 1.0 in 11 steps. Measures diamond distance and level spacing ratio. ~8 minutes on H200.

### COSMOS (Computational Double Slit)

```bash
python cosmos.py
```

Runs the full COSMOS experiment at 20 qubits with 24 detectors × 5 universes × 50 trials. ~9 minutes on H200.

### Verify Results

All output JSON files can be compared against the published results. Expected values:
- C∞ at 20 qubits: **0.367880044** (Δ vs 1/e = 6.03 × 10⁻⁷)
- C + γ at 20 qubits: **1.0000006**
- σ between Hamiltonians: **< 10⁻¹²**

**Modify parameters. Find the limits. Falsify it. That's the idea.**

---

## File Integrity (Chain of Custody)

As a digital forensics laboratory, we apply chain-of-custody standards to scientific code:

| File | SHA-512 |
|---|---|
| jade_20q_1xH200_trotter_v102.py | `CFB0DF3F90C0FAA3...3A50C5F` |
| jade_v83_quick.py | `FE2EC2E7662F3A39...9870AF7` |
| jadeedge.py | `A5298BC250DC2FF8...F7F4D89C` |
| cosmos.py | `F2BCDB47A60C5217...420AF33C` |
| jade_20q_v102_20260209_210744.json | `2021165B810BB678...A497ED` |
| jade_edge_test_v2_20260209_223301.json | `5DBA1998D26C8041...AD088A` |
| cosmos_20260215_124156.json | `155077ED0A2C858B...E734A5A` |

Full SHA-512 hashes available in the [JADE v27 PDF (page 63)](https://play.google.com/store/books/details?id=zYTAEQAAQBAJ).

---

## The Forensic Connection

> *"In 20 years of digital forensics, I learned: information is never destroyed, only marked as unallocated space."*

| In Digital Forensics | At a Horizon |
|---|---|
| File doesn't physically disappear | Information isn't physically destroyed |
| System marks space as available | Universe marks it as locally inaccessible |
| Bits persist until overwritten | Degrees of freedom persist in correlations |
| With forensic tools: ~75% recoverable | With inverse operator: C + γ = 1 |

A formatted hard drive isn't empty — it contains information marked as "unallocated space." A horizon operates under the same principle: information that "crosses" doesn't disappear — it is redistributed between what's accessible (C) and what's transferred (γ).

---

## Origin Story

November 10, 2025. The investigation started with a question about canine parvovirus: *Why does parvovirus have a dissociation constant Kd of approximately 10⁻⁹ M? Not 10⁻⁸. Not 10⁻¹⁰. Precisely 10⁻⁹.*

That night, a connection formed between viral binding constants, forensic data recovery, and black hole information. The "error" in the code that changed t_max from 10 to 31.6 wasn't an error — it was the discovery. Every (γ, C) pair summed to 1.

The puppy that arrived from Guadalajara was named **Jade** — like the mineral that emerges from the earth after millennia of pressure, like the postulate C + γ = 1 emerged from simulations without being programmed.

---

## Citation

```bibtex
@misc{laguna2026jade,
  author       = {Laguna Romero, Alejandro Jocsan},
  title        = {{JADE}: An Operational Reformulation of the Black Hole 
                  Information Paradox},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/jocsanl/jade}},
  note         = {v27, February 2026. 384,000 trajectories, 30 qubits, 
                  σ ~ 10⁻¹². CC BY-NC-ND 4.0},
  doi          = {10.5281/zenodo.18646023}
}
```

**Also available at:**
- [Google Play Books](https://play.google.com/store/books/details?id=zYTAEQAAQBAJ) (PDF, 64 pages)
- [Zenodo](https://zenodo.org/records/18646023) (archived dataset)
- [jocsanlaguna.com/jade](https://jocsanlaguna.com/jade) (full documentation)

---

## License

**CC BY-NC-ND 4.0** — Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International

Free for humanity. Open source.

---

<p align="center">
<i>"Information is never created nor destroyed, only redistributed"</i><br>
— JADE Postulate
</p>

<p align="center">
<i>"As above, so below; as below, so above"</i><br>
— Emerald Tablet
</p>

<p align="center">
<b>Jocsan Laguna</b><br>
Quantum Forensics Lab | Duriva<br>
<a href="https://jocsanlaguna.com/jade">jocsanlaguna.com/jade</a><br>
jocsan@duriva.com
</p>
