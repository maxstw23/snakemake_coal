# Coalescence Framework and Derivations

## Current Method (implemented in pipeline)

The pipeline uses a simplified version of the coalescence model that neglects the resonance decay fraction. Under assumptions 1-3 only (no secondary decay term), the pion and antiproton flows decompose as:

$$v_2^{\pi^+} = N_{trans}^u \, v_{2,t} + (2 - N_{trans}^u) \, v_{2,p}$$

$$v_2^{\pi^-} = N_{trans}^d \, v_{2,t} + (2 - N_{trans}^d) \, v_{2,p}$$

$$v_2^{\bar{p}} = 3 \, v_{2,p} \implies v_{2,p} = \tfrac{1}{3} v_2^{\bar{p}}$$

Substituting to eliminate $v_{2,p}$:

$$v_2^{\pi^-} - \tfrac{2}{3} v_2^{\bar{p}} = N_{trans}^d (v_{2,t} - v_{2,p})$$

$$v_2^{\pi^+} - \tfrac{2}{3} v_2^{\bar{p}} = N_{trans}^u (v_{2,t} - v_{2,p})$$

The unknown $(v_{2,t} - v_{2,p})$ cancels in the ratio:

$$\boxed{\frac{v_2^{\pi^-} - \tfrac{2}{3} v_2^{\bar{p}}}{v_2^{\pi^+} - \tfrac{2}{3} v_2^{\bar{p}}} = \frac{N_{trans}^d}{N_{trans}^u} = \frac{2N + Z}{N + 2Z}}$$

For Au-197 ($Z = 79$, $N = 118$): prediction is $315/276 \approx 1.14$.

This is computed in `scripts/plot_v2_new.py` after resolution correction and centrality merging. The $v_2$ values are integrated over $p_T/n_q \in [0.08, 0.6]$ GeV/c, and the ratio is evaluated per centrality bin.

**Limitation**: This derivation ignores the resonance decay fraction $f$. When $f \neq 0$, common additive terms appear in both numerator and denominator, biasing the ratio toward unity. See the extended derivation below.

---

## Assumptions

1. **Coalescence**: Particle flow is the sum of its constituent quarks' flow.
2. **Produced quark universality**: All primordial produced light quarks share a common flow:
   $$v_n^{u_p} = v_n^{\bar{u}_p} = v_n^{d_p} = v_n^{\bar{d}_p} \equiv v_{n,p}$$
3. **Transported quark universality**: All transported quarks share a common flow:
   $$v_n^{u_t} = v_n^{d_t} \equiv v_{n,t}$$
4. **Secondary decay universality**: Quarks from resonance decays (e.g., rho -> pi pi) share a common flow:
   $$v_n^{u_s} = v_n^{\bar{u}_s} = v_n^{d_s} = v_n^{\bar{d}_s} \equiv v_{n,s}$$
5. **Isobar universality**: The values $v_{n,p}$, $v_{n,t}$, $v_{n,s}$ are the same for Ru+Ru and Zr+Zr. Testable via antiproton and net-proton $v_2$ comparison between isobars.
6. **Decay fraction universality**: The average number of quarks from resonance decay per pion, $N_{decay}$, is the same for Ru+Ru and Zr+Zr.

## Au+Au Derivation (with resonance fraction)

Let $f$ be the fraction of measured pions that originate from resonance decays. The flow decomposition becomes:

$$v_2^{\pi^+} = (1-f)[N_{trans}^u \, v_{2,t} + (2 - N_{trans}^u) \, v_{2,p}] + f \, v_{2,s}$$

$$v_2^{\pi^-} = (1-f)[N_{trans}^d \, v_{2,t} + (2 - N_{trans}^d) \, v_{2,p}] + f \, v_{2,s}$$

$$v_2^{\bar{p}} = 3 \, v_{2,p}$$

$$v_2^{p} = (2 N_{trans}^u + N_{trans}^d) \, v_{2,t} + (3 - 2 N_{trans}^u - N_{trans}^d) \, v_{2,p}$$

### Defining shorthand

Let $A \equiv N_{trans}^u (v_{2,t} - v_{2,p})$ and $B \equiv N_{trans}^d (v_{2,t} - v_{2,p})$.

Then:

$$v_2^{\pi^-} - v_2^{\pi^+} = (1-f)(B - A)$$

$$v_2^p - v_2^{\bar{p}} = 2A + B$$

### Solving for A and B

$$A = \frac{1}{3}\left(\Delta v_2^p - \frac{1}{1-f} \Delta v_2^\pi\right)$$

$$B = \frac{1}{3}\left(\Delta v_2^p + \frac{2}{1-f} \Delta v_2^\pi\right)$$

where $\Delta v_2^\pi \equiv v_2^{\pi^-} - v_2^{\pi^+}$ and $\Delta v_2^p \equiv v_2^p - v_2^{\bar{p}}$.

### The ratio

$$\frac{N_{trans}^d}{N_{trans}^u} = \frac{B}{A} = \frac{\Delta v_2^p + \frac{2}{1-f} \Delta v_2^\pi}{\Delta v_2^p - \frac{1}{1-f} \Delta v_2^\pi}$$

**Note on the "simple" ratio**: The ratio $(v_2^{\pi^-} - 2/3 v_2^{\bar{p}}) / (v_2^{\pi^+} - 2/3 v_2^{\bar{p}})$ is NOT exactly $N_d^{tr}/N_u^{tr}$ when $f \neq 0$, because the common additive terms $f(2v_{2,p} + v_{2,s})$ appear in both numerator and denominator, biasing the ratio toward unity. The $B/A$ formulation above avoids this by using the proton-antiproton splitting to isolate transported quark contributions directly.

For Au-197 ($Z = 79$, $N = 118$): prediction is $B/A = (2N+Z)/(N+2Z) = 315/276 \approx 1.14$.

## Isobar Derivation (Ru vs Zr)

Using isotopes Ru-96 ($Z=44$, $N=52$) and Zr-96 ($Z=40$, $N=56$):

| | $N_{trans}^u \propto N+2Z$ | $N_{trans}^d \propto 2N+Z$ |
|---|---|---|
| Ru-96 | 140 | 148 |
| Zr-96 | 136 | 152 |
| $\Delta$ (Ru - Zr) | +4 | -4 |

### Method 1: Ratio of isobar differences per species

Taking the Ru$-$Zr difference for each pion species cancels produced and secondary quark flows:

$$\Delta v_2^{\pi^+} \equiv v_{2,Ru}^{\pi^+} - v_{2,Zr}^{\pi^+} = \Delta N_{trans}^u \, (v_{2,t} - v_{2,p})$$

$$\Delta v_2^{\pi^-} \equiv v_{2,Ru}^{\pi^-} - v_{2,Zr}^{\pi^-} = \Delta N_{trans}^d \, (v_{2,t} - v_{2,p})$$

$$\boxed{\frac{\Delta v_2^{\pi^-}}{\Delta v_2^{\pi^+}} = \frac{\Delta N_{trans}^d}{\Delta N_{trans}^u} = \frac{-4}{4} = -1}$$

### Method 2: Ratio of pion splittings between isobars

$$\frac{(v_2^{\pi^-} - v_2^{\pi^+})_{Ru}}{(v_2^{\pi^-} - v_2^{\pi^+})_{Zr}} = \frac{N_d^{Ru} - N_u^{Ru}}{N_d^{Zr} - N_u^{Zr}} = \frac{148 - 140}{152 - 136} = \boxed{\frac{1}{2}}$$

### Why isobars are hard

Both predictions ($-1$ and $1/2$) are clean because the Ru$-$Zr difference cancels the resonance decay fraction, produced quark flow, and secondary quark flow entirely. However, the absolute pion $v_2$ differences between isobars are extremely small — the $\pi^+ / \pi^-$ splitting within each isobar is already small (ratios close to 1), and the inter-isobar differences are smaller still. This makes the measurement statistics-limited in practice.

### Additional consistency check

The combination $(v_2^p + v_2^{\bar{p}}) - (v_2^{\pi^-} + 2 v_2^{\pi^+})$ isolates:

$$N_{trans}^u \, v_{2,p} + 3 N_{decay} (v_{2,p} - v_{2,s})$$

This should be the same for Ru and Zr (under assumptions 5-6), providing a testable constraint.
