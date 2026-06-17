#!/usr/bin/env python3
"""
shors_algorithm.py
==================
Quantum factorization of semiprimes using Shor's Algorithm,
simulated on a classical computer via Qiskit Aer.

How to install
──────────────
    pip install qiskit qiskit-aer numpy

Semiprimes less than 100 are recommended. (Larger N increases circuit depth and simulation time.)

Algorithm overview
──────────────────
    Shor's algorithm factors N = p × q in two stages:
    
    1. QUANTUM — Quantum Phase Estimation (QPE)
       Find the period r of  f(x) = a^x mod N  for a random a.
       The QPE circuit encodes the phase s/r into a superposition,
       then the Inverse QFT reads it out as a measurement.
    
    2. CLASSICAL — Continued-fractions + GCD
       Decode r from the measured phase, then compute:
           p = gcd(a^(r/2) − 1,  N)
           q = gcd(a^(r/2) + 1,  N)
"""

import sys
import random
from math import gcd, ceil, log2
from fractions import Fraction

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# Qiskit 2.x: use QFTGate; fall back to the (deprecated) QFT class for 1.x
try:
    from qiskit.circuit.library import QFTGate    # Qiskit ≥ 2.1  (preferred)
    def make_iqft(n: int) -> object:
        return QFTGate(n).inverse()
except ImportError:
    from qiskit.circuit.library import QFT        # Qiskit 1.x fallback
    def make_iqft(n: int) -> object:
        return QFT(n, inverse=True, do_swaps=True, name="IQFT")

# UnitaryGate: moved to circuit.library in Qiskit 1.0
try:
    from qiskit.circuit.library import UnitaryGate
except ImportError:
    from qiskit.extensions import UnitaryGate     # Qiskit 0.x fallback


# ─────────────────────────────────────────────────────────────────────────────
# Number-theory helpers
# ─────────────────────────────────────────────────────────────────────────────

"""Trial-division primality test (fast enough for the small numbers we use)."""
def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

"""Return True iff n = p × q for primes p ≤ q (p may equal q)."""
def is_semiprime(n: int) -> bool:
    if n < 4:
        return False
    for p in range(2, int(n ** 0.5) + 1):
        if n % p == 0:
            return is_prime(p) and is_prime(n // p)
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Quantum-circuit building
# ─────────────────────────────────────────────────────────────────────────────

"""
Build the 2^n × 2^n permutation (unitary) matrix for:
    |x⟩  ──►  |a_pow · x  mod  N⟩    (x < N)
    |x⟩  ──►  |x⟩                    (x ≥ N, identity on out-of-range states)
"""
def modmul_unitary(a_pow: int, N: int, n_qubits: int) -> np.ndarray:
    dim = 2 ** n_qubits
    U = np.zeros((dim, dim), dtype=complex)
    for x in range(dim):
        y = (a_pow * x) % N if x < N else x
        U[y, x] = 1.0
    return U


"""
Build a controlled-U gate where  U |x⟩ = |a^(2^power) · x  mod N⟩.
Returns a ControlledGate: 1 control qubit + n_work target qubits.
The control qubit is listed first when the gate is appended to a circuit.
"""
def ctrl_modmul_gate(a: int, power: int, N: int, n_work: int):
    a_pow = pow(a, 2 ** power, N)           # fast modular exponentiation
    U = modmul_unitary(a_pow, N, n_work)
    return UnitaryGate(U, label=f"×{a_pow}").control(1)


"""
Quantum Phase Estimation (QPE) circuit for Shor's order-finding.

┌── Qubit layout ─────────────────────────────────────────────────┐
│   qubits  [0 … n_count-1]        Counting / QPE register        │
│   qubits  [n_count … total-1]    Work register (|1⟩ initially)  │
└─────────────────────────────────────────────────────────────────┘

┌── Circuit steps ─────────────────────────────────────────────────┐
│   1. H^⊗n_count          → equal superposition on counting reg  │
│   2. X on qubit n_count  → initialise work register to |1⟩       │
│   3. Controlled-U^(2^q)  → QPE entanglement (q = 0…n_count-1)    │
│   4. Inverse QFT         → interference reveals the phase        │
│   5. Measure counting    → classical bitstring encodes φ ≈ s/r   │
└──────────────────────────────────────────────────────────────────┘

Key identity:  U^j |1⟩ = |a^j mod N⟩
The controlled-U^(2^q) operations collectively build the state
    Σ_j  |j⟩|a^j mod N⟩   (over the counting-register superposition)
so that QPE extracts an eigenphase of U, which encodes r.
"""
def build_shor_circuit(a: int, N: int, n_count: int) -> QuantumCircuit:
    n_work = ceil(log2(N + 1))      # qubits needed to store 0 … N-1
    total  = n_count + n_work

    qc = QuantumCircuit(total, n_count, name=f"Shor(N={N}, a={a})")

    # Step 1 — Hadamard superposition on counting register
    qc.h(range(n_count))

    # Step 2 — Set work register to |1⟩  (qubit n_count is the LSB)
    qc.x(n_count)

    # Step 3 — Controlled modular multiplications
    work_qubits = list(range(n_count, total))
    for q in range(n_count):
        gate = ctrl_modmul_gate(a, q, N, n_work)
        qc.append(gate, [q] + work_qubits)  # control=q, targets=work_qubits

    # Step 4 — Inverse QFT on counting register
    qc.append(make_iqft(n_count), range(n_count))

    # Step 5 — Measure counting register into classical bits
    qc.measure(range(n_count), range(n_count))

    return qc


# ─────────────────────────────────────────────────────────────────────────────
# Classical post-processing
# ─────────────────────────────────────────────────────────────────────────────

"""
Use the continued-fractions algorithm to extract the order r from a
QPE measurement bitstring.

The measurement encodes integer  m  where  m / 2^n_count ≈ s / r
for some integer s.  Fraction.limit_denominator(N) recovers a candidate
r' = r / gcd(s, r).  We scan small multiples of r' to find the true r
satisfying  a^r ≡ 1 (mod N).

Returns r (int) or None if decoding fails.
"""
def extract_order(bitstring: str, n_count: int, N: int, a: int):
    m = int(bitstring, 2)
    if m == 0:
        return None

    phase   = m / (2 ** n_count)
    r_cand  = Fraction(phase).limit_denominator(N).denominator
    if r_cand <= 1:
        return None

    # The true order divides φ(N) < N; scan multiples up to that bound
    for k in range(1, N // r_cand + 3):
        r = r_cand * k
        if r > N:
            break
        if pow(a, r, N) == 1:
            return r

    return None


"""
Use the period r to find non-trivial factors of N.
Given  a^r ≡ 1 (mod N)  with r even:
    x  =  a^(r/2) mod N
    If  x ≢ −1 (mod N):
        gcd(x − 1, N)  and  gcd(x + 1, N)  are non-trivial factors.

Returns (p, q) with p * q == N, or None.
"""
def factors_from_order(a: int, r, N: int):
    if r is None or r % 2 != 0:
        return None

    x = pow(a, r // 2, N)

    if x == N - 1:          # a^(r/2) ≡ −1 (mod N) → gcd gives only 1 or N
        return None

    for delta in (-1, +1):
        f = gcd(x + delta, N)
        if 1 < f < N:
            return f, N // f

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Shor's algorithm — main loop
# ─────────────────────────────────────────────────────────────────────────────

"""
Factor N using a simulated Shor's algorithm circuit.
Tries up to 20 random values of  a  (coprime to N), each time:
    •  Running the QPE circuit on AerSimulator
    •  Decoding the top measurements via continued fractions
    •  Testing whether the extracted period yields a factor

Falls back to classical trial division if the quantum part stalls.

Returns (p, q) with p * q == N, or (None, None).
"""
def shors_algorithm(N: int, verbose: bool = True):
    SEP = "─" * 58

    print(f"\n{SEP}")
    print(f"  Factorising  N = {N}")
    print(SEP)

    # ── Trivial shortcuts ────────────────────────────────────────────
    if N % 2 == 0:
        p, q = 2, N // 2
        print(f"  N is even  →  {p} × {q}")
        return p, q

    # Perfect powers: N = k^b
    for b in range(2, ceil(log2(N)) + 1):
        k = round(N ** (1 / b))
        for c in (k - 1, k, k + 1):
            if c > 1 and c ** b == N:
                print(f"  Perfect power: {N} = {c}^{b}")
                return c, N // c

    # ── Circuit dimensions ───────────────────────────────────────────
    n_count = 2 * ceil(log2(N))     # counting register: 2⌈log₂N⌉ qubits
    n_work  = ceil(log2(N + 1))     # work register:      ⌈log₂(N+1)⌉ qubits
    n_total = n_count + n_work

    print(f"\n  Counting qubits : {n_count}")
    print(f"  Work qubits     : {n_work}")
    print(f"  Total qubits    : {n_total}")
    print(f"  Shots per run   : 2048\n")

    backend   = AerSimulator()
    shots     = 2048
    tried_a: set = set()

    for attempt in range(1, 21):

        # ── Pick a random  a  that is coprime to N ───────────────────
        a = None
        for _ in range(300):
            cand = random.randint(2, N - 1)
            if cand not in tried_a and gcd(cand, N) == 1:
                a = cand
                break
        if a is None:
            print("  Could not find a suitable value of a. Aborting.")
            break
        tried_a.add(a)

        # Lucky shortcut: a happens to share a factor with N
        g = gcd(a, N)
        if 1 < g < N:
            print(f"  Lucky! gcd({a}, {N}) = {g}  →  {g} × {N // g}")
            return g, N // g

        print(f"  [Attempt {attempt:2d}]  a = {a}")

        # ── Build and run the QPE circuit ────────────────────────────
        try:
            qc      = build_shor_circuit(a, N, n_count)
            qc_t    = transpile(qc, backend, optimization_level=0)
            counts  = backend.run(qc_t, shots=shots).result().get_counts()
        except Exception as exc:
            print(f"    ⚠  Simulation error: {exc}")
            continue

        # ── Decode measurement outcomes ──────────────────────────────
        top_results = sorted(counts.items(), key=lambda kv: -kv[1])

        for idx, (bitstring, freq) in enumerate(top_results):
            r     = extract_order(bitstring, n_count, N, a)
            phase = int(bitstring, 2) / (2 ** n_count)

            if verbose and idx < 6:         # print the 6 most-frequent outcomes
                tag = f"r = {r}" if r else "r = ?"
                print(f"    {bitstring}  "
                      f"phase = {phase:.4f}  "
                      f"{tag:<12}  "
                      f"(×{freq} shots)")

            result_pair = factors_from_order(a, r, N)
            if result_pair:
                p, q = result_pair
                print(f"\n  ✓  Factors found:  {N} = {p} × {q}\n")
                return p, q

    # ── Classical fallback (trial division) ─────────────────────────
    print("\n  ⚠  Quantum circuit did not converge.")
    print("     Falling back to classical trial division …\n")
    for i in range(3, int(N ** 0.5) + 1, 2):
        if N % i == 0:
            print(f"  ✓  {N} = {i} × {N // i}  (classical)\n")
            return i, N // i

    print("  ✗  Factorisation failed.\n")
    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Command-line interface
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_SEMIPRIMES = [15, 21, 33, 35, 51, 55, 65, 77, 85, 91]

BANNER = r"""
╔════════════════════════════════════════════════════════════╗
║      SHOR'S ALGORITHM  —  Quantum Factorisation            ║
║      Simulated with Qiskit Aer                             ║
╠════════════════════════════════════════════════════════════╣
║  Factors a semiprime  N = p × q  using a quantum           ║
║  phase-estimation circuit executed on your local CPU.      ║
╚════════════════════════════════════════════════════════════╝
"""


def main() -> None:
    print(BANNER)
    print(f"  Suggested inputs : {SAMPLE_SEMIPRIMES}")
    print("  (Larger N → deeper circuit → longer simulation.)\n")

    while True:
        raw = input("  Enter a semiprime  (or 'q' to quit): ").strip()

        # ── Quit ──────────────────────────────────────────────────────
        if raw.lower() in ("q", "quit", "exit"):
            print("\n  Goodbye!\n")
            sys.exit(0)

        # ── Parse ─────────────────────────────────────────────────────
        try:
            N = int(raw)
        except ValueError:
            print(f"  ✗  '{raw}' is not a valid integer.\n")
            continue

        if N < 4:
            print("  ✗  Please enter a number ≥ 4.\n")
            continue

        # ── Validate semiprime ────────────────────────────────────────
        if not is_semiprime(N):
            divs = [i for i in range(2, N) if N % i == 0]
            print(f"  ✗  {N} is not a semiprime "
                  f"(must be a product of exactly two primes).")
            if divs:
                print(f"     Proper divisors of {N}: {divs}")
            else:
                print(f"     {N} appears to be prime — no proper divisors.")
            print()
            continue

        # ── Run Shor's algorithm ──────────────────────────────────────
        p, q = shors_algorithm(N, verbose=True)

        print("  " + "═" * 54)
        if p and q and p * q == N:
            print(f"  RESULT :  {N}  =  {p}  ×  {q}  ✓")
        else:
            print(f"  RESULT :  could not factor  {N}")
        print("  " + "═" * 54 + "\n")

        # ── Ask to continue ───────────────────────────────────────────
        if input("  Factor another number? (y / n): ").strip().lower() != "y":
            print("\n  Goodbye!\n")
            break
        print()


if __name__ == "__main__":
    main()
