"""Group Z_n^2 ⋊ C_4 (order 4n²).

The semidirect product of the translation group Z_n × Z_n with the cyclic
rotation group C_4, where the generator g of C_4 acts on Z_n² via the
90° rotation matrix A = ((0,-1),(1,0)). The homomorphism φ: C_4 → Aut(Z_n²) = GL(2, Z_n)
is given by φ(g) = A, φ(g^k) = A^k for k ∈ {0, 1, 2, 3}.

Composition rule:
    ((x1,y1), r1) * ((x2,y2), r2) = ((x1,y1) + A^{r1}(x2,y2) mod n, r1+r2 mod 4)

Element indexing (0 .. 4n²-1):
    index = r * n² + x * n + y
    where (x, y) ∈ Z_n² and r ∈ {0, 1, 2, 3}.
"""

import numpy as np

from src.groups.group import Group
from src.groups.irrep import IrreducibleRepresentation


class ZnxZnxC4Group(Group):
    """Z_n^2 ⋊ C_4: translations and 90° rotations on the discrete n×n toroidal grid.

    Parameters
    ----------
    n : int
        Order of each cyclic translation factor (n >= 2).
    """

    def __init__(self, n: int):
        if n < 2:
            raise ValueError(f"n must be >= 2, got {n}")
        self._n = n
        self._order_val = 4 * n * n
        self._irreps = self._build_irreps()
        self._regular = self._build_regular_rep()

    @property
    def order(self) -> int:
        return self._order_val

    def elements(self) -> list[int]:
        return list(range(self._order_val))

    def irreps(self) -> list[IrreducibleRepresentation]:
        return list(self._irreps)

    def regular_rep(self) -> np.ndarray:
        return self._regular

    def _encode(self, x: int, y: int, r: int) -> int:
        """Bijective encoding: (x, y, r) → flat index."""
        n = self._n
        return r * n * n + x * n + y

    def _decode(self, idx: int) -> tuple[int, int, int]:
        """Bijective decoding (inverse of _encode): flat index → (x, y, r)."""
        n = self._n
        r, rem = divmod(idx, n * n)
        x, y = divmod(rem, n)
        return x, y, r

    def _apply_rotation(self, r: int, x: int, y: int) -> tuple[int, int]:
        """Apply A^r to (x, y) mod n."""
        n = self._n
        if r % 4 == 0:
            return x, y
        if r % 4 == 1:
            return (-y) % n, x
        if r % 4 == 2:
            return (-x) % n, (-y) % n
        return y, (-x) % n

    def _cayley(self, g: int, h: int) -> int:
        """Product g * h in the group."""
        x1, y1, r1 = self._decode(g)
        x2, y2, r2 = self._decode(h)
        x2r, y2r = self._apply_rotation(r1, x2, y2)
        n = self._n
        return self._encode((x1 + x2r) % n, (y1 + y2r) % n, (r1 + r2) % 4)

    def _compute_char_orbits(self):
        """Partition Z_n² characters into orbits under the C_4 dual action.

        The generator g acts on a character (j1, j2) as g·(j1, j2) = (-j2, j1).
        Returns (fixed_points, size2_orbits, size4_orbits) where each orbit
        is a list of (j1, j2) tuples ordered by successive g-applications.
        """
        n = self._n
        visited = set()
        fixed_points = []
        size2_orbits = []
        size4_orbits = []

        for j1 in range(n):
            for j2 in range(n):
                if (j1, j2) in visited:
                    continue
                orbit = []
                a, b = j1, j2
                for _ in range(4):
                    if (a, b) not in visited:
                        orbit.append((a, b))
                        visited.add((a, b))
                    a, b = (-b) % n, a

                if len(orbit) == 1:
                    fixed_points.append(orbit[0])
                elif len(orbit) == 2:
                    size2_orbits.append(orbit)
                else:
                    size4_orbits.append(orbit)

        return fixed_points, size2_orbits, size4_orbits

    def _build_irreps(self) -> list[IrreducibleRepresentation]:
        """Construct all irreps via the Mackey little-group method.

        Orbit type → irreps:
          fixed point (stabiliser C_4) → 4 one-dimensional irreps
          size-2 orbit (stabiliser C_2) → 2 two-dimensional irreps
          size-4 orbit (trivial stab.)  → 1 four-dimensional irrep
        """
        n = self._n
        order = self._order_val
        omega = np.exp(2j * np.pi / n)
        zeta = 1j  # e^{2πi/4}

        fixed_pts, size2_orbits, size4_orbits = self._compute_char_orbits()
        irreps: list[IrreducibleRepresentation] = []

        # --- 1D irreps from fixed-point characters ---
        # ρ(x,y,r) = ω^{j1·x + j2·y} · ζ^{s·r}
        for fp_idx, (j1, j2) in enumerate(fixed_pts):
            for s in range(4):
                mats = np.empty((order, 1, 1), dtype=np.complex128)
                for idx in range(order):
                    x, y, r = self._decode(idx)
                    mats[idx, 0, 0] = omega ** (j1 * x + j2 * y) * zeta ** (s * r)
                name = f"ZnC4_n{n}|1d_fp{fp_idx}_s{s}"
                irreps.append(IrreducibleRepresentation(name, mats))

        # --- 2D irreps from size-2 orbits (even n only) ---
        # Coset reps of C_4 / C_2: {1, g}.
        # ρ(x,y,r)_{ij} = δ_{i,(r+j)%2} · λ_i(x,y) · (-1)^{s·(r+j-i)/2}
        for orb_idx, orbit in enumerate(size2_orbits):
            for s in range(2):
                mats = np.zeros((order, 2, 2), dtype=np.complex128)
                for idx in range(order):
                    x, y, r = self._decode(idx)
                    for j in range(2):
                        i = (r + j) % 2
                        ai, bi = orbit[i]
                        k = (r + j - i) // 2
                        mats[idx, i, j] = omega ** (ai * x + bi * y) * (-1) ** (s * k)
                name = f"ZnC4_n{n}|2d_orb{orb_idx}_s{s}"
                irreps.append(IrreducibleRepresentation(name, mats))

        # --- 4D irreps from size-4 orbits ---
        # ρ(x,y,r)_{ij} = δ_{i,(r+j)%4} · λ_i(x,y)
        for orb_idx, orbit in enumerate(size4_orbits):
            mats = np.zeros((order, 4, 4), dtype=np.complex128)
            for idx in range(order):
                x, y, r = self._decode(idx)
                for j in range(4):
                    i = (r + j) % 4
                    ai, bi = orbit[i]
                    mats[idx, i, j] = omega ** (ai * x + bi * y)
            name = f"ZnC4_n{n}|4d_orb{orb_idx}"
            irreps.append(IrreducibleRepresentation(name, mats))

        return irreps

    def _build_regular_rep(self) -> np.ndarray:
        """Assigns a permutation matrix to each group element."""
        n_elem = self._order_val
        reg = np.zeros((n_elem, n_elem, n_elem))
        for g in range(n_elem):
            for h in range(n_elem):
                i = self._cayley(g, h)
                reg[g, i, h] = 1.0
        return reg
