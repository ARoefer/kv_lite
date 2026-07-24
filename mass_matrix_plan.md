# Implementation Plan: Mass Matrix Support in `kv_lite`

## Goal

Add joint-space mass matrix computation to `kv_lite`'s `Graph`, exposed as a symbolic
query analogous to `get_fk`. The result is an `n x n` symbolic `KVArray` where `n` is the
number of symbols the caller passes in.

Target API:

```python
def get_mass_matrix(self, symbols, source_frame: str = 'world') -> gm.KVArray
```

## Context for the agent

`kv_lite` models articulated structures as a **directed forest of frames**. Key facts you
must respect:

- **There are no explicit joints.** Degrees of freedom are *symbols* appearing inside edge
  transform expressions. The public interface therefore takes a list of symbols, not joint
  names. Do not introduce a joint abstraction.
- `Graph._incoming_edges` maps `child_name -> DirectedEdge`. There is **no** child index;
  the graph can only be walked *upward* today.
- `Graph._get_path(start, end)` returns the ordered edge chain from `start` up to `end`.
- `Body` is a `Frame` subclass carrying an `Inertial` dataclass with fields `origin`
  (4x4 homogeneous `KVArray`), `mass` (float), and `moments` (3x3 `KVArray`, the inertia
  tensor about the CoM, expressed in the frame defined by `origin`, per URDF convention).
- All expressions expose a `.symbols` attribute.

The underlying math is the Jacobian formulation:

```
M = sum_i  J_i^T  I_i  J_i
```

where `J_i` is the 6xn body Jacobian of body `i` and `I_i` is its 6x6 spatial inertia,
both expressed in `source_frame`.

## Conventions (fix these first, then never deviate)

- **Twist ordering: `[omega; v]`** — angular first, linear second. Matches Featherstone.
  Every 6-vector, every 6x6 block layout, and every Jacobian column must agree.
- **Spatial inertia is taken about each body's own frame origin** — the same point the
  Jacobian's linear part refers to. **This corrects the plan as originally written** (it
  said "about the `source_frame` origin"): `jacobian_column` produces the *hybrid* geometric
  twist whose linear part is the velocity of the body **frame origin**, so the inertia's
  parallel-axis offset must be the CoM position *relative to that frame origin*, not the
  absolute CoM in the source frame. Pairing a hybrid Jacobian with a source-origin-referenced
  inertia double-counts the offset (verified: it inflates a point-mass `M` by a factor of 4).
  Each body's `J^T I J` term is internally consistent at its own frame origin; the summed
  mass matrix is frame-independent, so no common reference point is required.
- Symbol ordering in the output matrix follows the order of the caller's `symbols` list.

---

## Phase 0 — Verify assumptions in `spatial.py`  ✅ DONE

**Do this before writing any code.** Everything downstream depends on the answers.

All three questions were verified empirically against the real code (not just read).
`gm` is the `spatial` module (`from . import spatial as gm`), which re-exports everything
from `math.py`. `KVArray` is an `np.ndarray` subclass wrapping CasADi expressions.

### Findings

1. **Differentiation.** There is **no `gm.diff`**. The real API is
   `KVArray.jacobian(symbols)` (delegating to `KVExpr.jacobian` → `ca.jacobian`). It runs
   element-wise and **appends** the symbol axis, then squeezes it: for a 4x4 `fk`,
   `fk.jacobian([s])` returns a **`(4, 4)`** matrix (trailing length-1 axis squeezed), and
   `fk.jacobian([a, b])` returns `(4, 4, 2)`. So the plan's `gm.diff(fk, symbol)` maps to
   `fk.jacobian([symbol])`. We add a thin `diff(expr, symbol)` wrapper for readability.
   Note: a fk that depends on *no* symbols still yields a `(4, 4)` zero block, so
   single-symbol diff is always shape-safe.

2. **Edge symbol access.** `DirectedEdge` (graph.py) exposes only `eval()`.
   `TransformEdge`/`ConstrainedTransformEdge` (model.py) store the transform privately as
   `self._transform`, which **is** a `KVArray` and exposes `.symbols`. Per plan, we add a
   `symbols` property to the `DirectedEdge` base (returns `frozenset()`) and override it in
   the transform edges to return `self._transform.symbols`. (Done in Phase 4.)

3. **Construction helpers.** `zeros`, `eye`, `ones`, `diag`, `batched_eye`, `array`,
   `vstack`, `hstack`, `stack`, `concatenate` all exist. `@` and `.dot` both work on
   symbolic arrays (`J.T.dot(I6).dot(J)` yields a symbolic result). Indexed **assignment**
   (`M[:3,:3] = ...`, `J[:,i] = ...`) works — **but only on `dtype=object` arrays**.

### Corrections folded into downstream phases

- **`gm.skew` / `gm.unskew` do not exist publicly.** Batched, correct implementations live
  in `lie.py` as `_skew`/`_unskew` (`unskew(skew(v)) == v` verified). Per the user's
  instruction these move **into `spatial.py`** as public `skew`/`unskew` (right altitude:
  spatial owns the geometric helpers; no circular import since `spatial` only imports
  `.math`). `lie.py` then imports them from there. Done in Phase 1.
- **Accumulators must be object-dtype.** `gm.zeros((6,6))` defaults to float, and symbolic
  indexed assignment into a float array fails. Every `M6`/`M`/`J` accumulator is created as
  `gm.zeros(shape, dtype=object)` — the same pattern `Transform.inverse` and `lie._skew`
  already use.
- **Twists are built with `hstack`, not `vstack`.** The plan's `jacobian_column` used
  `gm.vstack([w, v])`; `vstack` of two `(3,)` vectors gives `(2, 3)`. The flat `(6,)` twist
  needs `gm.hstack([w, v])` (verified), which then assigns correctly into `J[:, i]`. Fixed
  in the Phase 3 sketch.

**Outcome:** algorithm is sound; indexed assignment works, so **no block-stacking fallback
is needed**. Three mechanical fixes (`diff` = `.jacobian([s])`, object-dtype accumulators,
`hstack` for twists) plus the skew/unskew relocation.

**Estimate:** 1 hour. *(actual: verification done, incl. empirical checks.)*

---

## Phase 1 — Spatial math primitives

Add to `spatial.py`. Pure math, no graph dependency, fully unit-testable in isolation.

```python
def skew(v)   -> gm.KVArray  # 3-vector -> 3x3 skew-symmetric
def unskew(S) -> gm.KVArray  # 3x3 skew-symmetric -> 3-vector (vee operator)
def adjoint(T) -> gm.KVArray # 4x4 homogeneous -> 6x6 adjoint, [omega; v] ordering
def spatial_inertia(inertial: Inertial, ref_T_body: gm.KVArray) -> gm.KVArray  # -> 6x6
```

Reference implementation of `spatial_inertia` (**corrected**: offset from the body frame
origin `d`, not the absolute CoM `c` — see Conventions; and object-dtype accumulator per
Phase 0):

```python
def spatial_inertia(inertial, ref_T_body):
    m = inertial.mass
    ref_T_com = ref_T_body.dot(inertial.origin)
    R = ref_T_com[:3, :3]
    d = ref_T_com[:3, 3] - ref_T_body[:3, 3]   # CoM offset from body frame origin, ref frame

    I  = R.dot(inertial.moments).dot(R.T)   # inertia about CoM, rotated into ref frame
    dx = skew(d)

    M6 = gm.zeros((6, 6), dtype=object)
    M6[:3, :3] = I + m * dx.dot(dx.T)   # parallel-axis shift from CoM to frame origin
    M6[:3, 3:] = m * dx
    M6[3:, :3] = m * dx.T
    M6[3:, 3:] = m * gm.eye(3)
    return M6
```

**Acceptance criteria:**

- `unskew(skew(v)) == v` for random `v`.
- `spatial_inertia` output is symmetric.
- For a body with identity `origin` at the reference origin, the top-left block equals
  `moments` and the bottom-right equals `m * I3`.

**Estimate:** 2–3 hours.

---

## Phase 2 — Subtree traversal on `Graph`

The graph can only walk upward. Add downward traversal:

```python
def _get_children_index(self) -> dict[str, list[str]]:
    children = {}
    for child, edge in self._incoming_edges.items():
        children.setdefault(edge.parent, []).append(child)
    return children

def get_subtree(self, root: str) -> list[str]:
    """All frame names in the subtree rooted at `root`, inclusive."""
```

Build the index **on demand**. Do *not* maintain an `_outgoing_edges` structure
incrementally at this stage — it adds invalidation burden in `add_edge`/`remove_edge` for
no measured benefit. Revisit only if profiling shows it matters.

**Acceptance criteria:**

- `get_subtree(root_node)` returns every connected frame.
- Leaf nodes return `[leaf]`.
- Raises `KeyError` for unknown frames, consistent with the rest of `Graph`.

**Estimate:** 1 hour.

---

## Phase 3 — Body Jacobians (correctness-critical)

**This is the phase to slow down on.** A wrong Jacobian still yields a symmetric,
positive-definite matrix that looks plausible but produces wrong dynamics.

`gm.diff(fk, symbol)` on a 4x4 homogeneous transform returns a **4x4 derivative matrix**,
not a twist. Extract the 6-vector:

```python
def jacobian_column(fk, symbol):
    dT = gm.diff(fk, symbol)         # 4x4
    R  = fk[:3, :3]
    v  = dT[:3, 3]                   # linear part, taken directly
    w  = unskew(dT[:3, :3].dot(R.T)) # angular part via vee(R_dot @ R^T)
    return gm.vstack([w, v])         # [omega; v]
```

The `R_dot @ R^T` form yields the **spatial** angular velocity (expressed in the reference
frame), which is required for consistency with the spatial inertia from Phase 1. Using
`R^T @ R_dot` instead gives the *body* angular velocity and is wrong here.

Then:

```python
def body_jacobian(self, body_name, symbols, source_frame='world') -> gm.KVArray  # 6xn
```

**Acceptance criteria:**

- Single revolute DOF about world Z, point at `(l, 0, 0)`: the Jacobian column must be
  `[0, 0, 1, 0, l, 0]` under `[omega; v]` ordering.
- Columns for symbols the body's FK does not depend on are exactly zero.
- Prismatic DOF along an axis produces zero angular part.

**Estimate:** 3–4 hours.

---

## Phase 4 — `get_mass_matrix`

```python
def get_mass_matrix(self, symbols, source_frame: str = 'world') -> gm.KVArray:
```

Algorithm:

1. **Guard the reference frame.** Compute `get_fk(source_frame).transform.symbols` and
   intersect with the requested symbol set. If nonempty, raise — see caveats below.
2. **Find active edges.** Walk `self._incoming_edges.values()`; for each edge, intersect
   `edge.symbols` with the requested set. Skip edges with an empty intersection.
3. **Flood down.** For each active edge, call `get_subtree(edge.child)` and accumulate into
   `dict[body_name -> set[symbol]]`, keeping only `Body` instances.

   **Deduplicate by body.** A body deep in the tree lies beneath several active edges. It
   must contribute exactly one `J^T I J` term, using the *union* of its active symbols.
   Accumulating once per edge is the most likely bug in this phase.
4. **Per body:** one `get_fk(body_name, source_frame)` call, one `spatial_inertia`, one
   `body_jacobian` restricted to that body's active symbols. Then `M += J.T @ I6 @ J`.
5. **Symmetrize:** `M = 0.5 * (M + M.T)` before returning. Cheap insurance against
   asymmetry introduced during symbolic expansion.

Sketch:

```python
symbol_set = set(symbols)
sym_index  = {s: i for i, s in enumerate(symbols)}
n = len(symbols)

body_symbols = {}
for edge in self._incoming_edges.values():
    active = symbol_set & edge.symbols
    if not active:
        continue
    for name in self.get_subtree(edge.child):
        if isinstance(self._nodes[name], Body):
            body_symbols.setdefault(name, set()).update(active)

M = gm.zeros((n, n))
for name, active in body_symbols.items():
    body = self._nodes[name]
    fk   = self.get_fk(name, source_frame).transform
    I6   = spatial_inertia(body.inertial, fk)
    J    = gm.zeros((6, n))
    for s in active:
        J[:, sym_index[s]] = jacobian_column(fk, s)
    M = M + J.T.dot(I6).dot(J)

return 0.5 * (M + M.T)
```

**Acceptance criteria:**

- Empty `symbols` returns a 0x0 matrix without error.
- Symbols not appearing anywhere in the graph produce zero rows/columns rather than an
  error.
- A graph with no `Body` instances returns a zero matrix.

**Estimate:** 2–3 hours.

---

## Phase 5 — Validation  ✅ DONE

All checks live in `test/test_mass_matrix.py` (runnable directly or under pytest; the
PyBullet check skips cleanly when PyBullet is absent). In increasing order of strength:

1. **Point mass, one revolute DOF at distance `l`.** `M == m * l**2`. ✅ Passes — **and this
   is the check that caught the factor-4 spatial-inertia bug** (wrong reference point; see
   Phase 1 / Conventions). Also verified with the CoM offset supplied via `inertial.origin`,
   which exercises the parallel-axis term independently.
2. **Two-link planar arm.** Matches the closed-form textbook `M(q)`; off-diagonal coupling
   confirmed to depend on `q2` (checked via a nonzero range over random `q2`). ✅
3. **Symmetry and positive definiteness** over 20 random configs of a random 4-link tree
   (random axes, masses, SPD inertias, CoM offsets). ✅
4. **Energy consistency (strongest self-contained).** `0.5 q̇ᵀ M q̇` equals the sum of
   per-body `0.5 m|v|² + 0.5 ωᵀ I ω`, with `v`, `ω` from independently differentiated FK.
   ✅ Matches to machine precision (diff ~1e-15) — validates the whole pipeline with no
   external reference.
5. **Cross-check against PyBullet** on the KUKA iiwa (clean 7R): **matches to ~9e-16**. ✅
   Two gotchas discovered, both documented in the test:
   - **`URDF_USE_INERTIA_FROM_FILE`** — without this flag PyBullet *ignores* the URDF
     `<inertia>` and recomputes it from collision geometry, producing a ~2% mismatch that
     looks like an algorithm bug but is not.
   - **Detached root** — the loaded URDF's root link is not connected to `world`, so
     `get_mass_matrix` must be called with `source_frame=uobj.root`. This also drove a fix
     to the Phase 4 reference-frame guard: it now walks the incoming-edge chain upward
     instead of doing a `get_fk(world, source)` lookup, which fails for a world-detached
     root.
   - Panda (with mimic-coupled fingers) matches on the 7-DOF **arm block** to ~2e-15; the
     full 9×9 differs only because PyBullet treats the mimic finger as an independent DOF
     while kv_lite's URDF loader couples it — a modeling difference, not a mass-matrix bug.

   `_parse_inertial` was confirmed to read off-diagonal inertia terms (`ixy/ixz/iyz`)
   correctly, so it was not implicated.

**Also fixed in this phase:** `URDFJoint` (a `ConstrainedEdge`, not a `TransformEdge`) was
missing the `symbols` property added in Phase 4, so URDF-loaded models produced an all-zero
mass matrix. Added `URDFJoint.symbols` (joint variable / mimic-expression symbols + any
symbolic origin/axis). This is the third and last transform-bearing edge type.

**Estimate:** 1 day. *(actual: done, incl. PyBullet cross-check.)*

---

## Phase 6 — Extensions  ✅ DONE

Because everything is symbolic, the rest of rigid-body dynamics follows by differentiation.

- **Gravity vector** — `Graph.get_gravity_vector(symbols, gravity=(0,0,-9.81), source_frame)`.
  `G = dU/dq`, `U = -sum_i m_i * g . p_i`. ✅ Matches PyBullet inverse dynamics (q'=q''=0)
  to ~1e-14, and a self-contained finite-difference-of-`U` check. *Impl note:* the summed
  potential is a scalar `KVExpr` (not a `KVArray`), so the "no bodies" guard must accept
  both types — the first cut returned zeros because of this.
- **Coriolis/centrifugal** — `Graph.get_coriolis_matrix(symbols, velocities, source_frame)`.
  Christoffel symbols of the first kind,
  `C[i,j] = sum_k 0.5*(dM[i,j]/dq_k + dM[i,k]/dq_j - dM[j,k]/dq_i) * q'_k`, built from the
  `n**3` derivative tensor `M.jacobian(symbols)`. ✅ Validated together with gravity against
  PyBullet inverse dynamics (q''=0 ⇒ `tau = C q' + G`) to ~1e-14 — which, with the separate
  mass-matrix check, validates the full manipulator equation `M q'' + C q' + G = tau`.
- **Operational-space inertia** — `spatial.operational_space_inertia(M, J)` returning
  `(J M^-1 J^T)^-1` for a target frame's Jacobian. **Numeric, not symbolic**: the backend has
  no general matrix inverse (only `Transform.inverse` for 4x4 homogeneous transforms), and a
  symbolic `M^-1` would blow up anyway — so evaluate `M` and `J = body_jacobian(target, ...)`
  at a configuration, then call this. Requires the task Jacobian to have full row rank (≥ 6
  DoF for a 6-D task, else `J M^-1 J^T` is singular). ✅ Validated for symmetry, PD, and the
  defining identity on a 7-DoF tree.

All three live in `test/test_mass_matrix.py`.

---

## Caveats to document in the docstring

**Non-inertial `source_frame`.** If the source frame's own pose depends on any of the
requested symbols, the result describes relative motion and is not a valid rigid-body mass
matrix — fictitious-force terms are missing. Phase 4 step 1 raises on this case; say so
explicitly in the docstring rather than silently returning a wrong matrix.

**Expression size.** `M` has `n**2` configuration-dependent entries, each a product of
Jacobian terms. Fine for a 7-DOF arm; symbolic expansion may blow up for a 30-DOF
humanoid. The fix, if needed, is CRBA — the composite-inertia formulation shares
subexpressions up the tree instead of recomputing per body. **Build the naive `J^T I J`
version first regardless:** it is the reference implementation that a future CRBA
implementation must be validated against.

---

## Effort summary

| Phase | Task | Estimate |
|---|---|---|
| 0 | Verify `gm` API | 1 hr |
| 1 | Spatial primitives | 2–3 hrs |
| 2 | Subtree traversal | 1 hr |
| 3 | Body Jacobians | 3–4 hrs |
| 4 | `get_mass_matrix` | 2–3 hrs |
| 5 | Validation | 1 day |
| 6 | Extensions | optional, 1 day |

**Core feature: 2–3 days.** Phases 0–5 are the deliverable; Phase 6 is follow-up work.

