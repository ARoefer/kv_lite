# kv_lite (Kineverse 2.0) — reference for agents

Symbolic, differentiable kinematics on top of CasADi and numpy. Build articulated structures
(robots, doors, drawers, arbitrary closed-form mechanisms) as a frame graph, pull analytic FK
expressions out of it, differentiate them, and hand them to a QP controller or an NLP solver.

**Read this file instead of the source.** Every snippet and every claim below was executed against
this tree. The last two sections list what is broken and where `README.md` is wrong — check them
before debugging anything surprising.

## Environment

- Run everything with `pyenv/bin/python` (at the `lh_ws` root). The package is installed
  **editable**, so `import kv_lite` already uses `src/kv_lite/src/kv_lite`.
- `import kv_lite as kv` is the whole import surface for the core. Sub-namespaces: `kv.gm`
  (= `spatial`, the math namespace), `kv.urdf` (= `urdf_utils`), `kv.splines`, `kv.ros` (only if
  `rospy` imports; prints "No ROS found." otherwise — that line is harmless).
  **`kv.exp` is *not* the `exp_utils` module** — `__init__.py` binds it and then `from .spatial
  import *` overwrites it with the elementwise exponential. Use `import kv_lite.exp_utils`.
- **Not** re-exported at top level — import these explicitly:
  `from kv_lite.control import QPController, Objective, ObjectiveTypes`,
  `from kv_lite.collisions import ...`, `from kv_lite.rerun import ModelRerunBroadcaster`.
- Optional deps, all present in this venv: `qpsolvers` (control), `robotic` (RAI NLP solver),
  `rerun`, `pyvis` (expression graphs), `scipy`/`jinja2` (ROS utils).

## The four building blocks

| Thing | Class | Note |
|---|---|---|
| Variable | `KVSymbol` | interned: same name ⇒ **same object**, globally, for the whole process |
| Expression | `KVExpr` | wraps a CasADi `SX` |
| Matrix | `KVArray` | `np.ndarray` subclass, `dtype=object` when symbolic |
| Constraint | `Constraint(lb, ub, expr)` | means `lb <= expr <= ub` |

```python
import kv_lite as kv

a, b = [kv.Symbol(x) for x in 'ab']   # NOTE: kv.symbol (lowercase) does not exist
e = kv.cos(a * 4) + b

e.is_symbolic      # True          e.symbols -> frozenset({KV(a), KV(b)})
e.eval({a: 2, b: 1})               # 0.854...   extra keys in the dict are ignored
e.jacobian([a, b])                 # KVArray (1, 2)
e.substitute({b: a**2})            # KVExpr
float(kv.expr(4) * kv.expr(5))     # 20.0 — float() raises on symbolic expressions
```

`KVArray` is a real numpy array: indexing, slicing, `.T`, `.reshape`, `@`, `hstack`/`vstack` all
work as usual. Constructors mirroring numpy: `kv.array`, `kv.asarray`, `kv.eye`, `kv.zeros`,
`kv.ones`, `kv.diag`, `kv.tri`, `kv.stack`, `kv.concatenate`, `kv.batched_eye`, `kv.diag_view`.

```python
m = kv.diag([1, 2, 3])              # left-hand side decides the type -> KVArray
v = kv.array([a, b, a]).reshape((3, 1))
r = m @ v
r.symbols; r.is_symbolic; r.jacobian([a, b]).shape   # (3, 1, 2)
```

### Math functions — use `kv.*`, never `np.*`

`np.cos(expr)` raises `TypeError: loop of ufunc does not support argument 0 of type KVExpr`.
`kv.sin/cos/tan/asin/acos/atan/atan2/sinh…/exp/log/sqrt/abs` dispatch to CasADi for symbols and to
numpy for floats (and return a plain `float` for float input). They broadcast over arrays.

`min`/`max` are a special case: **`arr.min()` / `np.min(arr)` raise on symbolic arrays** by design.
Use `kv.min(arr, axis=...)` / `kv.max(arr, axis=...)`, which fold to CasADi `fmin`/`fmax`.

## Evaluation, compilation and symbol order

Two evaluation paths. Both lazily compile the expression into a CasADi `Function` on first use and
cache it on the object.

```python
expr.eval({sym: value, ...})          # keyword-ish, safe, raises EvaluationError on a missing symbol
expr.unchecked_eval(np_array)         # positional, fast, batched — NO checks
expr(x)                               # dispatches: dict -> eval, else -> unchecked_eval
```

`unchecked_eval` broadcasts: an input of shape `(..., N_args)` returns `(..., *expr.shape)`, so
evaluating a `(4, 4)` FK over 500 configurations is one call returning `(500, 4, 4)`.

> **The single most dangerous gotcha in the library.** `unchecked_eval` consumes arguments in
> `expr.ordered_symbols` order. For a **`KVExpr`** that order comes from a `frozenset` and therefore
> **changes between processes** (string hash randomisation) — verified: three runs of the same
> script gave `[d,c,b,a]`, `[a,d,b,c]`, `[d,b,a,c]`. For a `KVArray` it is first-appearance order,
> which is stable but still not necessarily *your* order. Wrong order fails **silently** with wrong
> numbers, not an exception.
>
> Always pin it before batched evaluation:
> ```python
> fk.set_symbol_order(robot.q)          # also invalidates the compiled function
> fk.unchecked_eval(np.zeros((500, len(robot.q))))
> ```
> `set_symbol_order` raises if the given list does not cover every symbol; extra symbols are fine
> (they are filtered out). `gm.VEval(expr, symbols)` wraps this pattern plus masking for a fixed
> global vector layout.

Other caching rules: `+=`/`-=`/`*=`/`/=` on a `KVExpr` invalidate the cache; in-place mutation of a
`KVArray`'s cells does **not** — rebuild the array instead. A non-symbolic `KVArray` evaluates to a
copy of itself for any argument.

## Symbol typing (position / velocity / …)

```python
a = kv.Position('a')          # -> a__position;  also Velocity, Acceleration, Jerk, Snap
a.derivative()                # a__velocity      .integral() goes back
(4*a - b).tangent()           # (4*a__velocity - b__velocity)   == J(q) · q̇
```

- `Symbol(...)` (untyped) **cannot** be differentiated or integrated — `RuntimeError`. Use
  `Position` for anything that is a degree of freedom.
- `tangent()` on an expression is what turns FK into a velocity-level task Jacobian.
- Namespacing: `kv.Position('j1', prefix='rob')` and `kv.Position(Path('rob/j1'))` both produce
  `rob__j1__position` — `/` is rewritten to `__`, so path names and prefixes collide by design.
- `sym.set_stamp(t)` appends `__t{t}` and is how the trajectory layer identifies time steps.
  `expr.set_stamp(t)` stamps every symbol in an expression at once.

## Spatial types

4×4 homogeneous matrices everywhere. `kv.point3(x,y,z)` has `w=1` (translates), `kv.vector3(x,y,z)`
has `w=0` (does not). `kv.norm`, `kv.cross`, `kv.plane_projection`, `kv.unitX/Y/Z`.

```python
kv.Transform.identity()
kv.Transform.from_xyz(x, y, z)
kv.Transform.from_euler(roll, pitch, yaw)      # applied rz @ ry @ rx; (0,0,π/2) rotates about Z
kv.Transform.from_axis_angle(axis, angle)      # axis is a vector3
kv.Transform.from_quat(x, y, z, w)
kv.Transform.from_xyz_euler(...)               # also from_xyz_aa, from_xyz_quat
kv.Transform.inverse(tf)                       # proper SE3 inverse, not a matrix inverse
kv.Transform.rot(tf); kv.Transform.trans(tf)   # split into pure rotation / pure translation
kv.Transform.pos(tf)                           # (4,1) — same as .w(tf); also .x/.y/.z
```

All of `Transform` is **2-D only** — `inverse`, `pos`, `rot`, `x/y/z/w` index without ellipses and
raise or silently misbehave on a batch of transforms. Chain with `.dot()` / `@`.

## Lie groups — `SO3`, `SE3`

```python
R  = kv.SO3.expmap(w)                 # w: (..., 3) rotation vector -> (..., 3, 3)
w  = kv.SO3.logmap(R)                 # accepts 4x4 too (slices [:3,:3])
T  = kv.SE3.expmap(w, v)              # -> (4, 4)
wv = kv.SE3.logmap(T)                 # -> (6,)  [w | v]
kv.SO3.J_right(w), kv.SO3.J_right_inv(w), kv.SO3.J_left(w)
```

Validated against GTSAM. `SO3.expmap/logmap` are batched; `SE3.expmap/logmap` are **single-transform
only**. Every function takes `epsilon=1e-6` to dodge the singularity at `θ=0`; with non-symbolic
input the guard is applied automatically.

## Models: frames, edges, constraints

A `Model` is an acyclic directed forest. Nodes are `Frame`s (or `Body`s), edges each contribute a
transform, and `get_fk` composes/inverts along the path to the shared root.

> ### **Frame and edge names are `pathlib.Path`, never `str`**
> `Frame.__post_init__` converts its name to a `Path`, and the graph is keyed by those Paths.
> `Graph.get_fk` / `add_edge` do **not** coerce their arguments, and `Path('world') != 'world'`, so
> every string-named lookup dies with `KeyError: 'Target frame "world" is not known.'`
> The whole Models section of `README.md` is written with strings and does not run. Pass Paths.

```python
from pathlib import Path
import kv_lite as kv

W  = Path('world')                    # 'world' is the default root, and it is a Path
km = kv.Model()
km.add_frame(kv.Frame('lol'))         # constructor DOES convert -> Path('lol')

a, b = [kv.Position(x) for x in 'ab']
km.add_edge(kv.TransformEdge(W, Path('lol'), kv.Transform.from_xyz(a + 1, 0, 0)))

km.add_frame(kv.Frame('foo'))
km.add_edge(kv.ConstrainedTransformEdge(Path('lol'), Path('foo'),
                                        kv.Transform.from_euler(0, b, 0) @ kv.Transform.from_xyz(0, 0, 1),
                                        {Path('limit b'): kv.Constraint(-0.8, 0.8, b)}))

view = km.get_fk(Path('foo'))                  # default source is Path('world')
view.transform, view.name, view.reference      # FrameView: KVArray + both frame names
km.get_fk(Path('world'), Path('foo'))          # the inverse chain

km.get_constraints(view.transform.symbols)     # {Path('limit b'): C(-0.8 <= b__position <= 0.8)}
```

- Edge direction is parent → child; the transform of an edge maps **child coordinates into the
  parent frame**. `get_fk(target, source)` returns `source_T_target`.
- Frames in disconnected trees raise `FKChainException` (a subclass of `Exception`, *not* of
  `KeyError`); unknown names raise `KeyError`. Catch both.
- Constraints attached to a `ConstrainedEdge` are added/removed with the edge, and
  `get_constraints(symbols)` returns only the ones touching those symbols — that is the intended way
  to discover joint limits for whatever expression you just built.
- Other graph API: `get_frames`, `get_frame`, `has_frame`, `remove_frame`, `get_edge(name)`,
  `has_edge`, `get_edges`, `get_incoming_edge`, `get_root(start, filter=None, return_path=False)`,
  `reset_root`.
- Payload classes: `Body(name, inertial, geom_collision, geom_visual)` (a `Frame`),
  `Geometry(type, mesh_path, dim_scale, origin)`, `Inertial(origin, mass, moments)`.

## URDF

```python
from pathlib import Path
import kv_lite as kv, prime_bullet as pb
W = Path('world')

km = kv.Model()
with open(pb.res_pkg_path('package://prime_bullet/urdf/windmill.urdf')) as f:
    robot = kv.urdf.load_urdf(km, f.read())     # -> URDFObject; name= only needed if the XML has none

robot.name                 # 'windmill'          robot.root -> Path('windmill/base')
robot.links                # {'base': Body, 'head': Body, 'wings': Body}   local names
robot.joints               # {'head_pan': URDFJoint, ...}   .dynamic_joints drops fixed ones
robot.q                    # [KV(windmill__head_pan__position), ...]
robot.q_dot                # matching velocity symbols
robot.q_limit              # (N, 2) read-only, ±inf for continuous joints; also q_dot_limit
robot.joints_by_symbols    # position symbol -> local joint name
robot.make_full_joint_dict({'head_pan': 0.5})   # partial (by name OR symbol) -> full, clipped state

km.add_edge(kv.TransformEdge(W, robot.root, kv.Transform.from_xyz(1, 0, 0)))   # attach to the world
fk = robot.get_fk('wings', W).transform         # local link names resolve, then global ones
```

- `URDFObject.get_fk(link)` defaults `target='world'` **as a string**, so the one-argument form
  always raises `KeyError`. Pass `Path('world')` explicitly.
- Supported joint types: `revolute`, `continuous`, `prismatic`, `fixed`. Anything else raises.
  `mimic` is supported (resolved by re-queueing) and produces `m * other_position + b`.
- A `revolute`/`prismatic` joint whose `lower == upper` is silently **downgraded to `fixed`**, so it
  disappears from `q`. If a joint you expected is missing, check its limits first.
- URDF constraint names are Paths: `windmill/head_pan/position`, `.../velocity`.
- `robot.add_joints(...)` composes extra edges into the URDF view (e.g. gluing a gripper on).

## Velocity control — `QPController`

Velocity-resolved control: objectives are differentiated w.r.t. the control symbols, the resulting
Jacobians become the QP's `A`/`G`, and each soft objective dimension gets its own weighted slack.

```python
import numpy as np
from kv_lite.control import QPController, Objective, ObjectiveTypes

ee    = kv.Transform.pos(robot.get_fk('wings', W).transform)[:3].reshape((-1,))
error = (goal_point3[:3].reshape((-1,)) - ee)

ctrl = QPController(robot,
                    {'reach': Objective(ObjectiveTypes.EQUALITY, expr=error, gain=-error, weight=1.0)},
                    control_symbols=robot.q,
                    control_costs={},                 # per-symbol overrides
                    default_control_cost=0.01)

q = robot.make_full_joint_dict({'head_pan': 0.0})
for _ in range(100):
    cmd = ctrl.eval(q)                                # {symbol: velocity}
    q   = {s: q[s] + cmd[s] * dt for s in q}
```

- **`gain` is the *desired rate of change* of `expr`, not a scalar P-gain.** It becomes the QP's
  right-hand side (`A x = gain`). To drive `expr → 0` pass `gain=-k*expr` (symbolic, re-evaluated
  each tick). Passing `+error` drives the robot *away* from the goal — verified.
- `expr` is flattened to 1-D internally; **`gain`/`weight` must be scalar or already 1-D of the same
  length**, otherwise `__post_init__` broadcasts to a 2-D array and the QP build fails with a
  confusing `ValueError`.
- `weight=None` ⇒ **hard** constraint (no slack). Any number ⇒ soft. `weight=0` is coerced to `None`,
  i.e. it makes the objective *hard*, not disabled.
- `ctrl.eval()` returns a dict keyed by the **control symbols you passed in** (position symbols),
  even though the values are velocities. Don't look up `s.derivative()`.
- Options: `dampening_factor` (crude acceleration limit), `vel_bound_scale`,
  `position_integration_factor` (seconds per tick; shrinks commands so position limits aren't
  overshot), `regularization_weight`/`regularization_target` (nullspace posture),
  `lambda_damping` (regularise toward the previous command), `solver='daqp'` (any `qpsolvers` name).
- `ctrl.is_satisfied(tol)` tests the magnitude of the **last command**, not the residual error, so it
  reads False while the controller is still converging and True when it has stalled for any reason.
- Introspection: `ctrl.eval_objectives(q)`, `ctrl.last_objective_costs()`, `ctrl.symbols`,
  `ctrl.reset()`. `eval` raises `ValueError` when the QP is infeasible.

## Trajectory optimization — `VectorizedLayout` / `MacroLayout` / `RAI_NLPSolver`

KOMO-style: one `VectorizedLayout` per objective, each replicating an expression over time steps and
producing its sparse Jacobian. Symbols are either **series** (one instance per time step) or
**shared** (one instance for the whole problem). `order=k` builds a k-th order difference over the
series, which is how you get smoothness/velocity/acceleration costs; the extra leading steps it
needs are supplied as `pads`.

```python
import numpy as np
import kv_lite as kv

ee     = kv.Transform.pos(robot.get_fk('wings', W).transform)[:3].reshape((-1,))
target = np.array([1.287, -0.279, 1.42])
T      = 6

goal   = kv.VectorizedLayout(ee,             t_steps=[T-1],           args_series=robot.q,
                             args_shared=[], bias=-target)            # value = weights@expr + bias
smooth = kv.VectorizedLayout(kv.array(robot.q), t_steps=list(range(T)), args_series=robot.q,
                             args_shared=[], order=1)                 # penalise q_t - q_{t-1}

solver = kv.RAI_NLPSolver({'goal':   (kv.SolverObjectives.eq,  goal),
                           'smooth': (kv.SolverObjectives.sos, smooth)},
                          bounds=dict(zip(robot.q, robot.q_limit)),
                          pads=np.zeros(smooth.pad_size))

shared, series, ret = solver.solve(init_sample=np.zeros(T * len(robot.q)), verbose=0)
# series -> (T, N) trajectory; ret.feasible / ret.eq / ret.sos; solver.report(ret.x) per objective
```

- `args_series` must be **non-empty**; an all-shared layout crashes when `MacroLayout` stacks the
  bounds. Put the optimisation variables in `args_series` even for a single-step IK (`t_steps=[0]`).
- `diff_symbols=` restricts which symbols are differentiated; everything else becomes a *constant*
  and its value must then be supplied via `constants=` or the solver raises on construction.
- `bias` is added **after** `weights` are applied, so encode "reach `target`" as `bias=-target`.
- `order > 3` is not implemented. Multiple stamps in one expression must not overlap the order
  window (`Stride overlaps with higher-order`).
- `MacroLayout` alone (no `robotic`) gives you `eval_all(x) -> (values, sparse_J)`, `in_dim`,
  `out_dim`, `in_symbols`, `bounds`, `report(x)` — usable with any other optimiser.
- Without the `robotic` package, `RAI_NLP`/`RAI_NLPSolver` are stubs that raise `NotImplementedError`
  from their constructor and `SolverObjectives` is `None`.

## Everything else

- `kv.splines` — `interpolate_cspline(t, stamps, positions, velocities)` and `retime_spline`,
  `retime_path` (velocity/acceleration-limited retiming). Pure numpy, no symbols.
- `kv_lite.collisions` — `dist_sphere_sphere`, `J_dist_sphere_sphere`. Pure numpy, batched, not
  exported from `__init__`.
- `kv_lite.exp_utils` — `twist_to_se3`, `twist_to_se3_special`, `TwistJointEdge`: screw-axis joints
  for articulations that URDF cannot express. (Reachable only by full import; see Environment.)
- `kv_lite.rerun.ModelRerunBroadcaster(model, ref_frame)` — `.update(q)` logs every reachable frame
  to Rerun. `ref_frame` must be a `Path` (it reads `.parts`).
- `kv.ros` — `ModelTFBroadcaster`, `gen_urdf(model)` (renders a model back out as URDF via
  `data/urdf_template.jinja`). Only importable with `rospy`.
- `kv.generate_expression_graph(exprs, ...)` + `kv.graph_to_html(g, path)` — visualise an
  expression's structure with pyvis.

## Known rough edges — check here first

| Thing | Status |
|---|---|
| `kv.exp` | Is the elementwise exponential, **not** `exp_utils` — `from .spatial import *` shadows the module alias in `__init__.py`. |
| `str` frame/edge names | `KeyError`. The graph is keyed by `pathlib.Path`; only `Frame(...)` coerces. Affects `get_fk`, `add_edge`, `add_frame` lookups. |
| `URDFObject.get_fk(link)` | Its `target='world'` default is a `str` ⇒ always raises. Pass `Path('world')`. |
| `KVExpr.ordered_symbols` | Derived from a `frozenset`; order differs **between processes**. Call `set_symbol_order` before `unchecked_eval`. |
| `symbol == 5` | Returns **`None`**, not `False` (`KVSymbol.__eq__` has no fallback return). Falsy, so `if` works, but `assert (a == 5) is False` fails. `KVExpr` has no `__eq__` at all — comparison is identity, so `(a*4) == (a*4)` is `False`. |
| `ConstrainedEdge(..., constraints=None)` | `Model.add_edge` crashes with `AttributeError: 'NoneType' object has no attribute 'items'`. Pass `{}`. |
| `make_full_joint_dict({})` | `StopIteration` on an empty dict — it peeks at the first key to detect the key type. |
| `Transform.inverse` / `.pos` / `.rot` / `.x/y/z/w` | 2-D only; batched input raises or silently returns nonsense. |
| `SE3.expmap` / `SE3.logmap` | Single transform only (`SO3` versions are batched). |
| `KVArray.jacobian` | Ends in `.squeeze()`, so a 1-element array yields a `()`-shaped result. Reshape explicitly. |
| `arr.min()` / `np.min(arr)` on symbolic | Raises by design — use `kv.min` / `kv.max`. |
| `test/test_kvarray.py::TestMinMaxMethods` (4 tests) | Fail on the *wording* of that exception only; the behaviour is correct and the tests are stale. Everything else passes (336). |
| `notebooks/layout_test.ipynb` cell 9 | Stale: calls `VectorizedLayout` without `args_shared`. Cells 1–3 are accurate. |
| `scripts/sandbox.py` | Requires ROS; will not run in this venv. `scripts/exp_sandbox.py` does run. |

## README errata

`README.md` is a human tutorial and is wrong in several places; where the two disagree, this file
wins.

- `kv.symbol('a')` — no such function. It is `kv.Symbol` / `kv.Position` / …
- The entire **Models** and **URDF** section uses `str` frame names and cannot run as written
  (see above). `kv.Frame('lol')` is fine; every subsequent `'lol'` is not.
- `kv.Transform.from_euler(0, 0, np.deg2rad(90))` is described as "a 90 degree rotation around the
  Y-axis". Arguments are `(roll, pitch, yaw)`, so that is a rotation around **Z**.
- `kv.Constraint(np.deg2rad(-45), np.deg2rad(-45), b)` is presented as a ±45° limit; both bounds are
  the same number, which pins the joint.
- The QP-controller section describes the maths but shows no API; `Objective`'s `gain` semantics
  (see above) are documented nowhere else.
