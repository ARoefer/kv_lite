"""Validation ladder for Graph.get_mass_matrix (mass_matrix_plan.md, Phase 5).

Self-contained checks (no external dynamics library required):
  1. Point mass, one revolute DOF at distance l   -> M == m * l**2
  2. Two-link planar arm                           -> closed-form textbook M(q)
  3. Symmetry + positive-definiteness              -> random configs of a random tree
  4. Energy consistency (strongest)                -> 0.5 qd^T M qd == sum of per-body KE
                                                       from independently differentiated FK

Runnable directly (`python test/test_mass_matrix.py`) as well as under pytest.
"""
from pathlib import Path

import numpy as np

import kv_lite as gm
from kv_lite import Model,        \
                    TransformEdge, \
                    Frame,        \
                    Body,         \
                    Inertial,     \
                    Symbol,       \
                    KVArray


# ----------------------------------------------------------------------------- helpers
def evalf(arr : KVArray, subs : dict) -> np.ndarray:
    """Evaluate a (possibly symbolic) KVArray to a float ndarray."""
    return arr.eval(subs).astype(float) if arr.is_symbolic else np.asarray(arr).astype(float)


def random_spd(rng, scale=0.3) -> KVArray:
    """A random symmetric positive-definite 3x3 inertia tensor."""
    A = rng.standard_normal((3, 3))
    return KVArray(scale * (A @ A.T) + np.eye(3) * 0.1)


def body_kinetic_energy(graph : Model, name : Path, subs : dict, qdot : dict) -> float:
    """KE of one body from independently differentiated FK (reference for check 4).

    KE = 0.5 m |v_com|^2 + 0.5 omega^T I_world omega, with v_com the CoM linear velocity,
    omega the angular velocity, I_world the inertia about the CoM in the world frame — all
    obtained by differentiating the body's world FK, with no reference to get_mass_matrix.
    """
    body      = graph.get_frame(name)
    inertial  = body.inertial
    world_T_b = graph.get_fk(name).transform
    world_T_c = world_T_b.dot(inertial.origin)

    p = world_T_c[:3, 3]        # CoM position (symbolic)
    R = world_T_c[:3, :3]       # CoM orientation (symbolic)

    syms = list(qdot.keys())
    # v_com = sum_k (d p / d q_k) * qdot_k
    v = np.zeros(3)
    dR = np.zeros((3, 3))
    for s in syms:
        v  = v  + evalf(gm.diff(p, s), subs) * qdot[s]
        dR = dR + evalf(gm.diff(R, s), subs) * qdot[s]

    R_e     = evalf(R, subs)
    omega   = evalf(gm.unskew(KVArray(dR @ R_e.T)), {})   # omega = vee(R_dot @ R^T)
    I_world = R_e @ evalf(inertial.moments, subs) @ R_e.T

    return 0.5 * inertial.mass * (v @ v) + 0.5 * omega @ (I_world @ omega)


# ----------------------------------------------------------------------------- builders
def revolute_z_arm(l, mass, com_origin=None):
    """world --Rz(q)--> a --trans(l,0,0)--> body(mass)."""
    q = Symbol('q')
    md = Model(root_node=Path('world'))
    io = gm.Transform.identity() if com_origin is None else com_origin
    body = Body(Path('body'), Inertial(origin=io, mass=mass, moments=gm.zeros((3, 3))), geom_collision=[])
    md.add_frame(Frame(Path('a')))
    md.add_frame(body)
    md.add_edge(TransformEdge(Path('world'), Path('a'),    gm.Transform.from_euler(0, 0, q)))
    md.add_edge(TransformEdge(Path('a'),     Path('body'), gm.Transform.from_xyz(l, 0, 0)))
    return md, [q]


def two_link_planar(l1, l2, m1, m2):
    """Planar 2R arm with a point mass at the tip of each link.

    Structure:
      world --Rz(q1)-->            L1  (joint 1 at origin)
      L1    --trans(l1,0,0)-->     b1  (point mass m1 at link-1 tip)
      L1    --trans(l1,0,0)@Rz(q2)-> L2 (joint 2 at link-1 tip, rotated by q2)
      L2    --trans(l2,0,0)-->     b2  (point mass m2 at link-2 tip)
    """
    q1, q2 = Symbol('q1'), Symbol('q2')
    md = Model(root_node=Path('world'))
    md.add_frame(Frame(Path('L1')))
    md.add_frame(Body(Path('b1'), Inertial(gm.Transform.identity(), m1, gm.zeros((3, 3))), geom_collision=[]))
    md.add_frame(Frame(Path('L2')))
    md.add_frame(Body(Path('b2'), Inertial(gm.Transform.identity(), m2, gm.zeros((3, 3))), geom_collision=[]))

    md.add_edge(TransformEdge(Path('world'), Path('L1'), gm.Transform.from_euler(0, 0, q1)))
    md.add_edge(TransformEdge(Path('L1'),    Path('b1'), gm.Transform.from_xyz(l1, 0, 0)))
    md.add_edge(TransformEdge(Path('L1'),    Path('L2'), gm.Transform.from_xyz(l1, 0, 0).dot(gm.Transform.from_euler(0, 0, q2))))
    md.add_edge(TransformEdge(Path('L2'),    Path('b2'), gm.Transform.from_xyz(l2, 0, 0)))
    return md, [q1, q2]


def random_tree(rng, n_links=4):
    """Random serial chain of `n_links` revolute joints about random axes, each carrying a
    body with random mass, random SPD inertia and a random CoM offset via inertial.origin."""
    md   = Model(root_node=Path('world'))
    syms = []
    parent = Path('world')
    for i in range(n_links):
        q      = Symbol(f'q{i}')
        syms.append(q)
        axis   = rng.standard_normal(3)
        axis   = KVArray(axis / np.linalg.norm(axis))
        offset = rng.standard_normal(3) * 0.4
        com    = gm.Transform.from_xyz(*(rng.standard_normal(3) * 0.2))
        name   = Path(f'link{i}')
        joint_tf = gm.Transform.from_xyz(*offset).dot(gm.Transform.from_axis_angle(axis, q))
        md.add_frame(Body(name, Inertial(com, 0.5 + rng.random(), random_spd(rng)), geom_collision=[]))
        md.add_edge(TransformEdge(parent, name, joint_tf))
        parent = name
    return md, syms


# ----------------------------------------------------------------------------- tests
def test_point_mass():
    l, mass = 0.7, 2.5
    md, syms = revolute_z_arm(l, mass)
    M = md.get_mass_matrix(syms)
    for qi in [0.0, 0.5, 1.3, -2.0]:
        assert np.allclose(evalf(M, {syms[0]: qi}), mass * l**2)


def test_point_mass_com_via_inertial_origin():
    # CoM offset supplied through inertial.origin instead of the joint transform.
    l, mass = 0.9, 1.3
    q = Symbol('q')
    md = Model(root_node=Path('world'))
    md.add_frame(Body(Path('body'), Inertial(gm.Transform.from_xyz(l, 0, 0), mass, gm.zeros((3, 3))), geom_collision=[]))
    md.add_edge(TransformEdge(Path('world'), Path('body'), gm.Transform.from_euler(0, 0, q)))
    M = md.get_mass_matrix([q])
    assert np.allclose(evalf(M, {q: 0.6}), mass * l**2)


def test_two_link_closed_form():
    l1, l2, m1, m2 = 0.5, 0.4, 1.2, 0.8
    md, syms = two_link_planar(l1, l2, m1, m2)
    M = md.get_mass_matrix(syms)
    q1, q2 = syms
    rng = np.random.default_rng(0)
    off_diag_values = []
    for _ in range(6):
        a, b = rng.uniform(-np.pi, np.pi, 2)
        Mv = evalf(M, {q1: a, q2: b})
        c2 = np.cos(b)
        M11 = m1 * l1**2 + m2 * (l1**2 + l2**2 + 2 * l1 * l2 * c2)
        M12 = m2 * (l2**2 + l1 * l2 * c2)
        M22 = m2 * l2**2
        expected = np.array([[M11, M12], [M12, M22]])
        assert np.allclose(Mv, expected), f"\n{Mv}\n!=\n{expected}"
        off_diag_values.append(Mv[0, 1])
    # off-diagonal coupling must actually depend on q2
    assert np.ptp(off_diag_values) > 1e-6


def test_symmetry_and_pd():
    rng = np.random.default_rng(42)
    md, syms = random_tree(rng, n_links=4)
    M = md.get_mass_matrix(syms)
    for _ in range(20):
        subs = {s: rng.uniform(-np.pi, np.pi) for s in syms}
        Mv = evalf(M, subs)
        assert np.allclose(Mv, Mv.T, atol=1e-9), "not symmetric"
        eigs = np.linalg.eigvalsh(0.5 * (Mv + Mv.T))
        assert eigs.min() > 1e-9, f"not positive definite: min eig {eigs.min()}"


def test_energy_consistency():
    """0.5 qd^T M qd  ==  sum over bodies of KE from independently differentiated FK."""
    rng = np.random.default_rng(7)
    md, syms = random_tree(rng, n_links=4)
    M = md.get_mass_matrix(syms)
    body_names = [Path(f'link{i}') for i in range(4)]
    for _ in range(15):
        subs = {s: rng.uniform(-np.pi, np.pi) for s in syms}
        qdot = {s: rng.standard_normal()      for s in syms}
        qd   = np.array([qdot[s] for s in syms])

        Mv        = evalf(M, subs)
        ke_matrix = 0.5 * qd @ Mv @ qd
        ke_bodies = sum(body_kinetic_energy(md, n, subs, qdot) for n in body_names)
        assert np.isclose(ke_matrix, ke_bodies, rtol=1e-7, atol=1e-9), \
            f"energy mismatch: matrix={ke_matrix}  bodies={ke_bodies}"


def test_gravity_finite_difference():
    """Self-contained: G(q) must equal the numerical gradient of the potential energy
    U(q) = -sum_i m_i * g . p_i(q), computed independently of get_gravity_vector."""
    rng = np.random.default_rng(21)
    md, syms = random_tree(rng, n_links=4)
    g   = np.array([0.0, 0.0, -9.81])
    G   = md.get_gravity_vector(syms, gravity=tuple(g))
    names = [Path(f'link{i}') for i in range(4)]

    def potential(subs):
        U = 0.0
        for n in names:
            body = md.get_frame(n)
            p    = evalf(md.get_fk(n).transform.dot(body.inertial.origin)[:3, 3], subs)
            U   -= body.inertial.mass * g.dot(p)
        return U

    for _ in range(5):
        base = {s: rng.uniform(-np.pi, np.pi) for s in syms}
        G_kv = evalf(G, base).reshape(-1)
        eps  = 1e-6
        G_fd = np.zeros(len(syms))
        for i, s in enumerate(syms):
            up = dict(base); up[s] += eps
            dn = dict(base); dn[s] -= eps
            G_fd[i] = (potential(up) - potential(dn)) / (2 * eps)
        assert np.allclose(G_kv, G_fd, atol=1e-5), f"{G_kv} != {G_fd}"


def test_operational_space_inertia():
    """Lambda = (J M^-1 J^T)^-1 must be symmetric PD and satisfy its defining identity."""
    # Need >= 6 DOF so the 6xn tip Jacobian can span the full task space; otherwise
    # J M^-1 J^T is rank-deficient and Lambda does not exist.
    rng = np.random.default_rng(31)
    md, syms = random_tree(rng, n_links=7)
    M_sym = md.get_mass_matrix(syms)
    J_sym = md.body_jacobian(Path('link6'), syms)   # 6 x n Jacobian of the tip frame
    subs  = {s: rng.uniform(-np.pi, np.pi) for s in syms}
    M = evalf(M_sym, subs)
    J = evalf(J_sym, subs)
    Lam = gm.operational_space_inertia(M, J)
    assert np.allclose(Lam, Lam.T, atol=1e-9), "Lambda not symmetric"
    assert np.linalg.eigvalsh(0.5 * (Lam + Lam.T)).min() > 1e-9, "Lambda not PD"
    assert np.allclose(np.linalg.inv(Lam), J @ np.linalg.inv(M) @ J.T, atol=1e-9), "identity violated"


def test_cross_check_pybullet_dynamics():
    """External reference: gravity G(q) and Coriolis C(q,q') against PyBullet inverse
    dynamics on the KUKA iiwa. With q''=0, tau = C q' + G, so it validates both together
    (and, jointly with the mass-matrix check, the full manipulator equation)."""
    import pytest
    p             = pytest.importorskip('pybullet')
    pybullet_data = pytest.importorskip('pybullet_data')
    from kv_lite import Symbol
    from kv_lite.urdf_utils import load_urdf

    urdf_rel = 'kuka_iiwa/model.urdf'
    urdf_str = (Path(pybullet_data.getDataPath()) / urdf_rel).read_text()
    md   = Model(root_node=Path('world'))
    uobj = load_urdf(md, urdf_str)
    dj   = uobj.dynamic_joints

    cid  = p.connect(p.DIRECT)
    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        body    = p.loadURDF(urdf_rel, useFixedBase=True, flags=p.URDF_USE_INERTIA_FROM_FILE)
        movable = [p.getJointInfo(body, i)[1].decode() for i in range(p.getNumJoints(body))
                   if p.getJointInfo(body, i)[2] in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC)]
        n       = len(movable)
        q_syms  = [dj[jn].position for jn in movable]
        qd_syms = [Symbol(f'qd{i}') for i in range(n)]

        G = md.get_gravity_vector(q_syms, source_frame=uobj.root)
        C = md.get_coriolis_matrix(q_syms, qd_syms, source_frame=uobj.root)

        rng = np.random.default_rng(11)
        for _ in range(6):
            q  = rng.uniform(-2.5, 2.5, n)
            qd = rng.uniform(-1.5, 1.5, n)
            sq = {s: v for s, v in zip(q_syms, q)}
            sa = {**sq, **{s: v for s, v in zip(qd_syms, qd)}}
            G_kv = evalf(G, sq).reshape(-1)
            C_kv = evalf(C, sa)
            id_g  = np.array(p.calculateInverseDynamics(body, list(q), [0.] * n, [0.] * n))
            id_cg = np.array(p.calculateInverseDynamics(body, list(q), list(qd), [0.] * n))
            assert np.allclose(G_kv,             id_g,  atol=1e-6), "gravity mismatch"
            assert np.allclose(C_kv @ qd + G_kv, id_cg, atol=1e-6), "coriolis mismatch"
    finally:
        p.disconnect(cid)


def test_cross_check_pybullet_kuka():
    """External reference: match PyBullet's mass matrix on the KUKA iiwa (clean 7R).

    Two PyBullet gotchas, both handled here:
    - `URDF_USE_INERTIA_FROM_FILE`: without it PyBullet ignores the URDF `<inertia>` and
      recomputes it from collision geometry, so the matrices would differ by a few percent.
    - The loaded URDF's root link is detached from `world`; get_mass_matrix must be called
      with `source_frame` set to that root, not the default `world`.
    """
    import pytest
    p            = pytest.importorskip('pybullet')
    pybullet_data = pytest.importorskip('pybullet_data')

    from kv_lite.urdf_utils import load_urdf

    urdf_rel = 'kuka_iiwa/model.urdf'
    urdf_str = (Path(pybullet_data.getDataPath()) / urdf_rel).read_text()
    md   = Model(root_node=Path('world'))
    uobj = load_urdf(md, urdf_str)
    dj   = uobj.dynamic_joints

    cid  = p.connect(p.DIRECT)
    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        body    = p.loadURDF(urdf_rel, useFixedBase=True, flags=p.URDF_USE_INERTIA_FROM_FILE)
        movable = [p.getJointInfo(body, i)[1].decode() for i in range(p.getNumJoints(body))
                   if p.getJointInfo(body, i)[2] in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC)]
        symbols = [dj[jn].position for jn in movable]
        M       = md.get_mass_matrix(symbols, source_frame=uobj.root)

        rng = np.random.default_rng(3)
        for _ in range(8):
            q   = rng.uniform(-2.5, 2.5, len(movable))
            Mkv = M.eval({s: qi for s, qi in zip(symbols, q)}).astype(float)
            Mpb = np.array(p.calculateMassMatrix(body, list(q)))
            assert np.allclose(Mkv, Mpb, atol=1e-9), f"max diff {np.abs(Mkv-Mpb).max():.3e}"
    finally:
        p.disconnect(cid)


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f"PASS  {name}")
    print("\nALL PHASE 5 SELF-CONTAINED CHECKS PASSED")
