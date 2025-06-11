import xml.etree.ElementTree as ET

from functools import cached_property
from queue  import Queue

from .      import spatial as gm
from .model import Model,        \
                   ConstrainedEdge, \
                   Body,         \
                   Frame,        \
                   Constraint,   \
                   Inertial,     \
                   Geometry

from pathlib import Path


class URDFJoint(ConstrainedEdge):
    def __init__(self, name : Path, parent, child, origin, type, position, 
                       axis=gm.unitX, limit_pos=None, limit_vel=None, limit_effort=None) -> None:

        if type not in {'revolute', 'continuous', 'prismatic', 'fixed'}:
            raise RuntimeError(f'URDF joint type "{type}" not implemented.')

        self.name      = name
        self.origin    = origin
        self.type      = type
        self.position  = position
        self.axis      = axis
        self.limit_pos = gm.KVArray(limit_pos) if limit_pos is not None else None
        self.limit_vel = limit_vel
        self.limit_effort = limit_effort

        constraints = {}
        if self.limit_pos is not None:
            constraints[self.name / 'position'] = Constraint(self.limit_pos[0], self.limit_pos[1], self.position)
        if self.limit_vel is not None:
            constraints[self.name / 'velocity'] = Constraint(-self.limit_vel, self.limit_vel, self.position.tangent())

        super().__init__(parent, child, constraints)

    def eval(self, graph, current_tf : gm.KVArray) -> gm.KVArray:
        if self.type == 'fixed':
            return self.origin.dot(current_tf)
        elif self.type == 'revolute' or self.type == 'continuous':
            return self.origin.dot(gm.Transform.from_axis_angle(self.axis, self.position)).dot(current_tf)
        elif self.type == 'prismatic':
            return self.origin.dot(gm.Transform.from_xyz(*(self.axis[:3].T[0] * self.position))).dot(current_tf)

        raise RuntimeError(f'Unknown joint type {self.type}')


class URDFObject():
    """A lightweight interface block to interact the model class using the
    familiar URDF concepts (i.e. joints, links)
    """
    def __init__(self, model : Model, name, links, joints, root) -> None:
        self._model  = model
        self._name   = name
        self._links  = links
        self._joints = joints
        self._root   = root

    @property
    def model(self) -> Model:
        return self._model

    @property
    def root(self):
        return self._links[self._root]
    
    @property
    def root_link(self):
        return self._root

    @property
    def name(self):
        return self._name
    
    @property
    def links(self):
        return {l: self._model.get_frame(fn) for l, fn in self._links.items()}
    
    @property
    def joints(self):
        return {j: self._model.get_edge(en) for j, en in self._joints.items()}

    @property
    def dynamic_joints(self):
        return {j: edge for j, en in self._joints.items() if (edge:=self._model.get_edge(en)).type != 'fixed'}

    @cached_property
    def joints_by_symbols(self):
        return {j.position: jn for jn, j in self.dynamic_joints.items()}

    def get_link(self, name):
        return self._model.get_frame(self._links[name])
    
    def get_joint(self, name):
        return self._model.get_frame(self._joints[name])

    def get_fk(self, link, target='world'):
        l_fullname = self._links[link] if link in self._links else link
        target     = self._links[target] if target in self._links else target
        return self._model.get_fk(l_fullname, target)

    @cached_property
    def q(self) -> list[gm.KVSymbol]:
        acc = set()
        for j in self.dynamic_joints.values():
            if j.position is not None:
                acc.update(j.position.symbols)
        return list(acc)
    
    @cached_property
    def q_limit(self):
        q_lim = gm.ones((len(self.q), 2))
        q_lim[:, 0] *= -1
        q_lim *= gm.np.inf

        for x, j in enumerate(self.q):
            constraints = self._model.get_constraints([j])
            if len(constraints) > 0:
                most_constant_constraint = None
                score = 3
                for c in constraints.values():
                    s = int(gm.is_symbolic(c.lb)) + int(gm.is_symbolic(c.ub))
                    if s < score:
                        most_constant_constraint = c
                        score = s

                low, high, _ = most_constant_constraint
                q_lim[x] = low, high
        return q_lim

    @cached_property
    def q_dot(self) -> list[gm.KVSymbol]:
        return [p.derivative() for p in self.q]

    @cached_property
    def q_dot_limit(self):
        q_lim = gm.ones((len(self.q_dot), 2))
        q_lim[:, 0] *= -1
        q_lim *= gm.np.inf

        for x, j in enumerate(self.q_dot):
            constraints = self._model.get_constraints([j])
            if len(constraints) > 0:
                most_constant_constraint = None
                score = 3
                for c in constraints.values():
                    s = int(gm.is_symbolic(c.lb)) + int(gm.is_symbolic(c.ub))
                    if s < score:
                        most_constant_constraint = c
                        score = s

                low, high, _ = most_constant_constraint
                q_lim[x] = low, high
        return q_lim


def _parse_origin_node(on : ET.Element):
    if on is None:
        return gm.Transform.identity()

    translation = gm.Transform.from_xyz(*[float(v) for v in on.attrib['xyz'].split(' ') if v != '']) if 'xyz' in on.attrib else gm.Transform.identity()
    rotation    = gm.Transform.from_euler(*[float(v) for v in on.attrib['rpy'].split(' ') if v != '']) if 'rpy' in on.attrib else gm.Transform.identity()
    return translation.dot(rotation)


def _parse_inertial(node : ET.Element) -> Inertial:
    if node is None:
        return Inertial(gm.Transform.identity(), 1, gm.eye(3))
    mnode   = node.find('inertia')
    imatrix = gm.eye(3)
    if mnode is not None:
        imatrix = gm.KVArray([[float(mnode.attrib['ixx']), float(mnode.attrib['ixy']), float(mnode.attrib['ixz'])],
                              [float(mnode.attrib['ixy']), float(mnode.attrib['iyy']), float(mnode.attrib['iyz'])],
                              [float(mnode.attrib['ixz']), float(mnode.attrib['iyz']), float(mnode.attrib['izz'])]])

    return Inertial(_parse_origin_node(node.find('origin')),
                    float(node.find('mass').attrib['value']) if node.find('mass') is not None else 1.0,
                    imatrix)


def _parse_geom_node(node : ET.Element):
    origin   = _parse_origin_node(node.find('origin'))
    geometry = node.find('geometry') # type: ET.Element
    geom     = geometry[0]
    if geom.tag == 'mesh':
        scale = gm.vector3(1, 1, 1)
        if 'scale' in geom.attrib:
            scale = gm.vector3(*[float(c) for c in geom.attrib['scale'].split(' ') if c != ''])
        return Geometry('mesh', geom.attrib['filename'], scale, origin)
    elif geom.tag == 'box':
        return Geometry('box', None, gm.vector3(*[float(c) for c in geom.attrib['size'].split(' ') if c != '']), origin)
    elif geom.tag == 'cylinder':
        return Geometry('cylinder',
                        None,
                        gm.vector3(float(geom.attrib['radius']), float(geom.attrib['radius']), float(geom.attrib['length'])) * 2,
                        origin)
    elif geom.tag == 'sphere':
        return Geometry('sphere',
                        None,
                        gm.vector3(2, 2, 2) * float(geom.attrib['radius']),
                        origin)
    else:
        raise KeyError(f'Unknown URDF geometry "{geom.tag}"')


def _parse_link_node(model : Model, link_node : ET.Element,
                     name_prefix : Path, use_visual_as_collision=True):
    inertial_node = link_node.find('inertial')
    inertial = _parse_inertial(inertial_node)

    collision_nodes = link_node.findall('collision')
    if len(collision_nodes) == 0 and use_visual_as_collision:
        collision_nodes = link_node.findall('visual')

    coll_geometries = [_parse_geom_node(cn) for cn in collision_nodes]
    vis_geometries  = [_parse_geom_node(cn) for cn in link_node.findall('visual')]

    model.add_frame(Body(name_prefix / link_node.attrib['name'],
                         inertial,
                         coll_geometries,
                         vis_geometries if len(vis_geometries) > 0 else None))
    return name_prefix / link_node.attrib['name']


def _parse_joint_node(model : Model, joint_node : ET.Element, name_prefix : Path):
    
    if joint_node.find('parent') is None:
        raise RuntimeError(f'Joint "{joint_node.attrib["name"]}" has no "parent" element')
    
    if joint_node.find('child') is None:
        raise RuntimeError(f'Joint "{joint_node.attrib["name"]}" has no "child" element')

    type    = joint_node.attrib['type']
    parent  = joint_node.find('parent').attrib['link']
    child   = joint_node.find('child').attrib['link']
    origin  = _parse_origin_node(joint_node.find('origin'))
    ax_node = joint_node.find('axis')
    axis    = gm.vector3(1, 0, 0) if ax_node is None else gm.vector3(*[float(v) for v in ax_node.attrib['xyz'].split(' ') if v != ''])

    limit_node = joint_node.find('limit')
    if type in {'revolute', 'prismatic'} and limit_node is None:
        raise RuntimeError(f'Joint type "{type}" requires limit node.')
    
    limit_lb = None if limit_node is None or type == 'fixed' else 0
    limit_ub = None if limit_node is None or type == 'fixed' else 0
    limit_vel    = None if limit_node is None or type == 'fixed' else float(limit_node.attrib['velocity'])
    limit_effort = None if limit_node is None or type == 'fixed' else float(limit_node.attrib['effort'])

    if limit_node is not None and type != 'fixed' and 'lower' in limit_node.attrib:
        limit_lb = float(limit_node.attrib['lower'])
        limit_ub = float(limit_node.attrib['upper'])
    
    mimic_node = joint_node.find('mimic')
    if type != 'fixed':
        if mimic_node is not None:
            m = 1.0 if 'multiplier' not in mimic_node.attrib else float(mimic_node.attrib['multiplier'])
            b = 0.0 if 'offset' not in mimic_node.attrib else float(mimic_node.attrib['offset'])
            position = model.get_edge(name_prefix / mimic_node.attrib['joint']).position * m + b
        else:
            position = gm.Position(name_prefix / joint_node.attrib['name'])
    else:
        position = None
    
    joint = URDFJoint(name_prefix / joint_node.attrib['name'],
                      name_prefix / parent,
                      name_prefix / child,
                      origin=origin,
                      type=type,
                      position=position,
                      axis=axis,
                      limit_pos=(limit_lb, limit_ub) if limit_lb is not None else None,
                      limit_vel=limit_vel,
                      limit_effort=limit_effort)

    model.add_edge(joint, name=joint.name)
    return joint.name



def load_urdf(model : Model, urdf : str, name=None, use_visual_as_collision=True) -> URDFObject:

    # Using ElementTree, because the Python implementation of URDF is fickle
    root = ET.fromstring(urdf)

    if root.tag != 'robot':
        raise RuntimeError(f'Expected root of urdf to be tagged "robot" but got "{root.tag}"')
    
    if 'name' not in root.attrib and name is None:
        raise RuntimeError('URDFs without names can only parsed, if a name is given explicitly.')
    
    name = Path(name) if name is not None else root.attrib['name']

    links = {}
    for link in root.findall('link'):
        links[link.attrib['name']] = _parse_link_node(model,
                                                      link,
                                                      Path(name),
                                                      use_visual_as_collision=use_visual_as_collision)


    joint_queue = Queue()
    for j in root.findall('joint'):
        joint_queue.put(j)

    # Primitive way of handling joints' mimic dependencies
    joints = {}

    while not joint_queue.empty():
        j = joint_queue.get() # type: ET.Element
        
        mimics = j.findall('mimic')

        if len(mimics) > 1:
            raise RuntimeError(f'Multiple mimics in joint node {j["name"]}')
        elif len(mimics) == 1:
            # If joint has not been instantiated, re-queue this one
            if mimics[0].attrib['joint'] not in joints:
                joint_queue.put(j)
                continue

        joints[j.attrib['name']] = _parse_joint_node(model, j, Path(name))

    roots = [l for l, ln in links.items() if model.get_incoming_edge(ln) is None]

    if len(roots) > 1:
        raise RuntimeError(f'URDF has more than root. Candidates:\n  {roots}')

    return URDFObject(model, name, links, joints, roots[0])

