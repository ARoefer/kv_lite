import numpy as np
import rospy
import tf2_ros

from jinja2  import Environment, FileSystemLoader, select_autoescape, exceptions
from pathlib import Path
from typing  import Set

from geometry_msgs.msg  import TransformStamped as TransformStampedMsg
from scipy.spatial.transform import Rotation

from .      import spatial as gm
from .model import Model, \
                   Body


def rot3_to_rpy(rot3):
    sy = gm.sqrt(rot3[0,0] * rot3[0,0] + rot3[2,2] * rot3[2,2])

    if sy >= 1e-6:
        return (gm.atan2(rot3[2,1], rot3[2,2]), 
                gm.atan2(-rot3[2,0], sy), 
                gm.atan2(rot3[1,0], rot3[0,0]))
    else:
        return (gm.atan2(-rot3[1,2], rot3[1,1]), 
                gm.atan2(-rot3[2,0], sy), 0)


def tf_to_rpy_str(tf : gm.KVArray):
    rpy = rot3_to_rpy(tf)
    return ' '.join([str(float(x)) for x in rpy])


def tf_to_xyz_str(tf : gm.KVArray):
    xyz = gm.Transform.pos(tf).flatten()[:3]
    return ' '.join([str(float(x)) for x in xyz])


env = Environment(
    loader=FileSystemLoader(f'{Path(__file__).parent}/../../data'),
    autoescape=select_autoescape(['html', 'xml'])
)
env.globals.update(tf_to_xyz_str=tf_to_xyz_str)
env.globals.update(tf_to_rpy_str=tf_to_rpy_str)
env.globals.update(Body=Body)
env.globals.update(isinstance=isinstance)
env.globals.update(float=float)

urdf_template = env.get_template('urdf_template.jinja')


def gen_urdf(model : Model, collision_alpha=0.0):
    frames = [model.get_frame(f) for f in model.get_frames()]
    # frames.remove('world')

    joints = {f'{j.child}_joint': j for j in model.get_edges()}

    sorted_tfs = list(joints.items())
    tf_stack   = gm.KVArray([model.get_fk(j.child, j.parent).transform for _, j in sorted_tfs])

    return urdf_template.render(frames=frames, joints=joints, collision_alpha=collision_alpha), \
           [(str(j.parent), str(j.child)) for _, j in sorted_tfs], tf_stack


def real_quat_from_matrix(frame):
    tr = frame[0,0] + frame[1,1] + frame[2,2]

    if tr > 0:
        S = np.sqrt(tr+1.0) * 2 # S=4*qw
        qw = 0.25 * S
        qx = (frame[2,1] - frame[1,2]) / S
        qy = (frame[0,2] - frame[2,0]) / S
        qz = (frame[1,0] - frame[0,1]) / S
    elif frame[0,0] > frame[1,1] and frame[0,0] > frame[2,2]:
        S  = np.sqrt(1.0 + frame[0,0] - frame[1,1] - frame[2,2]) * 2 # S=4*qx
        qw = (frame[2,1] - frame[1,2]) / S
        qx = 0.25 * S
        qy = (frame[0,1] + frame[1,0]) / S
        qz = (frame[0,2] + frame[2,0]) / S
    elif frame[1,1] > frame[2,2]:
        S  = np.sqrt(1.0 + frame[1,1] - frame[0,0] - frame[2,2]) * 2 # S=4*qy
        qw = (frame[0,2] - frame[2,0]) / S
        qx = (frame[0,1] + frame[1,0]) / S
        qy = 0.25 * S
        qz = (frame[1,2] + frame[2,1]) / S
    else:
        S  = np.sqrt(1.0 + frame[2,2] - frame[0,0] - frame[1,1]) * 2 # S=4*qz
        qw = (frame[1,0] - frame[0,1]) / S
        qx = (frame[0,2] + frame[2,0]) / S
        qy = (frame[1,2] + frame[2,1]) / S
        qz = 0.25 * S
    return (float(qx), float(qy), float(qz), float(qw))




class ModelTFBroadcaster(object):
    def __init__(self, model : Model, collision_alpha=0.0, param='robot_description'):
        self._static_broadcaster  = tf2_ros.StaticTransformBroadcaster()
        self._dynamic_broadcaster = tf2_ros.TransformBroadcaster()
        self._param_name         = param
        self.refresh_model(model, collision_alpha=collision_alpha)

    def refresh_model(self, model : Model, collision_alpha=0.0):
        urdf, self._transforms, self._tf_stack = gen_urdf(model, collision_alpha=collision_alpha)

        rospy.set_param(self._param_name, urdf)

    def transforms(self, q) -> np.ndarray:
        return self._tf_stack(q)

    def update(self, q):
        poses = self._tf_stack(q)
        
        now = rospy.Time.now()

        positions = poses[:, :3, 3]
        rotations = Rotation.from_matrix(poses[:, :3, :3]).as_quat()

        transforms = []
        for (parent, child), pos, quat in zip(self._transforms, positions, rotations):

            msg  = TransformStampedMsg()
            msg.header.stamp    = now
            msg.header.frame_id = parent
            msg.child_frame_id  = child
            msg.transform.translation.x, \
            msg.transform.translation.y, \
            msg.transform.translation.z = pos
            msg.transform.rotation.x, \
            msg.transform.rotation.y, \
            msg.transform.rotation.z, \
            msg.transform.rotation.w, = quat
            transforms.append(msg)
        
        self._dynamic_broadcaster.sendTransform(transforms)

    @property
    def symbols(self):
        return self._tf_stack.symbols

    @property
    def ordered_symbols(self):
        return self._tf_stack.ordered_symbols
    
    def set_symbol_order(self, symbols):
        self._tf_stack.set_symbol_order(symbols)
