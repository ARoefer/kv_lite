import numpy as np
import rospy
import tf2_ros

from jinja2  import Environment, FileSystemLoader, select_autoescape, exceptions
from pathlib import Path

from geometry_msgs.msg  import TransformStamped as TransformStampedMsg

from .      import spatial as gm
from .model import Model

env = Environment(
    loader=FileSystemLoader(f'{Path(__file__).parent}/../../data'),
    autoescape=select_autoescape(['html', 'xml'])
)
urdf_template = env.get_template('urdf_template.jinja')


def gen_urdf(model : Model):
    frames = set(model.get_frames())
    frames.remove('world')

    joints = {f'{j.child}_joint': j for j in model.get_edges()}

    sorted_tfs = list(joints.items())
    tf_stack   = gm.KVArray(np.vstack([model.get_fk(j.child, j.parent).transform[:3] for _, j in sorted_tfs]))

    print(type(tf_stack))

    return urdf_template.render(frames=frames, joints=joints), \
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
    def __init__(self, model : Model):
        self.static_broadcaster  = tf2_ros.StaticTransformBroadcaster()
        self.dynamic_broadcaster = tf2_ros.TransformBroadcaster()
        self.refresh_model(model)

    def refresh_model(self, model : Model):
        urdf, self.transforms, self.tf_stack = gen_urdf(model)

        rospy.set_param('robot_description', urdf)

    def update(self, q):
        poses = self.tf_stack.eval(q)
        
        now = rospy.Time.now()

        transforms = []
        for i, (parent, child) in enumerate(self.transforms):
            tf   = poses[i*3:i*3+3]
            pos  = tf[:, 3]
            quat = real_quat_from_matrix(tf)

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
        
        self.dynamic_broadcaster.sendTransform(transforms)

    @property
    def symbols(self):
        return self.tf_stack.symbols

