from . import spatial    as gm
from . import urdf_utils as urdf
from . import exp_utils  as exp
from . import splines

try:
    from . import ros_utils as ros
except ModuleNotFoundError:
    print('No ROS found. ROS functions not loaded.')

from .spatial import *
from .lie     import SE3, \
                     SO3

from .graph import FKChainException, \
                   Frame,            \
                   FrameView

from .model import Model,           \
                   Constraint,      \
                   ConstrainedEdge, \
                   TransformEdge,   \
                   ConstrainedTransformEdge, \
                   Body, \
                   Geometry, \
                   Inertial

from .layouting import VectorizedLayout, \
                       MacroLayout, \
                       RAI_NLPSolver, \
                       SolverObjectives

from .visualization import generate_expression_graph, \
                           graph_to_html

def __init_ros_serialization():
    try:
        from roebots import ROS_SERIALIZER
        from roebots.ros_serializer import serialize_np_matrix_quat, \
                                           serialize_np_4x4_pose, \
                                           serialize_np_4x1_matrix 

        from geometry_msgs.msg import Quaternion as QuaternionMsg, \
                                      Pose       as PoseMsg,       \
                                      Point      as PointMsg,      \
                                      Vector3    as Vector3Msg

        def floatify(f):
            def g(m):
                return f(m.astype(float))
            return g

        ROS_SERIALIZER.add_serializer(floatify(serialize_np_matrix_quat), {gm.KVArray}, {QuaternionMsg})
        ROS_SERIALIZER.add_serializer(floatify(serialize_np_4x4_pose), {gm.KVArray}, {PoseMsg})
        ROS_SERIALIZER.add_serializer(floatify(serialize_np_4x1_matrix), {gm.KVArray}, {PointMsg, Vector3Msg})
    except (ModuleNotFoundError, ImportError):
        pass

__init_ros_serialization()
