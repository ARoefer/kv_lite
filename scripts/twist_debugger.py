import numpy as np
import rospy

import kv_lite as kv
import kv_control as kvc
import kv_lite.exp_utils as kve

from dataclasses import dataclass
from kv_lite     import gm
from roebots     import ROSVisualizer

from sensor_msgs.msg import JointState as JointStateMsg


if __name__ == '__main__':
    print('Run\n\nrosrun joint_state_publisher_gui joint_state_publisher_gui '
          'robot_description:=twist_description joint_states:=twist_params\n\n'
          'in other terminal. Set reference frame to "world" in RVIZ and '
          'visualize the marker array set on /vis_twist')

    rospy.init_node('twist_debug')

    vis = ROSVisualizer('vis_twist')

    # Build Twist
    twist_params = [gm.KVSymbol(f'twist_{x}') for x in 'vx vy vz wx wy wz'.split(' ')]
    theta = gm.Position('theta')

    v = gm.KVArray(twist_params[:3])
    w = gm.KVArray(twist_params[3:])

    twist = kve.twist_to_se3(theta, v, w)

    # Build ROS components
    parameter_description = '<robot name="gmm">\n{}\n{}\n</robot>'.format('\n'.join([
        f'<joint name="{str(n)}" type="prismatic">\n   <limit lower="0" upper="8"/>\n</joint>' for n in v
    ]),
    '\n'.join([
        f'<joint name="{str(n)}" type="prismatic">\n   <limit lower="-1" upper="1"/>\n</joint>' for n in w
    ]))

    rospy.set_param('/twist_description', parameter_description)

    def cb_update(msg : JointStateMsg):
        # Giving exponential influence to v
        state = {gm.KVSymbol(s): (np.exp(v) - 1 if 'v' in s else v) for s, v in zip(msg.name, msg.position)}

        poses = []
        for v in np.linspace(0, 1, 10):
            state[theta] = v

            poses.append(twist.eval(state))

        vis.begin_draw_cycle('poses', 'path')
        vis.draw_poses('poses', np.eye(4), 0.05, 0.002, poses)
        vis.draw_strip('path', np.eye(4), 0.001, [p[:, 3] for p in poses])
        vis.render()
    
    # Initialize this somehow
    cb_update(JointStateMsg(name=[str(s) for s in twist_params], position=[1, 0, 0, 1, 0, 0]))

    sub_params = rospy.Subscriber('twist_params', JointStateMsg, callback=cb_update, queue_size=1)

    while not rospy.is_shutdown():
        rospy.sleep(0.1)
