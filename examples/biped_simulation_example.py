import math
from robotics.basic_sim import Sim, BasicRobot, np, pin

inches2m = lambda value: value / 39.3701  # convert to sim units: from inches2m --> meters
grams2kg = lambda value: value / 1000.  # convert to sim units: from grams2kg --> kg


class BasicBiped(BasicRobot):
    leg_length = inches2m(9)
    pelvis_width = leg_length * 5 / 9
    pelvis_depth = pelvis_width / 4
    pelvis_height = pelvis_width / 2
    motor_mass = grams2kg(88.)
    leg_mass = grams2kg(10.)
    foot_mass = grams2kg(50.)
    joints = [{
        'name': "floor",
        'joint_model': 'fixed',
        'joint_placement': {'z': -0.5},
        'dimensions': (1000., 1000., 1.),
        'color': 'grey'
    }, {
        'name': "pelvis",
        'mass': motor_mass,
        'joint_placement': {'z': leg_length * 2.5},
        'body_placement': {'z': pelvis_height * 0.5},
        'dimensions': (pelvis_depth, pelvis_width, pelvis_height),
        'color': 'blue',
        'contacting': "floor"
    }] + BasicRobot.mirror_joints([{
        'name': "hip_side_rotation",
        'parent': "pelvis",
        'joint_model': "RX",
        'joint_placement': {'x': -pelvis_depth * 1.1, 'y': -pelvis_width / 2.},
        'shape': "cylinder",
        'dimensions': (pelvis_depth / 2., pelvis_depth / 2.),
        'mass': motor_mass * 3,
        'color': 'blue',
        'contacting': "floor"
    }, {
        'name': "upper_leg_forward_rotation",
        'parent': "hip_side_rotation",
        'joint_model': "RY",
        'joint_placement': {'x': pelvis_depth * 1.1},
        'body_placement': {'z': -leg_length / 2.},
        'shape': "cylinder",
        'dimensions': (pelvis_depth * 0.4, leg_length * 0.9),
        'mass': leg_mass,
        'color': 'white'
    }, {
        'name': "lower_leg",
        'parent': "upper_leg_forward_rotation",
        'joint_model': "RY",
        'joint_placement': {'z': -leg_length},
        'body_placement': {'z': -leg_length / 2.},
        'shape': "cylinder",
        'dimensions': (pelvis_depth * 0.4, leg_length * 0.9),
        'mass': leg_mass,
        'color': 'white',
        'contacting': "floor"
    }, {
        'name': "foot",
        'parent': "lower_leg",
        'joint_model': "fixed",
        'joint_placement': {},
        'body_placement': {'z': -leg_length},
        'shape': "sphere",
        'dimensions': pelvis_depth / 2.,
        'mass': foot_mass,
        'color': 'black',
        'contacting': "floor"
    }])


def _example_simulation():
    import robotics.meshcat_monitor
    import meshcat.transformations as tf

    # create robot
    robot = BasicBiped()

    # visualize robot
    robot.View(
        robotics.meshcat_monitor.MeshcatVisualizer,
        tf.rotation_matrix(-math.pi / 2, [0.2, 0.2, 1], [1, 0, 0.7]))

    # simulate the robot
    sim = Sim(robot, display=True, display_rate=1.)

    # modify initial pose
    for i in range(1, 6):
        sim.q[-i] -= math.pi / 16. * (1 - 2 * (i % 2 == 0))

    # let the user see the robot in its default pose briefly before starting the simulation
    sim.show(sleep_time=0.3)

    # sim until the user closes the visualizer window
    while robot.viz.is_open():

        # user defined control inputs (tau) and user defined external forces (fs_ext: expressed in the local frame of the joints)
        fs_ext = [pin.Force(np.zeros(6)) for _ in range(len(robot.model.joints))]
        tau = np.zeros((robot.model.nv))

        # apply some random joint torques to make the simulation more interesting
        tau[-6:] = 0.2

        # steps the simulation one time-step, recalculates all terms, and displays at the given FPS
        sim.step(tau, fs_ext)


if __name__ == "__main__":
    _example_simulation()
