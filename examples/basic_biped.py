
import basic_robot as br

inches = lambda value: value / 39.3701  # convert to sim units: from inches --> meters
grams = lambda value: value / 1000.  # convert to sim units: from grams --> kg


class BasicBiped(br.BasicRobot):
    pelvis_width = inches(5)
    pelvis_depth = pelvis_width / 4
    pelvis_height = pelvis_width / 2
    leg_length = inches(9)
    motor_mass = grams(88.)
    leg_mass = grams(10.)
    foot_mass = grams(50.)
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
    }] + br.mirror_joints([{
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


if __name__ == "__main__":
    import math
    import time
    import quadprog
    import numpy as np
    import pinocchio as pin
    import meshcat_monitor
    import meshcat.transformations as tf

    # create robot
    robot = BasicBiped()
    q = robot.q0  # positions in joint-space
    dq = robot.v0  # velocities in joint-space (derivative of q)
    ddq = robot.a0  # accelerations in joint-space (derivative of dq)

    # modify initial pose
    q[-2] -= math.pi / 8.

    # visualize robot
    robot.viz = robot.initViz(meshcat_monitor.MeshcatVisualizer)
    robot.initViewer(open=True)
    robot.loadViewerModel()
    robot.displayVisuals(True)
    robot.viz.viewer['/Cameras/default/rotated/<object>'].set_transform(
        tf.rotation_matrix(-math.pi / 2, [0.2, 0.2, 1], [1, 0, 0.7]))
    robot.display(q)
    time.sleep(0.3)  # let the user see the robot in its default pose briefly before simulating

    # simulation settings
    dt = 0.001                           # time step size to integrate with (can be different than display fps)
    fps = 30                             # frames per second to display
    Kp_contact = 1000.                   # stiffness constant for contact distance
    Kv_contact = 1000.                   # stiffness constant for contact velocity
    Kv_impact = 0.1                      # stiffness constant for impact velocity
    Kf = 0.01                            # joint-space friction constant
    tau = np.zeros((robot.model.nv))     # control torques/forces in joint-space
    loops_to_disp = int((1 / fps) / dt)  # number of time steps to simulate before re-displaying
    warn_if_not_real_time = True         # produce a warning if
    loop = 0
    target_time = time.time()

    # sim until the user closes the visualizer window
    while robot.viz.is_open():

        # Compute the model's "M" = Mass matrix and "b" = Nonlinear force components
        # in joint-space (Fq_ext w/gravity + coriolis forces)
        M = pin.crba(robot.model, robot.data, q)
        b = pin.nle(robot.model, robot.data, q, dq)

        # add some joint friction (but don't apply any joint friction forces to the
        # floating 6-dof root joint)
        tau[-6:] = -Kf * dq[-6:]

        # simulate the resulting acceleration (forward dynamics)
        Fq = tau - b
        ddq = np.linalg.inv(M) @ Fq  # solves F = M*a for acceleration in joint-space

        # check and account for collisions
        robot.computeCollisions(q, dq)
        collisions = robot.getCollisionList()
        if not collisions:
            already_contact = set()
        else:
            dist = robot.getCollisionDistances(collisions)
            J = robot.getCollisionJacobian(collisions)
            JdotQdot = robot.getCollisionJdotQdot(collisions)

            # for a new contact, add in an elastic impact velocity
            if Kv_impact > 0:
                col_id = [e[0] for e in collisions]
                new_col_idx = [i for i, e in enumerate(col_id) if e not in already_contact]
                already_contact = set(col_id)
                if new_col_idx:
                    J_proj = np.stack([J[i] for i in new_col_idx], axis=0)
                    impact_vel = (np.linalg.pinv(J_proj) @ J_proj) @ dq
                    dq -= impact_vel * Kv_impact

            # The adjusted joint-space acceleration with the contact(s) needs to
            # Use Gauss principle to find joint-space accelerations (ddq) that satisfies:
            #       Minimize:                1/2 ddq^T M ddq - Fq^T ddq
            #       Subject to constraint:   J.T ddq >= d
            d = - JdotQdot - Kp_contact * dist - Kv_contact * J @ dq
            ddq, _, _, _, _, _ = quadprog.solve_qp(M, Fq, J.T, d)

        # integrate the acceleration (ddq) to update the model's velocities (dq), and positions (q)
        dq += ddq * dt
        q = pin.integrate(robot.model, q, dq * dt)

        # update display
        loop += 1
        updated_display = loop % loops_to_disp == 0
        if updated_display:
            robot.display(q)
            robot.displayCollisionMarkers(collisions)

        # wait for next time step
        target_time += dt
        diff_time = target_time - time.time()
        sleep_time = max(0, diff_time)
        if sleep_time > 0:
            time.sleep(sleep_time)
        elif warn_if_not_real_time and diff_time < -0.5 and updated_display:
            print(f'Warning: Time slowed by {diff_time:0.3f} seconds. Consider '
                  'lowering the fps value, or raising the integration time (dt).')
