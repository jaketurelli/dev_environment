
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
    for i in range(1, 6):
        q[-i] -= math.pi / 16. * (1 - 2 * (i % 2 == 0))

    # visualize robot
    robot.viz = robot.initViz(meshcat_monitor.MeshcatVisualizer)
    robot.initViewer(open=True)
    robot.loadViewerModel()
    robot.displayVisuals(True)
    robot.viz.viewer['/Cameras/default/rotated/<object>'].set_transform(
        tf.rotation_matrix(-math.pi / 2, [0.2, 0.2, 1], [1, 0, 0.7]))  # moving the camera to a better spot
    robot.display(q)
    time.sleep(0.3)  # let the user see the robot in its default pose briefly before simulating

    # simulation settings
    dt = 0.001                                  # time step size to integrate with (can be different than display fps)
    fps = 30                                    # frames per second to display
    K_contact_dist = 1 / dt                     # stiffness constant for contact distance (Proportional control)
    K_contact_vel = 1 / dt                      # stiffness constant for contact velocity (Derivative control)
    K_joint_friction = 0.05                     # joint-space friction constant
    K_slip_dist_to_force = 40.                  # stiffness constant for the slip distance (Proportional control)
    max_slip_force = 1000.                      # maximum sliding force to apply before slipping
    tau = np.zeros((robot.model.nv))            # control torques/forces in joint-space
    loops_to_disp = int((1 / fps) / dt)         # number of time steps to simulate before re-displaying
    warn_if_not_real_time = True                # produce a warning if not able to display in real time
    floor_joint_id = robot.joint_ids["floor"]
    loop_counter = 0
    target_time = time.time()

    # sim until the user closes the visualizer window
    while robot.viz.is_open():
        fs_ext = [pin.Force(np.zeros(6)) for _ in range(len(robot.model.joints))]

        # compute the model's "M" = Mass matrix and "nle" = Nonlinear force components
        # in joint-space (force due to gravity + coriolis forces)
        pin.computeAllTerms(robot.model, robot.data, q, dq)
        M = robot.data.M
        nle = robot.data.nle

        # add some joint friction (but don't apply any joint friction forces to the
        # floating 6-dof root joint, by using [-6:] slicing)
        tau[-6:] = -K_joint_friction * dq[-6:]

        # check and account for collisions
        robot.computeCollisions(q, dq)
        collisions = robot.getAndProcessCollisions(floor_joint_id)
        if not collisions:

            # simulate the resulting acceleration without collisions (forward dynamics)
            ddq = pin.aba(robot.model, robot.data, q, dq, tau, fs_ext)

        else:

            # calculate an external restorative force caused by the contact point sliding
            # past its original point parallel to the contact normal, similar to the restorative penetration
            # spring force.
            # TODO: calculate the actual contact force and add a friction cone calc, instead of using this simple force threshold
            for joint_id in robot.colliding_joints:

                # predict where the contact point will be in 0.5 time steps, to reduce overshoot
                contact_vel = pin.getVelocity(robot.model, robot.data, joint_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
                predicted_contact_pos = robot.contact_dict[joint_id].pos + 0.5 * contact_vel.vector[:3] * dt
                predicted_dist_from_first_contact = predicted_contact_pos - robot.first_contacts[joint_id].pos

                # zero out any normal-dir error, that will be handled in quadprog.solve_qp
                predicted_dist_from_first_contact[2] = 0.

                # use a stiffness constant to estimate a restorative force
                restorative_force = K_slip_dist_to_force * predicted_dist_from_first_contact / dt

                # translate the restorative_force into the joint's LOCAL reference frame (fs_ext's reference frame)
                f_sliding = robot.data.oMi[joint_id].actInv(pin.Force(restorative_force, np.zeros(3)))

                # remove angular effects
                f_sliding.angular[:] = 0.

                # if the restorative force would exceed a max slip force
                if np.linalg.norm(f_sliding.linear) > max_slip_force:
                    # allow slipping from the first contact point
                    robot.first_contacts.pop(joint_id)
                else:
                    # otherwise apply the restorative force by adding it to the existing external forces
                    fs_ext[joint_id] += f_sliding

            # Compute the joint torques with the external force
            tau_ext = pin.rnea(robot.model, robot.data, q, dq, np.zeros(robot.model.nv), fs_ext)

            # Calculate all the joint torques, correcting for nonlinear-effect terms
            Fq = (tau - nle) + (tau_ext - nle)

            # Find new joint-space accelerations (ddq) that account for the contact, by using "Gauss'
            # principle of least constraint" to solve a "convex quadratic problem" of the form:
            #     Minimize:        1/2 ddq^T M ddq - Fq^T ddq (energy in the system in quadratic form, to minimize)
            #     With constraint: C^T ddq >= d               (constrained accel in the contact normal direction)
            # The reason this works is because the "Least action principle" implies that the energy
            # in the system will be minimized in the most "realistic" path of action. Therefore the most
            # realistic constrained acceleration (ddq) is the one with the least energy, and also doesn't
            # violate the constraint during the search.
            # To understand the constraint:
            #   d is used as a penalty term for penetration, since penetration may occur with finite
            #       time steps.
            #   "C^T ddq" is just "J_q_to_c_norm ddq" which is acceleration in the contact normal direction,
            #       which we need to add the coriolis acceleration to by subtracting it from the d side.
            J_q_to_c_norm = robot.getCollisionJacobian(collisions, direction=2)  # transforms joint space to collision norm dir (index 2)
            c_norm_velocity = J_q_to_c_norm @ dq
            c_norm_distances = robot.getCollisionDistances(collisions)
            c_norm_coriolis_accel = robot.getCollisionJdotQdot(collisions)
            C = J_q_to_c_norm.T
            d = - c_norm_coriolis_accel - K_contact_dist * c_norm_distances - K_contact_vel * c_norm_velocity
            ddq, _, _, _, _, _ = quadprog.solve_qp(M, Fq, C, d)

        # integrate the acceleration (ddq) to update the model's velocities (dq), and positions (q)
        dq += ddq * dt
        q = pin.integrate(robot.model, q, dq * dt)

        # update display
        loop_counter += 1
        updated_display = loop_counter % loops_to_disp == 0
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
