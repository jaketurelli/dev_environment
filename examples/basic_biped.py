
import basic_robot as br
import pinocchio as pin
import numpy as np

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
        'dimensions': (pelvis_depth * 0.4, leg_length * 0.8),
        'mass': leg_mass,
        'color': 'white'
    }, {
        'name': "lower_leg",
        'parent': "upper_leg_forward_rotation",
        'joint_model': "RY",
        'joint_placement': {'z': -leg_length},
        'body_placement': {'z': -leg_length / 2.},
        'shape': "cylinder",
        'dimensions': (pelvis_depth * 0.4, leg_length * 0.8),
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
    import meshcat_monitor
    import quadprog

    robot = BasicBiped()

    robot.viz = robot.initViz(meshcat_monitor.MeshcatVisualizer)
    robot.initViewer(open=True)
    robot.loadViewerModel()
    robot.displayVisuals(True)
    robot.q0[-2] -= math.pi / 8.
    robot.display(robot.q0)
    if robot.viz.is_open():
        robot.viz.viewer["/Cameras/default/rotated/<object>"].set_property("zoom", 3.0)
        time.sleep(0.6)

    dt = 0.001
    fps = 30
    Kp_c = 1000.
    Kv_c = 1000.
    Kv_impact = 0.1
    Kf = 0.01
    tau = np.zeros((robot.model.nv))
    loops_per_display = int((1 / fps) / dt)

    loop = 0
    target_time = time.time()
    while robot.viz.is_open():

        # Compute the model.
        M = pin.crba(robot.model, robot.data, robot.q0)
        b = pin.nle(robot.model, robot.data, robot.q0, robot.v0)

        # add some joint friction (but don't apply any joint friction forces to the floating 6-dof root joint)
        tau[-6:] = -Kf * robot.v0[-6:]

        # Simulated the resulting acceleration (forward dynamics)
        joint_forces = tau - b
        aq = np.linalg.inv(M) @ joint_forces  # F = M*a

        # Check collision
        robot.computeCollisions(robot.q0, robot.v0)
        collisions = robot.getCollisionList()
        if not collisions:
            already_contact = set()
        else:
            dist = robot.getCollisionDistances(collisions)
            J = robot.getCollisionJacobian(collisions)
            JdotQdot = robot.getCollisionJdotQdot(collisions)

            # Update contact tracking and nullify velocity of new contact
            col_id = [e[0] for e in collisions]
            new_col_idx = [i for i, e in enumerate(col_id) if e not in already_contact]
            already_contact = set(col_id)
            if new_col_idx:
                J_proj = np.stack([J[i] for i in new_col_idx], axis=0)
                impact_impulse_vel = (np.linalg.pinv(J_proj) @ J_proj) @ robot.v0
                robot.v0 -= impact_impulse_vel * Kv_impact

            # Find real acceleration using Gauss principle
            A = M
            b = joint_forces  # same as "M @ aq"
            C = J
            d = - JdotQdot - Kp_c * dist - Kv_c * J @ robot.v0
            [aq, cost, _, niter, lag, iact] = quadprog.solve_qp(A, b, C.T, d)

        # Integrate the acceleration.
        robot.v0 += aq * dt
        robot.q0 = pin.integrate(robot.model, robot.q0, robot.v0 * dt)

        # update display
        loop += 1
        updated_display = loop % loops_per_display == 0
        if updated_display:
            robot.display(robot.q0)
            robot.displayCollisionMarkers(collisions)

        # wait for next time step
        target_time += dt
        diff_time = target_time - time.time()
        sleep_time = max(0, diff_time)
        if sleep_time > 0:
            time.sleep(sleep_time)
        elif diff_time < -0.5 and updated_display:
            print(f'Warning: Time slowed by {diff_time:0.3f} seconds. Consider '
                  'lowering the fps value, or raising the integration time (dt).')
