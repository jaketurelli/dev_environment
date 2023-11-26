import time
import types
import quadprog
import pinocchio as pin
import numpy as np
from robotics.basic_robot import BasicRobot


class Sim():
    def __init__(self, robot: BasicRobot, **user_settings):

        # assigning defaults
        self.robot = robot
        self.q = robot.q0.copy()           # positions in joint-space
        self.dq = robot.v0.copy()          # velocities in joint-space (derivative of q)
        self.ddq = robot.a0.copy()         # accelerations in joint-space (derivative of dq)
        self.dt = 0.0005                   # time step size to integrate with (can be different than display fps)
        self.fps = 30                      # frames per second to display
        self.K_joint_friction = 0.1        # joint-space friction constant
        self.K_slip_dist_to_force = 50.    # stiffness constant for the slip distance (Proportional control)
        self.max_slip_force = 50.          # maximum sliding force to apply before slipping
        self.warn_if_not_real_time = True  # produce a warning if not able to display in real time
        self.display_rate = 1.             # 1. is real-time, .5 is half-speed, 2. is sped up by 2x
        self.K_contact_dist = 1.           # stiffness constant for contact distance (Proportional control)
        self.K_contact_vel = 1.            # stiffness constant for contact velocity (Derivative control)
        self.floor_joint_id = robot.joint_ids.get("floor")
        self.display = False

        # apply any user specified overrides
        self.__dict__.update(user_settings)

        # process any functions that might rely on prior setting overrides last
        for name, value in self.__dict__.items():
            if isinstance(value, types.FunctionType):
                setattr(self, name, value())

        # create some non-overridable tracking variables
        self.loop_counter = 0
        self.start_time = time.time()
        self.target_time = 0.
        self.next_warning_time = self.start_time
        self.last_update = -1

        # update all the terms once, but don't remember the update
        self.updateAllTermsForLoopIteration()
        self.last_update = -1

    def updateAllTermsForLoopIteration(self):
        if self.last_update != self.loop_counter:
            pin.computeAllTerms(self.robot.model, self.robot.data, self.q, self.dq)
            self.last_update = self.loop_counter

    def show(self, sleep_time=None):
        self.robot.display(self.q)
        if sleep_time is not None:
            time.sleep(sleep_time)
        self.updateAllTermsForLoopIteration()

    def step(self, tau=None, fs_ext=None, **new_settings):
        for name, value in new_settings.items():
            if not hasattr(self, name):
                raise AttributeError(f"Sim class doesn't have a setting called: {name}")
            setattr(self, name, value)
        if fs_ext is None:
            fs_ext = [pin.Force(np.zeros(6)) for _ in range(len(self.robot.model.joints))]

        if tau is None:
            tau = np.zeros((self.robot.model.nv))  # control torques/forces in joint-space

        # compute the model's "M" = Mass matrix and "nle" = Nonlinear force components
        # in joint-space (force due to gravity + coriolis forces)
        self.updateAllTermsForLoopIteration()
        M = self.robot.data.M
        nle = self.robot.data.nle

        # add some joint friction (but don't apply any joint friction forces to the
        # floating 6-dof root joint, by using [6:] slicing)
        tau[6:] += -self.K_joint_friction * self.dq[6:]

        # check and account for collisions
        self.robot.computeCollisions(self.q, self.dq)
        collisions = self.robot.getAndProcessCollisions(self.floor_joint_id)
        if not collisions:

            # simulate the resulting acceleration without collisions (forward dynamics)
            self.ddq = pin.aba(self.robot.model, self.robot.data, self.q, self.dq, tau, fs_ext)

        else:

            # calculate an external restorative force caused by the contact point sliding
            # past its original point parallel to the contact normal, similar to the restorative penetration
            # spring force.
            # TODO: calculate the actual contact force and add a friction cone calc, instead of using this simple force threshold
            for joint_id in self.robot.colliding_joints:

                # predict where the contact point will be in 0.5 time steps, to reduce overshoot
                contact_vel = pin.getVelocity(self.robot.model, self.robot.data, joint_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
                predicted_contact_pos = self.robot.contact_dict[joint_id].pos + 0.5 * contact_vel.vector[:3] * self.dt
                predicted_dist_from_first_contact = predicted_contact_pos - self.robot.first_contacts[joint_id].pos

                # zero out any normal-dir error, that will be handled in quadprog.solve_qp
                predicted_dist_from_first_contact[2] = 0.

                # use a stiffness constant to estimate a restorative force
                restorative_force = self.K_slip_dist_to_force * predicted_dist_from_first_contact / self.dt

                # translate the restorative_force into the joint's LOCAL reference frame (fs_ext's reference frame)
                f_sliding = self.robot.data.oMi[joint_id].actInv(pin.Force(restorative_force, np.zeros(3)))

                # remove angular effects
                f_sliding.angular[:] = 0.

                # if the restorative force would exceed a max slip force
                if np.linalg.norm(f_sliding.linear) > self.max_slip_force:
                    # allow slipping from the first contact point
                    self.robot.first_contacts.pop(joint_id)
                else:
                    # otherwise apply the restorative force by adding it to the existing external forces
                    fs_ext[joint_id] += f_sliding

            # Compute the joint torques with the external force
            tau_ext = pin.rnea(self.robot.model, self.robot.data, self.q, self.dq, np.zeros(self.robot.model.nv), fs_ext)

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
            J_q_to_c_norm = self.robot.getCollisionJacobian(collisions, direction=2)  # transforms joint space to collision norm dir (index 2)
            c_norm_velocity = J_q_to_c_norm @ self.dq
            c_norm_distances = self.robot.getCollisionDistances(collisions)
            c_norm_coriolis_accel = self.robot.getCollisionJdotQdot(collisions)
            C = J_q_to_c_norm.T
            d = - c_norm_coriolis_accel - (
                self.K_contact_dist * c_norm_distances + self.K_contact_vel * c_norm_velocity) / self.dt
            self.ddq, _, _, _, _, _ = quadprog.solve_qp(M, Fq, C, d)

        # integrate the acceleration (ddq) to update the model's velocities (dq), and positions (q)
        self.dq += self.ddq * self.dt
        self.q = pin.integrate(self.robot.model, self.q, self.dq * self.dt)

        # compute all terms after update
        self.loop_counter += 1
        self.updateAllTermsForLoopIteration()

        # update display
        self.target_time += self.dt / self.display_rate
        if self.display:
            loops_to_disp = int(self.display_rate / (self.fps * self.dt))  # number of steps to simulate before re-displaying
            updated_display = self.loop_counter % loops_to_disp == 0
            if updated_display:
                self.robot.display(self.q)
                self.robot.displayCollisionMarkers(collisions)

            # wait for next time step
            new_time = time.time()
            diff_time = self.target_time + self.start_time - new_time
            sleep_time = max(0, diff_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            elif self.warn_if_not_real_time and diff_time < -0.5 and updated_display and new_time > self.next_warning_time:
                self.next_warning_time = new_time + 5.  # don't allow a warning or another 5 seconds
                print(f'Warning: Time slowed by {diff_time:0.3f} seconds. Consider '
                      'lowering the fps value, or raising the integration time (dt).')
