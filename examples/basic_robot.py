import copy
import pinocchio as pin
import hppfcl as fcl
import math
import numpy as np
import meshcat
import typing
import time
import types
import quadprog


class BasicRobot(pin.RobotWrapper):

    _default_colors = {"red": [1, 0, 0, 1],
                       "green": [0, 1, 0, 1],
                       "blue": [0, 0, 1, 1],
                       "cyan": [0, 1, 1, 1],
                       "magenta": [1, 0, 1, 1],
                       "yellow": [1, 1, 0, 1],
                       "black": [0, 0, 0, 1],
                       "white": [1, 1, 1, 1],
                       "grey": [0.7, 0.7, 0.7, 1],
                       "r": [1, 0, 0, 1],
                       "g": [0, 1, 0, 1],
                       "b": [0, 0, 1, 1],
                       "c": [0, 1, 1, 1],
                       "m": [1, 0, 1, 1],
                       "y": [1, 1, 0, 1],
                       "k": [0, 0, 0, 1],
                       "w": [1, 1, 1, 1], }

    @staticmethod
    def placement_SE3(x=0, y=0, z=0, rx=0, ry=0, rz=0):
        m = pin.SE3.Identity()
        m.translation = np.array([x, y, z])
        m.rotation *= pin.utils.rotate('x', rx)
        m.rotation *= pin.utils.rotate('y', ry)
        m.rotation *= pin.utils.rotate('z', rz)
        return m

    def __init__(self, name=None):

        # initialize model objects
        self.model = pin.Model()
        self.collision_model = pin.GeometryModel()
        self.visual_model = self.collision_model
        super().__init__(self.model, self.collision_model, self.visual_model)

        # add joints
        self.name = self.__class__.__name__ if name is None else name
        self._fixed_joint_poses = {}
        self.joint_ids = {}
        self.joint_frame_ids = {}
        self.long_joint_ids = {}
        self.body_frame_ids = {}
        self.joint_id_to_body_frame_id = {}
        self.contact_pairs = {}
        self.geom_ids = {}
        self.contact_joint_ids = {}
        self.contact_body_frame_ids = {}
        self.add_joints()

        # add contacts (must happen before "createData" calls so data is recreated with contact pairs to avoid segmentation fault)
        if self.contact_pairs:
            for name1, target_list in self.contact_pairs.items():
                if not isinstance(target_list, list):
                    target_list = [target_list]
                for name2 in target_list:
                    self.collision_model.addCollisionPair(pin.CollisionPair(self.geom_ids[name1], self.geom_ids[name2]))

        # create data and default q/v
        self.data = self.model.createData()
        self.collision_data = self.collision_model.createData()
        self.collision_data.collisionRequests.enable_contact = bool(self.contact_pairs)
        self.visual_data = self.visual_model.createData()
        self.a0 = pin.utils.zero(self.nv)
        self.v0 = pin.utils.zero(self.nv)
        self.q0 = pin.neutral(self.model)

        # to record the collision/contact data
        self.first_contacts = {}
        self.colliding_joints = set()
        self.contact_dict = {}
        self.new_collisions = set()
        self.contact_joint_to_index = {}

    def View(self, viewer, camera_position=None):
        self.initViz(viewer)
        self.initViewer(open=True)
        self.loadViewerModel()
        self.displayVisuals(True)
        if camera_position is not None:
            self.viz.viewer['/Cameras/default/rotated/<object>'].set_transform(camera_position)

    def getAndProcessCollisions(self, floor_joint_id):
        if floor_joint_id is None:
            return None
        collisions = self.getCollisionList()
        old_collisions = self.contact_dict
        self.colliding_joints = set()
        self.new_collisions = set()
        if not collisions:
            self.first_contacts = {}
        else:
            self.contact_dict = {}
            for index, (_, col, res) in enumerate(collisions):
                joint1 = self.collision_model.geometryObjects[col.first].parentJoint
                joint2 = self.collision_model.geometryObjects[col.second].parentJoint
                joint_id = joint1 if joint1 != floor_joint_id else joint2
                self.contact_joint_to_index[joint_id] = index
                self.colliding_joints.add(joint_id)
                contact = res.getContact(0)
                if joint_id in self.contact_dict and joint_id in old_collisions:
                    # NOTE: this simplification throws away all but one collision per joint
                    # by using the new contact that's closest to the old contact, to prevent jitter
                    old_contact_pos = old_collisions[joint_id].pos
                    new_pos_dist_1 = np.linalg.norm(old_contact_pos - self.contact_dict[joint_id].pos)
                    new_pos_dist_2 = np.linalg.norm(old_contact_pos - contact.pos)
                    if new_pos_dist_1 > new_pos_dist_2:
                        self.contact_dict[joint_id] = contact
                self.contact_dict[joint_id] = contact

            for joint_id in range(len(self.model.joints)):
                if joint_id in self.colliding_joints:
                    if joint_id not in self.first_contacts:
                        self.first_contacts[joint_id] = self.contact_dict[joint_id]
                        self.new_collisions.add(joint_id)
                else:
                    self.first_contacts.pop(joint_id, None)
        return collisions

    def initViz(self, class_obj):
        self.viz = class_obj(model=self.model,
                             collision_model=self.collision_model,
                             visual_model=self.visual_model,
                             copy_models=False,
                             data=self.data,
                             collision_data=self.collision_data,
                             visual_data=self.visual_data)
        return self.viz

    def add_joints(self):
        if hasattr(self, 'joints'):
            for joint in self.joints:
                self.add_joint(**joint)

    def add_joint(self, name, joint_model='float', joint_placement=None, body_placement=None, lever=None, shape="box",
                  dimensions=1, mass=None, color='grey', density=1, parent=0, contacting=False):

        # resolve the joint model
        if isinstance(joint_model, str):
            if joint_model == 'fixed':
                joint_model = None
            elif joint_model == 'float':
                joint_model = pin.JointModelFreeFlyer()
            elif 'JointModel' + joint_model in pin.__dict__:
                joint_model = pin.__dict__['JointModel' + joint_model]()
            else:
                available_joints = ['float', 'fixed'] + [
                    x[len('JointModel'):] for x in pin.__dict__
                    if isinstance(x, str) and x.startswith('JointModel')]
                raise RuntimeError(
                    f"Unrecognized joint {joint_model}. Available_joints: {available_joints}")

        # get body_inertia and geometry
        if shape == "box":
            w, h, d = (float(i) for i in dimensions) if isinstance(
                dimensions, tuple) else [float(dimensions)] * 3
            if mass is None:
                mass = w * h * d * density
            body_inertia = pin.Inertia.FromBox(mass, w, h, d)
            geometry = fcl.Box(w, h, d)
        elif shape == "cylinder":
            r, h = dimensions
            if mass is None:
                mass = math.pi * r ** 2 * h * density
            body_inertia = pin.Inertia.FromCylinder(mass, r, h)
            geometry = fcl.Cylinder(r, h)
        elif shape == "sphere":
            w, h, d = (float(i) for i in dimensions) if isinstance(dimensions, tuple) else [float(dimensions)] * 3
            if mass is None:
                mass = 4. / 3. * math.pi * w * h * d * density
            body_inertia = pin.Inertia.FromEllipsoid(mass, w, h, d)
            geometry = fcl.Sphere(dimensions)
        else:
            raise TypeError(f"unrecognized shape {shape}")

        # determine joint placement
        if joint_placement is None:
            joint_placement = pin.SE3.Identity()
        elif isinstance(joint_placement, dict):
            joint_placement = BasicRobot.placement_SE3(**joint_placement)
        # correct the placement for any fixed pose joint parents
        joint_placement = self._fixed_joint_poses.get(
            parent, pin.SE3.Identity()) * joint_placement

        # determine body placement
        if body_placement is None:
            body_placement = pin.SE3.Identity()
        elif isinstance(body_placement, dict):
            body_placement = BasicRobot.placement_SE3(**body_placement)

        # if there is no joint, then save the placement offsets for the next child joint and also adjust the current body
        if joint_model is None:
            self._fixed_joint_poses[name] = joint_placement
            body_placement = joint_placement * body_placement

        # add joint
        parent_id = self.joint_ids.get(parent, parent)  # allows numeric or str parents
        if joint_model is None:
            joint_id = parent_id
        else:
            frame_name = f'{self.name}.{name}.joint'
            joint_id = self.model.addJoint(parent_id, joint_model, joint_placement, frame_name)
            self.long_joint_ids[frame_name] = joint_id
            self.joint_frame_ids[frame_name] = self.model.addJointFrame(joint_id, -1)
        self.joint_ids[name] = joint_id
        if contacting:
            self.contact_joint_ids[name] = joint_id
            self.contact_pairs[name] = contacting

        # add body
        self.model.appendBodyToJoint(joint_id, body_inertia, body_placement)
        frame_name = f'{self.name}.{name}.body'
        self.body_frame_ids[frame_name] = self.model.addBodyFrame(frame_name, joint_id, body_placement, -1)
        self.joint_id_to_body_frame_id[joint_id] = self.body_frame_ids[frame_name]
        if contacting:
            self.contact_body_frame_ids[frame_name] = self.body_frame_ids[frame_name]

        # add geometry
        if geometry is not None:
            geom_name = f'{self.name}.{name}.geom'
            geom_obj = pin.GeometryObject(geom_name, joint_id, geometry, body_placement)
            geom_obj.meshColor = np.array(BasicRobot._default_colors.get(color, color))
            geom_id = self.collision_model.addGeometryObject(geom_obj)
            self.geom_ids[name] = geom_id

    def computeCollisions(self, q, vq=None):
        res = pin.computeCollisions(self.model, self.data, self.collision_model, self.collision_data, q, False)
        pin.computeDistances(self.model, self.data,
                             self.collision_model, self.collision_data, q)
        pin.computeJointJacobians(self.model, self.data, q)
        if vq is not None:
            pin.forwardKinematics(self.model, self.data, q, vq, 0 * vq)
        return res

    def getCollisionList(self):
        '''Return a list of triplets [ index,collision,result ] where index is the
        index of the collision pair, collision is collision_model.collisionPairs[index]
        and result is gdata.collisionResults[index].
        '''
        return [[ir, self.collision_model.collisionPairs[ir], r]
                for ir, r in enumerate(self.collision_data.collisionResults) if r.isCollision()]

    def getOneCollisionJacobian(self, col, res, dir_index):
        '''Compute the jacobian for one collision only. '''
        contact = res.getContact(0)
        g1 = self.collision_model.geometryObjects[col.first]
        g2 = self.collision_model.geometryObjects[col.second]
        oMc = pin.SE3(pin.Quaternion.FromTwoVectors(
            np.array([0, 0, 1]), contact.normal).matrix(), contact.pos)

        joint1 = g1.parentJoint
        joint2 = g2.parentJoint
        oMj1 = self.data.oMi[joint1]
        oMj2 = self.data.oMi[joint2]

        cMj1 = oMc.inverse() * oMj1
        cMj2 = oMc.inverse() * oMj2

        J1 = pin.getJointJacobian(
            self.model, self.data, joint1, pin.ReferenceFrame.LOCAL)
        J2 = pin.getJointJacobian(
            self.model, self.data, joint2, pin.ReferenceFrame.LOCAL)
        Jc1 = cMj1.action @ J1
        Jc2 = cMj2.action @ J2
        J = Jc2 - Jc1
        if dir_index is None:
            return J
        if isinstance(dir_index, list):
            return J[:dir_index[0], :]
        return J[dir_index, :]

    def getOneCollisionVelDiff(self, col, res):
        '''Compute the velocity difference between the two colliding objects. '''
        contact = res.getContact(0)
        g1 = self.collision_model.geometryObjects[col.first]
        g2 = self.collision_model.geometryObjects[col.second]
        joint1 = g1.parentJoint
        joint2 = g2.parentJoint
        vel1 = pin.getVelocity(self.model, self.data, joint1, pin.ReferenceFrame.LOCAL)
        vel2 = pin.getVelocity(self.model, self.data, joint2, pin.ReferenceFrame.LOCAL)
        return vel2 - vel1

    def getCollisionVelDiff(self, collisions=None):
        '''From a collision list, compute the velocity differences between all the pairs of colliding objects. '''
        if collisions is None:
            collisions = self.getCollisionList()
        if len(collisions) == 0:
            return np.ndarray([0, self.model.nv])
        velocities = np.vstack([self.getOneCollisionVelDiff(c, r)
                                for (i, c, r) in collisions])
        return velocities

    def getOneCollisionJdotQdot(self, col, res):
        '''Compute the Coriolis acceleration for one collision only. '''
        contact = res.getContact(0)
        g1 = self.collision_model.geometryObjects[col.first]
        g2 = self.collision_model.geometryObjects[col.second]
        oMc = pin.SE3(pin.Quaternion.FromTwoVectors(
            np.array([0, 0, 1]), contact.normal).matrix(), contact.pos)

        joint1 = g1.parentJoint
        joint2 = g2.parentJoint
        oMj1 = self.data.oMi[joint1]
        oMj2 = self.data.oMi[joint2]

        cMj1 = oMc.inverse() * oMj1
        cMj2 = oMc.inverse() * oMj2

        a1 = self.data.a[joint1]
        a2 = self.data.a[joint2]
        a = (cMj1 * a1 - cMj2 * a2).linear[2]
        return a

    def getCollisionJacobian(self, collisions=None, direction=None):
        '''From a collision list, return the Jacobian corresponding to the normal direction.  '''
        if collisions is None:
            collisions = self.getCollisionList()
        if len(collisions) == 0:
            return np.ndarray([0, self.model.nv])
        J = np.vstack([self.getOneCollisionJacobian(c, r, direction)
                      for (i, c, r) in collisions])
        return J

    def getCollisionJdotQdot(self, collisions=None):
        if collisions is None:
            collisions = self.getCollisionList()
        if len(collisions) == 0:
            return np.array([])
        a0 = np.vstack([self.getOneCollisionJdotQdot(c, r) for (i, c, r) in collisions])
        return a0.squeeze()

    def getCollisionDistances(self, collisions=None):
        if collisions is None:
            collisions = self.getCollisionList()
        if len(collisions) == 0:
            return np.array([])
        dist = np.array([self.collision_data.distanceResults[i].min_distance for (i, c, r) in collisions])
        return dist

    def displayCollisionMarkers(self, collisions=None):
        '''Display in the viewer the collision list get from getCollisionList().'''
        if self.viz is None:
            return

        # initialize if needed
        if not hasattr(self, 'num_collisions'):
            self.patchName = 'world/contact_{}'
            self.num_collisions = 0
            self.red_material = meshcat.geometry.MeshPhongMaterial()
            self.red_material.color = 0xFF0000
            self.red_material.transparent = False

        if collisions is None:
            collisions = self.getCollisionList()

        # make sure the correct number of collision geometries exist
        num_collisions = len(collisions)
        if num_collisions != self.num_collisions:
            if num_collisions < self.num_collisions:
                for i in range(num_collisions, self.num_collisions):
                    self.viz.viewer[self.patchName.format(i)].delete()
            self.num_collisions = num_collisions

        # move the geometries into place for each collision point
        for i, (_, _, collision_result) in enumerate(collisions):
            contact = collision_result.getContact(0)
            R = pin.Quaternion.FromTwoVectors(
                np.array([0, 1, 0]), contact.normal).matrix()
            T = np.r_[np.c_[R, contact.pos], [[0, 0, 0, 1]]]
            self.viz.viewer[self.patchName.format(i)].set_transform(T)
            self.viz.viewer[self.patchName.format(i)].set_object(
                meshcat.geometry.Cylinder(.01, .05), self.red_material)

    def drawFrameVelocities(self):
        for frame_id in self.body_frame_ids.values():
            self.viz.drawFrameVelocities(frame_id=frame_id)

    def forwardKinematics(
        self,
        frame_ids: typing.List[np.ndarray],
        joint_positions: typing.List[np.ndarray],
        joint_velocities: typing.Optional[np.ndarray] = None,
    ) -> typing.Union[
        typing.List[np.ndarray],
        typing.Tuple[typing.List[np.ndarray], typing.List[np.ndarray]],
    ]:
        """Compute end-effector positions (and velocities) for the given joint configuration.

        Args:
            joint_positions:  Flat list of angular joint positions.
            joint_velocities: Optional. Flat list of angular joint
                velocities.

        Returns:
            If only joint positions are given: List of end-effector
            positions. Each position is given as an np.array with x,y,z
            positions.
            If joint positions and velocities are given: Tuple with
            (i) list of end-effector positions and (ii) list of
            end-effector velocities. Each position and velocity is given
            as an np.array with x,y,z components.
        """
        pin.framesForwardKinematics(
            self.model, self.data, joint_positions
        )
        positions = [
            np.asarray(self.data.oMf[frame_id].translation).reshape(-1).tolist()
            for frame_id in frame_ids
        ]
        if joint_velocities is None:
            return positions
        else:
            pin.forwardKinematics(
                self.model, self.data, joint_positions, joint_velocities
            )
            velocities = []
            for frame_id in frame_ids:
                local_to_world_transform = pin.SE3.Identity()
                local_to_world_transform.rotation = self.data.oMf[
                    frame_id
                ].rotation
                v_local = pin.getFrameVelocity(
                    self.model, self.data, frame_id
                )
                v_world = local_to_world_transform.act(v_local)
                velocities.append(v_world.linear)
            return positions, velocities

    @staticmethod
    def mirror_joints(one_sides_values, flip_direction='y', side_is_right=False, rename_first_parent=False):
        """mirrors a list of joints into a set of left and right joints"""
        left_side = []
        right_side = []
        for cnt, item in enumerate(one_sides_values):
            left_dict = copy.deepcopy(item)
            right_dict = copy.deepcopy(item)
            for key in ['joint_placement', 'body_placement']:
                original_value = item.get(key, {}).get(flip_direction)
                if original_value is not None:
                    if side_is_right:
                        left_dict[key][flip_direction] = -original_value
                    else:
                        right_dict[key][flip_direction] = -original_value
            left_dict['name'] = 'left_' + left_dict['name']
            right_dict['name'] = 'right_' + right_dict['name']
            if cnt > 0 or rename_first_parent:
                left_dict['parent'] = 'left_' + left_dict['parent']
                right_dict['parent'] = 'right_' + right_dict['parent']
            left_side += [left_dict]
            right_side += [right_dict]
        return left_side + right_side


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
        self.K_slip_dist_to_force = 20.    # stiffness constant for the slip distance (Proportional control)
        self.max_slip_force = 100.         # maximum sliding force to apply before slipping
        self.warn_if_not_real_time = True  # produce a warning if not able to display in real time
        self.display_rate = 1.             # 1. is real-time, .5 is half-speed, 2. is sped up by 2x
        self.K_contact_dist = lambda s=self: 1 / s.dt  # stiffness constant for contact distance (Proportional control)
        self.K_contact_vel = lambda s=self: 1 / s.dt   # stiffness constant for contact velocity (Derivative control)
        self.loops_to_disp = lambda s=self: int(       # number of time steps to simulate before re-displaying
            (1 / s.fps) / (s.dt / s.display_rate))
        self.floor_joint_id = robot.joint_ids.get("floor")
        self.loop_counter = 0
        self.target_time = time.time()
        self.next_warning_time = time.time()
        self.display = False

        # apply any user specified overrides
        self.__dict__.update(user_settings)

        # process any functions that might rely on prior setting overrides last
        for name, value in self.__dict__.items():
            if isinstance(value, types.FunctionType):
                setattr(self, name, value())

    def show(self, sleep_time=None):
        self.robot.display(self.q)
        if sleep_time is not None:
            time.sleep(sleep_time)

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
        pin.computeAllTerms(self.robot.model, self.robot.data, self.q, self.dq)
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
            d = - c_norm_coriolis_accel - self.K_contact_dist * c_norm_distances - self.K_contact_vel * c_norm_velocity
            self.ddq, _, _, _, _, _ = quadprog.solve_qp(M, Fq, C, d)

        # integrate the acceleration (ddq) to update the model's velocities (dq), and positions (q)
        self.dq += self.ddq * self.dt
        self.q = pin.integrate(self.robot.model, self.q, self.dq * self.dt)

        # update display
        self.loop_counter += 1
        self.target_time += self.dt / self.display_rate
        if self.display:
            updated_display = self.loop_counter % self.loops_to_disp == 0
            if updated_display:
                self.robot.display(self.q)
                self.robot.displayCollisionMarkers(collisions)

            # wait for next time step
            new_time = time.time()
            diff_time = self.target_time - new_time
            sleep_time = max(0, diff_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            elif self.warn_if_not_real_time and diff_time < -0.5 and updated_display and new_time > self.next_warning_time:
                self.next_warning_time = new_time + 5.  # don't allow a warning or another 5 seconds
                print(f'Warning: Time slowed by {diff_time:0.3f} seconds. Consider '
                      'lowering the fps value, or raising the integration time (dt).')


BasicRobot.Sim = Sim
