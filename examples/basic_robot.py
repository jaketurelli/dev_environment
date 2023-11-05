import copy
import pinocchio as pin
import hppfcl as fcl
import math
import numpy as np
import meshcat


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

    def getOneCollisionJacobian(self, col, res):
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
        J = (Jc2 - Jc1)[2, :]
        return J

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

    def getCollisionJacobian(self, collisions=None):
        '''From a collision list, return the Jacobian corresponding to the normal direction.  '''
        if collisions is None:
            collisions = self.getCollisionList()
        if len(collisions) == 0:
            return np.ndarray([0, self.model.nv])
        J = np.vstack([self.getOneCollisionJacobian(c, r)
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
            else:
                for i in range(self.num_collisions, num_collisions):
                    self.viz.viewer[self.patchName.format(i)].set_object(
                        meshcat.geometry.Cylinder(.01, .05), self.red_material)
            self.num_collisions = num_collisions

        # move the geometries into place for each collision point
        for i, (_, _, collision_result) in enumerate(collisions):
            contact = collision_result.getContact(0)
            R = pin.Quaternion.FromTwoVectors(
                np.array([0, 1, 0]), contact.normal).matrix()
            T = np.r_[np.c_[R, contact.pos], [[0, 0, 0, 1]]]
            self.viz.viewer[self.patchName.format(i)].set_transform(T)

    def drawFrameVelocities(self):
        for frame_id in self.body_frame_ids.values():
            self.viz.drawFrameVelocities(frame_id=frame_id)


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
