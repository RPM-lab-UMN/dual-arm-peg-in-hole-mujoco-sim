import os
from manipulator_mujoco.robots.gripper import Gripper

_2F85_XML = os.path.join(
    os.path.dirname(__file__),
    '../assets/robots/robotiq_2f85/2f85.xml',
)

_JOINT = 'right_driver_joint'

_ACTUATOR = 'fingers_actuator'

class Robotiq_2F85(Gripper):
    def __init__(self, name: str = None):
        super().__init__(_2F85_XML, _JOINT, _ACTUATOR, name)
        self._object_site = self._mjcf_root.find('site', 'pinch')
        self._force_sensor = self._mjcf_root.find('sensor', 'force_ee')
        self._torque_sensor = self._mjcf_root.find('sensor', 'torque_ee')

    @property
    def object_site(self):
        return self._object_site
    
    @property
    def force_sensor(self):
        return self._force_sensor
    
    @property
    def torque_sensor(self):
        return self._torque_sensor
    
    def attach_object(self, child, pos: list = [0, 0, 0], quat: list = [1, 0, 0, 0]):
        frame = self._object_site.attach(child._mjcf_model)
        frame.pos = pos
        frame.quat = quat
        return frame