import os
import glob
from dm_control import mjcf
from manipulator_mujoco.props.primitive import Primitive

class PegHole():
    def __init__(self, ph_type, shape, **kwargs):
        assert ph_type in ['peg', 'hole']
        # xml_path = os.path.join(
        #     os.path.dirname(__file__), 
        #     '../assets/interactables/peg_hole.xml'
        # )

        # TODO: Get assets relative path to remove the need for the peg_hole.xml file
        self.meshes_path = os.path.join(
            os.path.dirname(__file__),
            '../assets/interactables/'
        )
        # xml_path = os.path.join(
        #     os.path.dirname(__file__), 
        #     '../assets/interactables/peg_hole.xml'
        # )

        # TODO: Get assets relative path to remove the need for the peg_hole.xml file
        self.meshes_path = os.path.join(
            os.path.dirname(__file__),
            '../assets/interactables/'
        )

        xml_path = os.path.join(self.meshes_path, 'peg_hole.xml')

        xml_path = os.path.join(self.meshes_path, 'peg_hole.xml')

        self._mesh_name = f'{shape}_{ph_type}'

        self._mjcf_model = mjcf.from_path(xml_path)
        type = 'cap' if ph_type == 'peg' else 'bottle'

        self._body = self._mjcf_model.worldbody.add('body', name=self._mesh_name)

        # if type == 'cap':
        #     for i in range(len(glob.glob(os.path.join(self.meshes_path, 'meshes', 'collision_meshes',
        #                                               f'{shape}_cube_{type}_decomp_*.obj')))):
        #         self._mjcf_model.asset.add('mesh', name=self._mesh_name+f'_{i}', file=f'collision_meshes/{shape}_cube_{type}_decomp_{i}.obj')
        #         # TODO: Adjust friction parameters to be align more with real-world setup
        #         self._body.add('geom', type='mesh', name=self._mesh_name+f'_{i}', mesh=self._mesh_name+f'_{i}', friction=[1,1,1], dclass='collision')
        #     self._mjcf_model.asset.add('mesh', name=self._mesh_name+'_visual', file=f'{shape}_cube_{type}.obj')
        #     self._body.add('geom', type='mesh', name=self._mesh_name+'_visual', mesh=self._mesh_name+'_visual', dclass='visual')

        # else:
        for i in range(len(glob.glob(os.path.join(self.meshes_path, 'meshes', 'collision_meshes',
                                                    f'{shape}_cube_{type}_decomp_*.obj')))):
            self._mjcf_model.asset.add('mesh', name=self._mesh_name+f'_{i}', file=f'collision_meshes/{shape}_cube_{type}_decomp_{i}.obj')
            # TODO: Adjust friction parameters to be align more with real-world setup
            self._body.add('geom', type='mesh', name=self._mesh_name+f'_{i}', mesh=self._mesh_name+f'_{i}', friction=[0.01, 0.3, 0.0001], dclass='collision') # [0.01, 0.3, 0.0001]
        self._mjcf_model.asset.add('mesh', name=self._mesh_name+'_visual', file=f'{shape}_cube_{type}.obj')
        self._body.add('geom', type='mesh', name=self._mesh_name+'_visual', mesh=self._mesh_name+'_visual', dclass='visual')

    
    # @property
    # def geom(self):
    #     """Returns the primitive's geom, e.g., to change color or friction."""
    #     return self._body.geom[self._mesh_name]
    
    @property
    def mjcf_model(self):
        """Returns the primitive's mjcf model."""
        return self._mjcf_model

