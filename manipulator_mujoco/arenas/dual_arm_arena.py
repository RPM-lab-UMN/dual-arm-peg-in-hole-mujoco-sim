from dm_control import mjcf
import os

class DualArmArena(object):
    def __init__(self) -> None:
        """
        Initializes the DualArmArena object by creating a new MJCF model and adding a checkerboard floor and lights.
        """
        self._mjcf_model = mjcf.RootElement()

        self._mjcf_model.option.timestep = 0.002
        self._mjcf_model.option.flag.warmstart = "enable"

        self._mjcf_model.compiler.texturedir = os.path.dirname(__file__) + "/../assets/"

        # TODO don't use checker floor in future
        texture = self._mjcf_model.asset.add(
            "texture",
            type="2d",
            file="arenas/light-wood.png",
            width=300,
            height=300,
        )
        grid = self._mjcf_model.asset.add(
            "material",
            name="grid",
            texture=texture,
            texrepeat=[5, 5]
        )
        self._mjcf_model.worldbody.add("geom", type="plane", size=[2, 2, 0.1], material=grid)
        self._mjcf_model.worldbody.add('camera', name='overhead', pos=[0, -0.6, 1.15], quat=[ 0, 0, 0, 1 ])
        for x in [-1, 1]:
            # TODO randomize lighting?
            self._mjcf_model.worldbody.add("light", pos=[2*x, -1, 4], dir=[-2*x, 1, -2])

    def attach(self, child, pos: list = [0, 0, 0], quat: list = [1, 0, 0, 0]) -> mjcf.Element:
        """
        Attaches a child element to the MJCF model at a specified position and orientation.

        Args:
            child: The child element to attach.
            pos: The position of the child element.
            quat: The orientation of the child element.

        Returns:
            The frame of the attached child element.
        """
        frame = self._mjcf_model.attach(child)
        frame.pos = pos
        frame.quat = quat
        return frame
    
    def attach_free(self, child,  pos: list = [0, 0, 0], quat: list = [1, 0, 0, 0]) -> mjcf.Element:
        """
        Attaches a child element to the MJCF model with a free joint.

        Args:
            child: The child element to attach.

        Returns:
            The frame of the attached child element.
        """
        frame = self.attach(child)
        frame.add('freejoint')
        frame.pos = pos
        frame.quat = quat
        return frame
    
    @property
    def mjcf_model(self) -> mjcf.RootElement:
        """
        Returns the MJCF model for the StandardArena object.

        Returns:
            The MJCF model.
        """
        return self._mjcf_model