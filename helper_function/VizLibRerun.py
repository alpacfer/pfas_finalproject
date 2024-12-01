from __future__ import annotations

import rerun as rr
from rerun.datatypes import Angle, Quaternion, Rotation3D, RotationAxisAngle


import argparse
from typing import Any, Iterable

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import numpy as np



# def obj2box(obj, color=(255, 0, 0),mode='cam'):

#     dim = obj['dimensions']
#     box = CustomBoxes3D(obj) # [x, y, z]
#         ],
#         radii=0.025,
#         colors=[color],
#         # labels = [f'{pos[1]:.1f}']
#     )

#     return box

def objs2boxs(objs, color=(255, 0, 0)):

    # if mode == 'cam' :
    #     pos = obj['pos']
    #     pos = [pos[0],pos[2],-pos[1]]
    # else:
    #     pos = obj['pos']
    #     pos = [-pos[1],pos[0],pos[2]]

    # if 'pos_rr' not in objs[0]:
    #     for obj in objs:
    #         obj['pos_rr'] = [obj['pos'][0],obj['pos'][2],-obj['pos'][1]]


    box = CustomBoxes3D(objs, color) # [x, y, z]

    return box


class ClassBatch(rr.ComponentBatchLike):
    """A batch of class data."""

    def __init__(self: Any, cls: npt.ArrayLike) -> None:
        self.cls = cls

    def component_name(self) -> str:
        """The name of the custom component."""
        return "Class"

    def as_arrow_array(self) -> pa.Array:
        """The arrow batch representing the custom component."""
        return pa.array(self.cls, type=pa.string())
    
class OcclusionBatch(rr.ComponentBatchLike):
    """A batch of occlusion data."""

    def __init__(self: Any, occlusions: npt.ArrayLike) -> None:
        self.occlusions = occlusions

    def component_name(self) -> str:
        """The name of the custom component."""
        return "Occlusion"

    def as_arrow_array(self) -> pa.Array:
        """The arrow batch representing the custom component."""
        return pa.array(self.occlusions, type=pa.float32())
    
class TruncationBatch(rr.ComponentBatchLike):
    """A batch of truncation data."""

    def __init__(self: Any, truncations: npt.ArrayLike) -> None:
        self.truncations = truncations

    def component_name(self) -> str:
        """The name of the custom component."""
        return "Truncation"

    def as_arrow_array(self) -> pa.Array:
        """The arrow batch representing the custom component."""
        return pa.array(self.truncations, type=pa.float32())
    
class TrackIDBatch(rr.ComponentBatchLike):
    """A batch of trackID data."""

    def __init__(self: Any, trackID: npt.ArrayLike) -> None:
        self.trackID = trackID

    def component_name(self) -> str:
        """The name of the custom component."""
        return "TrackID"

    def as_arrow_array(self) -> pa.Array:
        """The arrow batch representing the custom component."""
        return pa.array(self.trackID, type=pa.int32())

class CustomBoxes3D(rr.AsComponents):
    """A custom archetype that extends Rerun's builtin `Points3D` archetype with a custom component."""

    def __init__(self: Any, objs: npt.ArrayLike, color) -> None:
        self.objs = objs
        self.color = color
        # self.occlusions = occlusions
        # self.truncations = truncations

    def as_component_batches(self) -> Iterable[rr.ComponentBatchLike]:
        return (
            list(rr.Boxes3D(
                centers=[[obj['location'][0],obj['location'][2],-obj['location'][1]] for obj in self.objs], # [x, y, z]
                sizes=[obj['dimensions'] for obj in self.objs], # [x, y, z]
                rotations=[
                    RotationAxisAngle(axis=[0, 0, 1], angle=Angle(rad=-obj['rotation_y'])) # [x, y, z]
                for obj in self.objs],
                radii=0.025,
                colors=[self.color for obj in self.objs],
                # labels = [f'{pos[1]:.1f}']
            ).as_component_batches())  # The components from Points3D
            + [rr.IndicatorComponentBatch("KITTI Objects")]  # Our custom indicator
            + [ClassBatch([obj['type'] for obj in self.objs])]  # Custom confidence data
            + [TrackIDBatch([obj['track_id'] for obj in self.objs])]  # Custom confidence data
            + [OcclusionBatch([obj['occluded'] for obj in self.objs])]  # Custom confidence data
            + [TruncationBatch([obj['truncated'] for obj in self.objs])]  # Custom confidence data
        )