from rastertoolkit.raster import raster_clip, raster_clip_weighted  # noqa : F401
from rastertoolkit.shape import shape_subdivide  # noqa : F401


# module level doc-string
__doc__ = """
**rastertoolkit** package provides features for extracting data from raster files using shapes/geometries as selectors.
"""

# Use __all__ to let type checkers know what is part of the public API.
_all_ = ["raster_clip", "raster_clip_weighted", "shape_subdivide"]
