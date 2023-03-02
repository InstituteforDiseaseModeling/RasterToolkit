
from rastertools.data import download, get_remote, get_metadata
from rastertools.raster import raster_clip, raster_clip_weighted, get_tiff_tags
from rastertools.shape import ShapeView, area_sphere, centroid_area, shape_subdivide


# module level doc-string
__doc__ = """
**rastertools** package provides features for extracting data from raster files using shapes/geometries as selectors.  
"""

# Use __all__ to let type checkers know what is part of the public API.
_all_ = ['get_remote', 'download', 'get_metadata', 'raster_clip', 'raster_clip_weighted', 'get_tiff_tags',
         'ShapeView', 'area_sphere', 'centroid_area', 'shape_subdivide']

