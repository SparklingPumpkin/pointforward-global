"""Optional preprocessing, independent of Gaussian construction and fitting.

Array API: depth_mask.repair, range.measure_scene, sky.build_sky.
Dataset CLI: python -m pointforward_preprocessing --help.
LIM-r remains in the separately reusable pointforward_calibration package.
"""

MODULES = ('lim-r', 'depth-mask', 'scene-sky')
