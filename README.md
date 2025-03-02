# unfinished

Fake NVDiffRast module for non-NV GPUs<br><br>
Wraps on import:<br>
 - nvdiffrast.torch.rasterize, texture, interpolate, antialias
 - torch_scatter.scatter_max, scatter_add, scatter_mean
 - torch.amp.custom_fwd, custom_bwd if torch is <2.5
<br>
