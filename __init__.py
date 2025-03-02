import pyopencl as cl
import numpy as np
import torch
import sys
import os
import re
#import faulthandler

#faulthandler.enable()

class OpenCLContext:
    ctx: cl._cl.Context
    queue: any
    prog: cl.Program

    def __init__(self, type: str = "cl", device_id: int = 0):

        self.ctx = cl.create_some_context(False, [device_id])
        self.queue = cl.CommandQueue(self.ctx)
        print(f"Initialized OpenCL context for device {self.ctx.devices[0].name}")

        cl_code_file = open(os.path.dirname(os.path.realpath(__file__)) + "/rasterize.cl")
        cl_code = cl_code_file.read()
        cl_code_file.close()

        self.prog = cl.Program(self.ctx, cl_code).build()
        print(f"Built OpenCL code")

global_ish_context = OpenCLContext()

class fake_NVDR:

    class NVDiffRasterizerContext:
        clctx: OpenCLContext

        def __init__(self, output_db=False, type: str = "cl", device_id: int = 0, device = None):
            global global_ish_context
            self.clctx = global_ish_context

    class RasterizeGLContext(NVDiffRasterizerContext):
        def __init__(self, output_db=False, device_id: int = 0, device = None):
            super().__init__("cl",  device_id)

    class RasterizeCudaContext(NVDiffRasterizerContext):
        def __init__(self, output_db=False, device_id: int = 0, device = None):
            super().__init__("cl",  device_id)

    @staticmethod
    def rasterize(
        ctx: NVDiffRasterizerContext,
        pos: torch.FloatTensor,
        tri: torch.IntTensor,
        resolution: int | tuple[int, int],
        ranges: torch.IntTensor = None,
        grad_db: bool = True
    ) -> tuple[torch.FloatTensor, any]:
        """
        pos     - Batch * Number of verts * 4: (x, y, z, w) [clip space]\n
        tri     - Number of tris * 3\n
        returns - Batch * height * width * 4: (u, v, z/w, triangle id)\n
        Input batches are optional. Return is batched.
        """
        if type(resolution) is int:
            resolution = (resolution, resolution)
        clctx = ctx.clctx
        mf = cl.mem_flags

        results = []

        if(len(pos.size()) == 2):
            if(not(type(ranges) is torch.Tensor) or len(ranges.size()) != 2 or ranges.size()[1] != 2):
                raise RuntimeError(f"range mode - ranges must have shape [>0, 2] - {ranges.size()} passed")
            raise RuntimeError(f"range mode is not implemented")
        
        for b in range(pos.size()[0]):
            pos_buf = cl.Buffer(clctx.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=pos[b].numpy(force=True))
            tri_buf = cl.Buffer(clctx.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=tri.numpy(force=True))
            
            pixels = resolution[0] * resolution[1]
            res_cl_floats = cl.Buffer(clctx.ctx, mf.WRITE_ONLY, pixels * 4 * np.dtype(np.float32).itemsize)
            res_np_floats = np.empty(pixels * 4, dtype=np.float32)

            rasterize = clctx.prog.rasterize
            rasterize(clctx.queue, (pixels,), None, pos_buf, tri_buf, np.int32(tri.size()[0]), np.int32(resolution[1]), np.int32(resolution[0]), res_cl_floats)

            cl.enqueue_copy(clctx.queue, res_np_floats, res_cl_floats)

            results.append(torch.from_numpy(res_np_floats).to(dtype=torch.float32).reshape((resolution[1], resolution[0], 4)))

        return torch.stack(results).to(pos.device), None

    @staticmethod
    def interpolate(
        attr: torch.FloatTensor,
        rast: torch.FloatTensor,
        tri: torch.IntTensor,
        rast_db=None,
        diff_attrs=None,
    ) -> tuple[torch.FloatTensor, any]:
        """
        attr    - Batch * number of vertices * number of attributes\n
        rast    - Returned by rasterize(), batch * height * width * 4: (u, v, z/w, triangle id)\n
        tri     - Number of triangles * 3\n
        returns - Batch * height * width * number of attributes
        Input batches are optional. Return is batched.
        """
        resolution = (rast.size()[1], rast.size()[2])

        global global_ish_context
        clctx = global_ish_context
        mf = cl.mem_flags
        
        results = []
        attr_count = attr.size()[1]

        for b in range(rast.size()[0]):
            attr_buf = cl.Buffer(clctx.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=attr.numpy(force=True))
            rast_buf = cl.Buffer(clctx.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=rast[b].numpy(force=True))
            tri_buf = cl.Buffer(clctx.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=tri.numpy(force=True))

            pixels = resolution[0] * resolution[1]
            res_cl_floats = cl.Buffer(clctx.ctx, mf.WRITE_ONLY, pixels * attr_count * np.dtype(np.float32).itemsize)
            res_np_floats = np.empty(pixels * attr_count, dtype=np.float32)

            interpolate = clctx.prog.interpolate

            #import pdb; pdb.set_trace()
            interpolate(clctx.queue, (pixels,), None, attr_buf, rast_buf, tri_buf, 
                        np.int32(attr_count), np.int32(resolution[0]), np.int32(resolution[1]), res_cl_floats)

            cl.enqueue_copy(clctx.queue, res_np_floats, res_cl_floats)

            results.append(torch.from_numpy(res_np_floats).reshape((resolution[0], resolution[1], attr_count)))

        return torch.stack(results).to(attr.device), None

    # color: B H W C
    # rast: B H W 4 (output of rasterize)
    # pos: B Nv 4
    # tri: Nf 3
    # returns: B H W C
    @staticmethod
    def antialias(
        color: torch.FloatTensor,
        rast: torch.FloatTensor,
        pos: torch.FloatTensor,
        tri: torch.IntTensor,
    ) -> torch.FloatTensor:
        global global_ish_context
        ctx = global_ish_context.ctx
        return color
    
    @staticmethod
    def texture(
        tex: torch.FloatTensor,
        uv: torch.FloatTensor,
        uv_da = None,
    ) -> torch.FloatTensor:
        """
        tex     - Batch * texture height * texture width * 3\n
        uv      - Batch * height * width * 2\n
        returns - Batch * height * width * 3\n
        Input batch is optional. Output always batched.
        """

        global global_ish_context
        clctx = global_ish_context
        mf = cl.mem_flags

        if(len(tex.size()) != 4):
            raise RuntimeError(f"tex must have shape [>0, >0, >0, >0] - {tex.size()} passed")

        if(len(uv.size()) != 4 or uv.size()[3] != 2):
            raise RuntimeError(f"tex must have shape [>0, >0, >0, 2] - {uv.size()} passed")

        resolution_render = (uv.size()[1], uv.size()[2])
        resolution_tex = (tex.size()[2], tex.size()[1])
        
        results = []
        
        for b in range(uv.size()[0]):

            tex_buf = cl.Buffer(clctx.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=tex.numpy(force=True))
            uv_buf = cl.Buffer(clctx.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=uv[b].numpy(force=True))

            pixels = resolution_render[0] * resolution_render[1]
            res_cl_floats = cl.Buffer(clctx.ctx, mf.WRITE_ONLY, pixels * 3 * np.dtype(np.float32).itemsize)
            res_np_floats = np.empty(pixels * 3, dtype=np.float32)

            texture = clctx.prog.texture

            texture(clctx.queue, (pixels,), None, tex_buf, uv_buf, 
                    np.int32(resolution_tex[0]), np.int32(resolution_tex[1]), res_cl_floats)

            cl.enqueue_copy(clctx.queue, res_np_floats, res_cl_floats)

            results.append(torch.from_numpy(res_np_floats).reshape((resolution_render[0], resolution_render[1], 3)))

        return torch.stack(results).to(uv.device)


class fake_NVDR_top:
    __path__ = []
    torch = fake_NVDR

import torch_scatter as ts

def safecopy(te):
    if te == None:
        return te
    return te.cpu().detach().clone()

class torch_scatter:

    @staticmethod
    def scatter_max(src, index, dim, out = None, dim_size = None):

        newout = safecopy(out)
        res, argmax = ts.scatter_max(src.to('cpu'), index.to('cpu'), dim, newout, dim_size)

        if out != None:
            out.copy_(newout.to(src.device))
              
        return (res.to(src.device), argmax.to(src.device))

    @staticmethod
    def scatter_add(src, index, dim, out = None, dim_size = None):

        newout = safecopy(out)
        res = ts.scatter_add(src.to('cpu'), index.to('cpu'), dim, newout, dim_size)
        
        if out != None:
            out.copy_(newout.to(src.device))

        return res.to(src.device)
    
    @staticmethod
    def scatter_mean(src, index, dim=-1, out = None, dim_size = None):
        
        newout = safecopy(out)
        res = ts.scatter_mean(src.to('cpu'), index.to('cpu'), dim, newout, dim_size)
        
        if out != None:
            out.copy_(newout.to(src.device))

        return res.to(src.device)

def loadobj(filepath, paduv=0.0, padw=1):
    re_vertex = re.compile(r"v ([\-\.0-9]+) ([\-\.0-9]+) ([\-\.0-9]+)")
    re_uv = re.compile(r"vt ([\-\.0-9]+) ([\-\.0-9]+)(?: ([\-\.0-9]+))?")
    re_face = re.compile(r"f ([0-9]+)\/([0-9]+)\/[0-9]+ ([0-9]+)\/([0-9]+)\/[0-9]+ ([0-9]+)\/([0-9]+)\/[0-9]+")
    file = open(filepath)
    verts = []
    tris = []
    uvs = []
    tris_uv = []
    for line in file:

        m = re.match(re_vertex, line)
        if(m):
            if padw is None:
                verts.append((float(m[1]), float(m[2]), float(m[3])))
            else:
                verts.append((float(m[1]), float(m[2]), float(m[3]), padw))
        
        m = re.match(re_face, line)
        if(m):
            tris.append((int(m[1])-1, int(m[3])-1, int(m[5])-1))
            tris_uv.append((int(m[2])-1, int(m[4])-1, int(m[6])-1))

        m = re.match(re_uv, line)
        if(m):
            if(m[3] is None and paduv is None):
                uvs.append((float(m[1]), float(m[2])))
            else:
                uv3 = paduv
                if(m[3] is not None):
                    uv3 = float(m[3])
                uvs.append((float(m[1]), float(m[2]), uv3))
    
    return torch.tensor(verts, dtype=torch.float32), torch.tensor(tris, dtype=torch.int32), \
        torch.tensor(uvs, dtype=torch.float32), torch.tensor(tris_uv, dtype=torch.int32)

print("Patching NVDR.")
sys.modules["nvdiffrast"] = fake_NVDR_top
sys.modules["nvdiffrast.torch"] = fake_NVDR
sys.modules["torch_scatter"] = torch_scatter

def id_decorator(*args, **kwargs):
    def inner0(f):
        def inner1(*args, **kwargs):
            return f(*args, **kwargs)
        return inner1
    return inner0

if(torch.__version__ < "2.5"):
    torch.amp.custom_fwd = id_decorator
    torch.amp.custom_bwd = id_decorator

def __test_scatter():
    import torch_scatter

    src = torch.tensor([[2, 0, 1, 4, 3], [0, 2, 1, 3, 4]])
    index = torch.tensor([[4, 5, 4, 2, 3], [0, 0, 2, 2, 1]])

    out, argmax = torch_scatter.scatter_max(src, index, dim=-1)
    out_target = torch.tensor([[0, 0, 4, 3, 2, 0], [2, 4, 3, 0, 0, 0]])
    argmax_target = torch.tensor([[5, 5, 3, 4, 0, 1], [1, 4, 3, 5, 5, 5]])
    if(out.equal(out_target)):
        print("Out is correct")
    else:
        print("Out is not correct:")
        print(out)
        print("=/")
        print(out_target)
    
    if(argmax.equal(argmax_target)):
        print("Argmax is correct")
    else:
        print("Argmax is not correct:")
        print(argmax)
        print("=/")
        print(argmax_target)

if __name__ == "__main__":
    from torchvision.utils import save_image
    from torchvision.io import read_image, ImageReadMode
    import nvdiffrast.torch as dr

    from torch.amp import custom_fwd, custom_bwd

    @custom_fwd(cast_inputs=torch.float32, device_type="cuda")
    def decorator_test(val):
        print(f"Decorator test: {val}")


    decorator_test(111)

    #__test_scatter()
    
    def maprange(val, oldmin, oldmax, newmin, newmax):
        fac = (val - oldmin)/(oldmax - oldmin)
        return newmin + (newmax - newmin)*fac
    
    DEVICE = "cpu"

    verts, tris, uvs, tris_uv = loadobj("./test/quad.obj")#./test/monkey.obj
    ctx = dr.RasterizeCudaContext()

    verts = verts.unsqueeze(0)

    rast, _ = dr.rasterize(ctx, verts.to(DEVICE), tris.to(DEVICE), (1024, 1024))
    dumb_rast = rast[0].narrow(2, 0, 3).permute((2, 0, 1))
    save_image(dumb_rast, "./test/out/rast.png")
    print("Saved rast")

    interp, _ = dr.interpolate(uvs.to(DEVICE), rast.to(DEVICE), tris_uv.to(DEVICE))
    dumb_interp = interp[0].permute((2, 0, 1))
    save_image(dumb_interp, "./test/out/interp.png")
    print("Saved interp")

    tex_8b = read_image("./test/grad2.png", ImageReadMode.RGB)
    tex = tex_8b.to(torch.float32) / 255.0
    tex = tex.permute((1, 2, 0))

    #save_image(tex.permute(2, 0, 1), "./test/out/readtex.png")
    #print("saved loaded image")
    
    tex = tex.unsqueeze(0)

    uv, some_empty_channel = torch.split(interp, [2, 1], dim=3)
    uv = uv.contiguous()

    texed = dr.texture(tex.to(DEVICE), uv.to(DEVICE))
    dumb_texed = texed[0].permute((2, 0, 1))
    save_image(dumb_texed, "./test/out/texed.png")
    print("saved texed")
