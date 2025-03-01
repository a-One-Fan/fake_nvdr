 float3 barycentric(const float2 p, const float2 a, const float2 b, const float2 c){
    float2 v0 = b - a, v1 = c - a, v2 = p - a;
    float den = v0.x * v1.y - v1.x * v0.y;
    float v = (v2.x * v1.y - v1.x * v2.y) / den;
    float w = (v0.x * v2.y - v2.x * v0.y) / den;
    float u = 1.0f - v - w;

    return (float3)(u, v, w);
}

float maprange(float val, float oldmin, float oldmax, float newmin, float newmax){
    float fac = (val - oldmin) / (oldmax - oldmin);
    return newmin + fac*(newmax - newmin);
}

#define MAXDEPTH 100000.0f

__kernel void rasterize(__global const float *vert_pos, __global const int *tri_verts, const int tris_count,
    const int width, const int height, __global float *res_floats) {

    int gid = get_global_id(0);
    res_floats[gid*4+0] = 0.0f;
    res_floats[gid*4+1] = 0.0f;
    res_floats[gid*4+2] = MAXDEPTH;
    res_floats[gid*4+3] = 0.0f;
    for(int i=0; i<tris_count; i++){
        int3 vid = (int3)(tri_verts[i*3+0], tri_verts[i*3+1], tri_verts[i*3+2]);

        float2 p0 = (float2)(vert_pos[vid.x*4+0], vert_pos[vid.x*4+1]);
        float2 p1 = (float2)(vert_pos[vid.y*4+0], vert_pos[vid.y*4+1]);
        float2 p2 = (float2)(vert_pos[vid.z*4+0], vert_pos[vid.z*4+1]);

        float2 screen = (float2)(gid % width, gid / width);
        screen = (screen / (float2)(width, height)) * 2.0f - 1.0f;

        float3 uvw = barycentric(screen, p0, p1, p2);
        if(uvw.x < 0.0f || uvw.y < 0.0f || uvw.z < 0.0f) {
            continue;
        }

        float depth = dot((float3)(vert_pos[vid.x*4+2], vert_pos[vid.y*4+2], vert_pos[vid.z*4+2]), uvw);

        if(res_floats[gid*4 + 2] > depth) {
            res_floats[gid*4 + 0] = uvw.x;
            res_floats[gid*4 + 1] = uvw.y;
            res_floats[gid*4 + 2] = depth;
            res_floats[gid*4 + 3] = (float)(i+1);
        }
    }
    if(res_floats[gid*4+2] == MAXDEPTH){
        res_floats[gid*4+2] = 0.0f;
    }
}

__kernel void interpolate(__global const float *attributes, __global const float *rast, __global const int *tri_verts, 
    const int attr_count, const int width, const int height, __global float *res_floats){

    int gid = get_global_id(0);
    for(int i=0; i<attr_count; i++){
        int triid = (int)(rast[gid*4 + 3])-1;
        if(triid < 0) {
            res_floats[gid * attr_count + i] = 0.0f;
            continue;
        }

        int v0id = tri_verts[triid*3 + 0];
        int v1id = tri_verts[triid*3 + 1];
        int v2id = tri_verts[triid*3 + 2];
        
        float u = rast[gid*4 + 0];
        float v = rast[gid*4 + 1];
        float w = 1.0f - u - v;

        res_floats[gid * attr_count + i] = attributes[v0id * attr_count + i] * u + attributes[v1id * attr_count + i] * v + attributes[v2id * attr_count + i] * w;
    }
}

// extend_mode:
// 0 - return black
// 1 - extend edge pixels
// 2 - repeat image (modulo)
float3 sample_tex(__global const float *tex, const int width, const int height, const int2 coord, const int extend_mode){
    int2 coord_extended = coord;
    if(coord.x >= width || coord.x < 0 || coord.y >= height || coord.y < 0){
        if(extend_mode == 0){
            return (float3)(0.0f, 0.0f, 0.0f);
        }
        if(extend_mode == 1){
            if(coord.x >= width){
                coord_extended.x = width-1;
            }
            if(coord.x < 0){
                coord_extended.x = 0;
            }
            if(coord.y >= height){
                coord_extended.y = height-1;
            }
            if(coord.y < 0){
                coord_extended.y = 0;
            }
        }
        if(extend_mode == 2){
            coord_extended.x = (coord.x % width) * (1 - 2*(coord.x<0));
            coord_extended.y = (coord.y % height) * (1 - 2*(coord.y<0));
        }
    }

    int offs = (coord_extended.x+coord_extended.y*width)*3;
    return (float3)(tex[offs + 0], tex[offs + 1], tex[offs + 2]);
}

// tex_interpolate_mode:
// 0 - closest
// 1 - linear
// 2 - cubic - unimplemented
__kernel void texture(__global const float *tex, __global const float *render_uv, 
    const int texture_width, const int texture_height, __global float *res){

    const int tex_intepolate_mode = 1;
    const int extend_mode = 1;

    int gid = get_global_id(0);
    
    float2 uv_coord = (float2)(render_uv[gid*2+0], render_uv[gid*2+1]);

    float2 uv_id_f = uv_coord * (float2)(texture_width, texture_height); 
    int2 uv_id = (int2)(floor(uv_id_f.x), floor(uv_id_f.y));

    float3 col = (float3)(0, 0, 0);
    if(tex_intepolate_mode == 0){
        col = sample_tex(tex, texture_width, texture_height, uv_id, extend_mode);
    }
    if(tex_intepolate_mode == 1){
        float3 col1, col2, col3, col4;
        col1 = sample_tex(tex, texture_width, texture_height, (int2)(uv_id.x+0, uv_id.y+0), extend_mode);
        col2 = sample_tex(tex, texture_width, texture_height, (int2)(uv_id.x+1, uv_id.y+0), extend_mode);
        col3 = sample_tex(tex, texture_width, texture_height, (int2)(uv_id.x+0, uv_id.y+1), extend_mode);
        col4 = sample_tex(tex, texture_width, texture_height, (int2)(uv_id.x+1, uv_id.y+1), extend_mode);

        float3 row1, row2;

        float facx = uv_id_f.x - (float)(uv_id.x);
        float facy = uv_id_f.y - (float)(uv_id.y);

        row1 = col2*facx + col1*(1.0f-facx);
        row2 = col4*facx + col3*(1.0f-facx);
        col = row2*facy+row1*(1.0f-facy);
    }

    res[gid*3+0] = col.x;
    res[gid*3+1] = col.y;
    res[gid*3+2] = col.z;
}

__kernel void antialias(
    __global const float *vert_pos, __global const int *tri_verts, __global float *res_g){

    int gid = get_global_id(0);
    res_g[gid] = vert_pos[gid] + tri_verts[gid];
    // ???
}
