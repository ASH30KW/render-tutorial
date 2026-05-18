import trimesh
import numpy as np
import glfw
import glm
from OpenGL.GL import *
from OpenGL.GL import shaders
from PIL import Image
import sys
import ctypes

# ============ 加载 GLB 文件 ============

scene = trimesh.load("/home/ai/Desktop/chair.glb")

cam = scene.camera
cam_transform = scene.camera_transform
cam_pos = cam_transform[:3, 3].copy()
cam_fov = float(cam.fov[1])
cam_res = cam.resolution

lights_data = []
for light in scene.lights:
    color = np.array(light.color[:3], dtype=np.float32) / 255.0
    transform, _ = scene.graph.get(light.name)
    pos = transform[:3, 3].copy()
    lights_data.append({"color": color, "intensity": light.intensity, "position": pos})

mesh_data = []
for name, geom in scene.geometry.items():
    mat = geom.visual.material
    metallic = mat.metallicFactor if mat.metallicFactor is not None else 0.0
    roughness = mat.roughnessFactor if mat.roughnessFactor is not None else 0.5
    mesh_data.append({
        "vertices": np.array(geom.vertices, dtype=np.float32),
        "normals": np.array(geom.vertex_normals, dtype=np.float32),
        "uvs": np.array(geom.visual.uv, dtype=np.float32),
        "faces": np.array(geom.faces, dtype=np.uint32),
        "tex_img": mat.baseColorTexture,
        "metallic": metallic,
        "roughness": roughness,
    })

combined = trimesh.util.concatenate(scene.dump())
bounds = combined.bounds
center = (bounds[0] + bounds[1]) / 2.0

# ============ GLSL 着色器源码 ============

# --- 生成程序化 HDR 环境 cubemap ---
EQUIRECT_TO_CUBE_VS = """
#version 330 core
layout(location=0) in vec3 aPos;
out vec3 localPos;
uniform mat4 projection;
uniform mat4 view;
void main() {
    localPos = aPos;
    gl_Position = projection * view * vec4(aPos, 1.0);
}
"""

EQUIRECT_TO_CUBE_FS = """
#version 330 core
out vec4 FragColor;
in vec3 localPos;

// 程序化 HDR 环境（模拟室内工作室光照）
vec3 proceduralEnv(vec3 dir) {
    vec3 d = normalize(dir);

    // 基础天空渐变
    float t = d.y * 0.5 + 0.5;
    vec3 sky = mix(vec3(0.3, 0.3, 0.35), vec3(0.7, 0.75, 0.85), t);

    // 地面
    if (d.y < 0.0) {
        sky = mix(vec3(0.2, 0.2, 0.2), sky, smoothstep(-0.5, 0.0, d.y));
    }

    // 主区域光（上方，模拟柔光箱）
    float topLight = smoothstep(0.7, 1.0, d.y);
    sky += vec3(2.0, 1.9, 1.8) * topLight;

    // 前方补光
    float frontLight = pow(max(dot(d, normalize(vec3(0.0, 0.3, 1.0))), 0.0), 8.0);
    sky += vec3(1.0, 1.0, 1.1) * frontLight * 0.5;

    // 侧面补光
    float sideLight = pow(max(dot(d, normalize(vec3(1.0, 0.2, 0.0))), 0.0), 6.0);
    sky += vec3(0.8, 0.7, 0.6) * sideLight * 0.3;

    float sideLight2 = pow(max(dot(d, normalize(vec3(-1.0, 0.2, 0.0))), 0.0), 6.0);
    sky += vec3(0.6, 0.7, 0.8) * sideLight2 * 0.3;

    return sky;
}

void main() {
    FragColor = vec4(proceduralEnv(localPos), 1.0);
}
"""

# --- 辐照度卷积（漫反射 IBL）---
IRRADIANCE_FS = """
#version 330 core
out vec4 FragColor;
in vec3 localPos;
uniform samplerCube environmentMap;
const float PI = 3.14159265359;

void main() {
    vec3 normal = normalize(localPos);
    vec3 irradiance = vec3(0.0);

    vec3 up = vec3(0.0, 1.0, 0.0);
    vec3 right = normalize(cross(up, normal));
    up = normalize(cross(normal, right));

    float sampleDelta = 0.025;
    float nrSamples = 0.0;
    for (float phi = 0.0; phi < 2.0 * PI; phi += sampleDelta) {
        for (float theta = 0.0; theta < 0.5 * PI; theta += sampleDelta) {
            vec3 tangentSample = vec3(sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta));
            vec3 sampleVec = tangentSample.x * right + tangentSample.y * up + tangentSample.z * normal;
            irradiance += texture(environmentMap, sampleVec).rgb * cos(theta) * sin(theta);
            nrSamples++;
        }
    }
    irradiance = PI * irradiance / nrSamples;
    FragColor = vec4(irradiance, 1.0);
}
"""

# --- 预过滤环境贴图（镜面反射 IBL）---
PREFILTER_FS = """
#version 330 core
out vec4 FragColor;
in vec3 localPos;
uniform samplerCube environmentMap;
uniform float roughness;
const float PI = 3.14159265359;

float RadicalInverse_VdC(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10;
}

vec2 Hammersley(uint i, uint N) {
    return vec2(float(i)/float(N), RadicalInverse_VdC(i));
}

vec3 ImportanceSampleGGX(vec2 Xi, vec3 N, float r) {
    float a = r * r;
    float phi = 2.0 * PI * Xi.x;
    float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a*a - 1.0) * Xi.y));
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
    vec3 H = vec3(cos(phi)*sinTheta, sin(phi)*sinTheta, cosTheta);
    vec3 up = abs(N.z) < 0.999 ? vec3(0,0,1) : vec3(1,0,0);
    vec3 tangent = normalize(cross(up, N));
    vec3 bitangent = cross(N, tangent);
    return normalize(tangent*H.x + bitangent*H.y + N*H.z);
}

void main() {
    vec3 N = normalize(localPos);
    vec3 R = N;
    vec3 V = R;
    const uint SAMPLE_COUNT = 1024u;
    float totalWeight = 0.0;
    vec3 prefilteredColor = vec3(0.0);
    for (uint i = 0u; i < SAMPLE_COUNT; i++) {
        vec2 Xi = Hammersley(i, SAMPLE_COUNT);
        vec3 H = ImportanceSampleGGX(Xi, N, roughness);
        vec3 L = normalize(2.0 * dot(V, H) * H - V);
        float NdotL = max(dot(N, L), 0.0);
        if (NdotL > 0.0) {
            prefilteredColor += texture(environmentMap, L).rgb * NdotL;
            totalWeight += NdotL;
        }
    }
    prefilteredColor /= totalWeight;
    FragColor = vec4(prefilteredColor, 1.0);
}
"""

# --- BRDF LUT 生成 ---
BRDF_VS = """
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec2 aUV;
out vec2 TexCoords;
void main() {
    TexCoords = aUV;
    gl_Position = vec4(aPos, 1.0);
}
"""

BRDF_FS = """
#version 330 core
out vec2 FragColor;
in vec2 TexCoords;
const float PI = 3.14159265359;

float RadicalInverse_VdC(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10;
}
vec2 Hammersley(uint i, uint N) {
    return vec2(float(i)/float(N), RadicalInverse_VdC(i));
}
vec3 ImportanceSampleGGX(vec2 Xi, vec3 N, float r) {
    float a = r*r;
    float phi = 2.0*PI*Xi.x;
    float cosTheta = sqrt((1.0-Xi.y)/(1.0+(a*a-1.0)*Xi.y));
    float sinTheta = sqrt(1.0-cosTheta*cosTheta);
    vec3 H = vec3(cos(phi)*sinTheta, sin(phi)*sinTheta, cosTheta);
    vec3 up = abs(N.z)<0.999 ? vec3(0,0,1) : vec3(1,0,0);
    vec3 tangent = normalize(cross(up,N));
    vec3 bitangent = cross(N,tangent);
    return normalize(tangent*H.x+bitangent*H.y+N*H.z);
}
float GeometrySchlickGGX(float NdotV, float r) {
    float a=r; float k=a*a/2.0;
    return NdotV/(NdotV*(1.0-k)+k);
}
float GeometrySmith(vec3 N, vec3 V, vec3 L, float r) {
    return GeometrySchlickGGX(max(dot(N,V),0.0),r)*GeometrySchlickGGX(max(dot(N,L),0.0),r);
}
vec2 IntegrateBRDF(float NdotV, float r) {
    vec3 V = vec3(sqrt(1.0-NdotV*NdotV), 0.0, NdotV);
    float A=0.0, B=0.0;
    vec3 N=vec3(0,0,1);
    const uint SAMPLE_COUNT=1024u;
    for(uint i=0u;i<SAMPLE_COUNT;i++){
        vec2 Xi=Hammersley(i,SAMPLE_COUNT);
        vec3 H=ImportanceSampleGGX(Xi,N,r);
        vec3 L=normalize(2.0*dot(V,H)*H-V);
        float NdotL=max(L.z,0.0);
        float NdotH=max(H.z,0.0);
        float VdotH=max(dot(V,H),0.0);
        if(NdotL>0.0){
            float G=GeometrySmith(N,V,L,r);
            float G_Vis=G*VdotH/(NdotH*NdotV);
            float Fc=pow(1.0-VdotH,5.0);
            A+=(1.0-Fc)*G_Vis;
            B+=Fc*G_Vis;
        }
    }
    return vec2(A,B)/float(SAMPLE_COUNT);
}
void main(){
    FragColor=IntegrateBRDF(TexCoords.x,TexCoords.y);
}
"""

# --- 主 PBR 着色器（带 IBL）---
PBR_VS = """
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
layout(location=2) in vec2 aUV;
out vec3 FragPos;
out vec3 Normal;
out vec2 UV;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat3 normalMatrix;
void main(){
    FragPos = vec3(model * vec4(aPos,1.0));
    Normal = normalize(normalMatrix * aNormal);
    UV = aUV;
    gl_Position = projection * view * vec4(FragPos,1.0);
}
"""

PBR_FS = """
#version 330 core
out vec4 FragColor;
in vec3 FragPos;
in vec3 Normal;
in vec2 UV;

uniform sampler2D albedoMap;
uniform samplerCube irradianceMap;
uniform samplerCube prefilterMap;
uniform sampler2D brdfLUT;
uniform vec3 camPos;
uniform float metallic;
uniform float roughness;

struct PointLight {
    vec3 position;
    vec3 color;
    float intensity;
};
#define MAX_LIGHTS 4
uniform int numLights;
uniform PointLight lights[MAX_LIGHTS];

const float PI = 3.14159265359;

float DistributionGGX(vec3 N, vec3 H, float r){
    float a=r*r; float a2=a*a;
    float NdotH=max(dot(N,H),0.0); float d=NdotH*NdotH*(a2-1.0)+1.0;
    return a2/(PI*d*d);
}
float GeometrySchlickGGX(float NdotV, float r){
    float k=(r+1.0)*(r+1.0)/8.0;
    return NdotV/(NdotV*(1.0-k)+k);
}
float GeometrySmith(vec3 N, vec3 V, vec3 L, float r){
    return GeometrySchlickGGX(max(dot(N,V),0.0),r)*GeometrySchlickGGX(max(dot(N,L),0.0),r);
}
vec3 FresnelSchlick(float cosTheta, vec3 F0){
    return F0+(1.0-F0)*pow(clamp(1.0-cosTheta,0.0,1.0),5.0);
}
vec3 FresnelSchlickRoughness(float cosTheta, vec3 F0, float r){
    return F0+(max(vec3(1.0-r),F0)-F0)*pow(clamp(1.0-cosTheta,0.0,1.0),5.0);
}

void main(){
    vec3 albedo = pow(texture(albedoMap, UV).rgb, vec3(2.2));
    vec3 N = normalize(Normal);
    vec3 V = normalize(camPos - FragPos);
    vec3 R = reflect(-V, N);
    float NdotV = max(dot(N,V), 0.0);

    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    float r = max(roughness, 0.05);

    // === 直接光照（点光源）===
    vec3 Lo = vec3(0.0);
    for(int i=0; i<numLights; i++){
        vec3 L = normalize(lights[i].position - FragPos);
        vec3 H = normalize(V+L);
        float dist = length(lights[i].position - FragPos);
        float attenuation = 1.0/(dist*dist);
        vec3 radiance = lights[i].color * lights[i].intensity * attenuation;

        float NDF = DistributionGGX(N,H,r);
        float G = GeometrySmith(N,V,L,r);
        vec3 F = FresnelSchlick(max(dot(H,V),0.0), F0);
        vec3 spec = NDF*G*F / (4.0*NdotV*max(dot(N,L),0.0)+0.0001);
        vec3 kD = (vec3(1.0)-F)*(1.0-metallic);
        Lo += (kD*albedo/PI + spec)*radiance*max(dot(N,L),0.0);
    }

    // === IBL 环境光照 ===
    vec3 F = FresnelSchlickRoughness(NdotV, F0, r);
    vec3 kS = F;
    vec3 kD = (1.0 - kS) * (1.0 - metallic);

    // 漫反射 IBL
    vec3 irradiance = texture(irradianceMap, N).rgb;
    vec3 diffuse = irradiance * albedo;

    // 镜面反射 IBL
    const float MAX_REFLECTION_LOD = 4.0;
    vec3 prefilteredColor = textureLod(prefilterMap, R, r * MAX_REFLECTION_LOD).rgb;
    vec2 brdf = texture(brdfLUT, vec2(NdotV, r)).rg;
    vec3 specular = prefilteredColor * (F * brdf.x + brdf.y);

    vec3 ambient = kD * diffuse + specular;
    vec3 color = ambient + Lo;

    // tone mapping + gamma
    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0/2.2));

    FragColor = vec4(color, 1.0);
}
"""

# --- 天空盒着色器 ---
SKYBOX_VS = """
#version 330 core
layout(location=0) in vec3 aPos;
out vec3 localPos;
uniform mat4 projection;
uniform mat4 view;
void main(){
    localPos = aPos;
    vec4 pos = projection * mat4(mat3(view)) * vec4(aPos, 1.0);
    gl_Position = pos.xyww;
}
"""

SKYBOX_FS = """
#version 330 core
out vec4 FragColor;
in vec3 localPos;
uniform samplerCube environmentMap;
void main(){
    vec3 color = texture(environmentMap, localPos).rgb;
    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0/2.2));
    FragColor = vec4(color, 1.0);
}
"""

# --- 光源标记着色器 ---
LIGHT_VS = """
#version 330 core
layout(location=0) in vec3 aPos;
uniform mat4 mvp;
void main(){ gl_Position = mvp * vec4(aPos,1.0); }
"""

LIGHT_FS = """
#version 330 core
out vec4 FragColor;
uniform vec3 lightColor;
void main(){ FragColor = vec4(lightColor, 1.0); }
"""

# ============ 交互 ============

rotation_x = 0.0
rotation_y = 0.0
zoom_offset = 0.0
last_mouse = [0.0, 0.0]
mouse_left = False
mouse_right = False

def mouse_button_callback(window, button, action, mods):
    global mouse_left, mouse_right, last_mouse
    x, y = glfw.get_cursor_pos(window)
    last_mouse = [x, y]
    if button == glfw.MOUSE_BUTTON_LEFT:
        mouse_left = (action == glfw.PRESS)
    elif button == glfw.MOUSE_BUTTON_RIGHT:
        mouse_right = (action == glfw.PRESS)

def cursor_pos_callback(window, x, y):
    global rotation_x, rotation_y, zoom_offset, last_mouse
    dx = x - last_mouse[0]
    dy = y - last_mouse[1]
    if mouse_left:
        rotation_y += dx * 0.3
        rotation_x += dy * 0.3
    if mouse_right:
        zoom_offset -= dy * 0.005
        zoom_offset = max(-1.0, min(3.0, zoom_offset))
    last_mouse = [x, y]

def scroll_callback(window, xoff, yoff):
    global zoom_offset
    zoom_offset += yoff * 0.1
    zoom_offset = max(-1.0, min(3.0, zoom_offset))

def key_callback(window, key, scancode, action, mods):
    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        glfw.set_window_should_close(window, True)

# ============ OpenGL 工具函数 ============

def compile_shader(vs_src, fs_src):
    vs = shaders.compileShader(vs_src, GL_VERTEX_SHADER)
    fs = shaders.compileShader(fs_src, GL_FRAGMENT_SHADER)
    return shaders.compileProgram(vs, fs)

def upload_texture_2d(img):
    img = img.transpose(Image.FLIP_TOP_BOTTOM).convert("RGBA")
    data = img.tobytes()
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_SRGB_ALPHA, img.width, img.height, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, data)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glGenerateMipmap(GL_TEXTURE_2D)
    return tex

def create_mesh_vao(verts, normals, uvs, faces):
    interleaved = np.hstack([verts, normals, uvs]).astype(np.float32)
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    ebo = glGenBuffers(1)
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, interleaved.nbytes, interleaved, GL_STATIC_DRAW)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, faces.nbytes, faces, GL_STATIC_DRAW)
    stride = 8 * 4
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(24))
    glEnableVertexAttribArray(2)
    glBindVertexArray(0)
    return vao, len(faces) * 3

def create_cube_vao():
    verts = np.array([
        -1, 1,-1, -1,-1,-1, 1,-1,-1, 1,-1,-1, 1, 1,-1, -1, 1,-1,
        -1,-1, 1, -1,-1,-1, -1, 1,-1, -1, 1,-1, -1, 1, 1, -1,-1, 1,
         1,-1,-1,  1,-1, 1,  1, 1, 1,  1, 1, 1,  1, 1,-1,  1,-1,-1,
        -1,-1, 1, -1, 1, 1,  1, 1, 1,  1, 1, 1,  1,-1, 1, -1,-1, 1,
        -1, 1,-1,  1, 1,-1,  1, 1, 1,  1, 1, 1, -1, 1, 1, -1, 1,-1,
        -1,-1,-1, -1,-1, 1,  1,-1,-1,  1,-1,-1, -1,-1, 1,  1,-1, 1,
    ], dtype=np.float32)
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    glBindVertexArray(0)
    return vao

def create_quad_vao():
    data = np.array([
        -1,-1,0, 0,0,  1,-1,0, 1,0,  1,1,0, 1,1,
        -1,-1,0, 0,0,  1,1,0, 1,1,  -1,1,0, 0,1,
    ], dtype=np.float32)
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(12))
    glEnableVertexAttribArray(1)
    glBindVertexArray(0)
    return vao

def create_sphere_vao(radius=0.015, slices=16, stacks=16):
    verts = []
    for i in range(stacks + 1):
        phi = np.pi * i / stacks
        for j in range(slices + 1):
            theta = 2.0 * np.pi * j / slices
            verts.append([
                radius * np.sin(phi) * np.cos(theta),
                radius * np.cos(phi),
                radius * np.sin(phi) * np.sin(theta),
            ])
    verts = np.array(verts, dtype=np.float32)
    indices = []
    for i in range(stacks):
        for j in range(slices):
            a = i * (slices + 1) + j
            b = a + slices + 1
            indices.extend([a, b, a+1, b, b+1, a+1])
    indices = np.array(indices, dtype=np.uint32)
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    ebo = glGenBuffers(1)
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    glBindVertexArray(0)
    return vao, len(indices)

# ============ IBL 预计算 ============

capture_projection = glm.perspective(glm.radians(90.0), 1.0, 0.1, 10.0)
capture_views = [
    glm.lookAt(glm.vec3(0), glm.vec3( 1, 0, 0), glm.vec3(0,-1, 0)),
    glm.lookAt(glm.vec3(0), glm.vec3(-1, 0, 0), glm.vec3(0,-1, 0)),
    glm.lookAt(glm.vec3(0), glm.vec3( 0, 1, 0), glm.vec3(0, 0, 1)),
    glm.lookAt(glm.vec3(0), glm.vec3( 0,-1, 0), glm.vec3(0, 0,-1)),
    glm.lookAt(glm.vec3(0), glm.vec3( 0, 0, 1), glm.vec3(0,-1, 0)),
    glm.lookAt(glm.vec3(0), glm.vec3( 0, 0,-1), glm.vec3(0,-1, 0)),
]

def render_cubemap(shader, cube_vao, fbo, cubemap, size, extra_setup=None):
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    glViewport(0, 0, size, size)
    glUseProgram(shader)
    glUniformMatrix4fv(glGetUniformLocation(shader, "projection"), 1, GL_FALSE,
                       glm.value_ptr(capture_projection))
    if extra_setup:
        extra_setup()
    for i in range(6):
        glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_FALSE,
                           glm.value_ptr(capture_views[i]))
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, cubemap, 0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glBindVertexArray(cube_vao)
        glDrawArrays(GL_TRIANGLES, 0, 36)
    glBindFramebuffer(GL_FRAMEBUFFER, 0)


def generate_ibl(cube_vao, quad_vao):
    # 1. 生成程序化环境 cubemap
    env_size = 512
    env_cubemap = glGenTextures(1)
    glBindTexture(GL_TEXTURE_CUBE_MAP, env_cubemap)
    for i in range(6):
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB16F,
                     env_size, env_size, 0, GL_RGB, GL_FLOAT, None)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    fbo = glGenFramebuffers(1)
    rbo = glGenRenderbuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    glBindRenderbuffer(GL_RENDERBUFFER, rbo)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, env_size, env_size)
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo)

    env_shader = compile_shader(EQUIRECT_TO_CUBE_VS, EQUIRECT_TO_CUBE_FS)
    render_cubemap(env_shader, cube_vao, fbo, env_cubemap, env_size)
    glBindTexture(GL_TEXTURE_CUBE_MAP, env_cubemap)
    glGenerateMipmap(GL_TEXTURE_CUBE_MAP)

    print("  环境 cubemap 生成完成")

    # 2. 辐照度卷积（漫反射 IBL）
    irr_size = 32
    irr_cubemap = glGenTextures(1)
    glBindTexture(GL_TEXTURE_CUBE_MAP, irr_cubemap)
    for i in range(6):
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB16F,
                     irr_size, irr_size, 0, GL_RGB, GL_FLOAT, None)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    glBindRenderbuffer(GL_RENDERBUFFER, rbo)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, irr_size, irr_size)

    irr_shader = compile_shader(EQUIRECT_TO_CUBE_VS, IRRADIANCE_FS)
    def bind_env():
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_CUBE_MAP, env_cubemap)
        glUniform1i(glGetUniformLocation(irr_shader, "environmentMap"), 0)
    render_cubemap(irr_shader, cube_vao, fbo, irr_cubemap, irr_size, bind_env)

    print("  辐照度贴图生成完成")

    # 3. 预过滤环境贴图（镜面反射 IBL）
    pref_size = 128
    pref_cubemap = glGenTextures(1)
    glBindTexture(GL_TEXTURE_CUBE_MAP, pref_cubemap)
    for i in range(6):
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB16F,
                     pref_size, pref_size, 0, GL_RGB, GL_FLOAT, None)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glGenerateMipmap(GL_TEXTURE_CUBE_MAP)

    pref_shader = compile_shader(EQUIRECT_TO_CUBE_VS, PREFILTER_FS)
    glUseProgram(pref_shader)
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_CUBE_MAP, env_cubemap)
    glUniform1i(glGetUniformLocation(pref_shader, "environmentMap"), 0)

    max_mip = 5
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    for mip in range(max_mip):
        mip_size = int(pref_size * (0.5 ** mip))
        glBindRenderbuffer(GL_RENDERBUFFER, rbo)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, mip_size, mip_size)
        glViewport(0, 0, mip_size, mip_size)

        r = mip / (max_mip - 1)
        glUniform1f(glGetUniformLocation(pref_shader, "roughness"), r)
        glUniformMatrix4fv(glGetUniformLocation(pref_shader, "projection"), 1, GL_FALSE,
                           glm.value_ptr(capture_projection))

        for i in range(6):
            glUniformMatrix4fv(glGetUniformLocation(pref_shader, "view"), 1, GL_FALSE,
                               glm.value_ptr(capture_views[i]))
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                   GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, pref_cubemap, mip)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glBindVertexArray(cube_vao)
            glDrawArrays(GL_TRIANGLES, 0, 36)

    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    print("  预过滤环境贴图生成完成")

    # 4. BRDF LUT
    brdf_size = 512
    brdf_tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, brdf_tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RG16F, brdf_size, brdf_size, 0, GL_RG, GL_FLOAT, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    brdf_shader = compile_shader(BRDF_VS, BRDF_FS)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    glBindRenderbuffer(GL_RENDERBUFFER, rbo)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, brdf_size, brdf_size)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, brdf_tex, 0)
    glViewport(0, 0, brdf_size, brdf_size)
    glUseProgram(brdf_shader)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glBindVertexArray(quad_vao)
    glDrawArrays(GL_TRIANGLES, 0, 6)
    glBindFramebuffer(GL_FRAMEBUFFER, 0)

    print("  BRDF LUT 生成完成")

    glDeleteFramebuffers(1, [fbo])
    glDeleteRenderbuffers(1, [rbo])

    return env_cubemap, irr_cubemap, pref_cubemap, brdf_tex


# ============ 主程序 ============

def main():
    if not glfw.init():
        sys.exit(1)

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    w, h = int(cam_res[0]), int(cam_res[1])
    window = glfw.create_window(w, h, "Chair Viewer (PBR + IBL)", None, None)
    if not window:
        glfw.terminate()
        sys.exit(1)

    glfw.make_context_current(window)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_pos_callback)
    glfw.set_scroll_callback(window, scroll_callback)
    glfw.set_key_callback(window, key_callback)

    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)
    glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS)
    glClearColor(0.15, 0.15, 0.15, 1.0)

    cube_vao = create_cube_vao()
    quad_vao = create_quad_vao()

    print("正在预计算 IBL 贴图...")
    env_cubemap, irr_cubemap, pref_cubemap, brdf_tex = generate_ibl(cube_vao, quad_vao)
    print("IBL 预计算完成！\n")

    pbr_shader = compile_shader(PBR_VS, PBR_FS)
    skybox_shader = compile_shader(SKYBOX_VS, SKYBOX_FS)
    light_shader = compile_shader(LIGHT_VS, LIGHT_FS)

    gpu_meshes = []
    for m in mesh_data:
        vao, count = create_mesh_vao(m["vertices"], m["normals"], m["uvs"], m["faces"])
        tex = upload_texture_2d(m["tex_img"])
        gpu_meshes.append({
            "vao": vao, "count": count, "tex": tex,
            "metallic": m["metallic"], "roughness": m["roughness"],
        })

    sphere_vao, sphere_count = create_sphere_vao()

    print(f"相机位置: {cam_pos}, FOV: {cam_fov}")
    print(f"光源数量: {len(lights_data)}")
    for i, l in enumerate(lights_data):
        print(f"  光源{i+1}: 位置={l['position']}, 颜色={l['color']}, 强度={l['intensity']}")
    print("操作: 左键旋转, 右键/滚轮缩放, Esc退出")

    while not glfw.window_should_close(window):
        glfw.poll_events()
        fb_w, fb_h = glfw.get_framebuffer_size(window)
        glViewport(0, 0, fb_w, fb_h)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        aspect = fb_w / fb_h if fb_h > 0 else 1.0
        projection = glm.perspective(glm.radians(cam_fov), aspect, 0.01, 100.0)
        eye = glm.vec3(cam_pos[0], cam_pos[1], cam_pos[2] + zoom_offset)
        target = glm.vec3(center[0], center[1], center[2])
        view = glm.lookAt(eye, target, glm.vec3(0, 1, 0))

        model = glm.mat4(1.0)
        model = glm.rotate(model, glm.radians(rotation_x), glm.vec3(1, 0, 0))
        model = glm.rotate(model, glm.radians(rotation_y), glm.vec3(0, 1, 0))
        normal_mat = glm.mat3(glm.transpose(glm.inverse(model)))

        # --- 绘制椅子（PBR + IBL）---
        glUseProgram(pbr_shader)
        glUniformMatrix4fv(glGetUniformLocation(pbr_shader, "model"), 1, GL_FALSE, glm.value_ptr(model))
        glUniformMatrix4fv(glGetUniformLocation(pbr_shader, "view"), 1, GL_FALSE, glm.value_ptr(view))
        glUniformMatrix4fv(glGetUniformLocation(pbr_shader, "projection"), 1, GL_FALSE, glm.value_ptr(projection))
        glUniformMatrix3fv(glGetUniformLocation(pbr_shader, "normalMatrix"), 1, GL_FALSE, glm.value_ptr(normal_mat))
        glUniform3f(glGetUniformLocation(pbr_shader, "camPos"), eye.x, eye.y, eye.z)
        glUniform1i(glGetUniformLocation(pbr_shader, "numLights"), len(lights_data))

        for i, l in enumerate(lights_data):
            p = l["position"]
            c = l["color"]
            glUniform3f(glGetUniformLocation(pbr_shader, f"lights[{i}].position"), p[0], p[1], p[2])
            glUniform3f(glGetUniformLocation(pbr_shader, f"lights[{i}].color"), c[0], c[1], c[2])
            glUniform1f(glGetUniformLocation(pbr_shader, f"lights[{i}].intensity"), l["intensity"])

        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_CUBE_MAP, irr_cubemap)
        glUniform1i(glGetUniformLocation(pbr_shader, "irradianceMap"), 1)
        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_CUBE_MAP, pref_cubemap)
        glUniform1i(glGetUniformLocation(pbr_shader, "prefilterMap"), 2)
        glActiveTexture(GL_TEXTURE3)
        glBindTexture(GL_TEXTURE_2D, brdf_tex)
        glUniform1i(glGetUniformLocation(pbr_shader, "brdfLUT"), 3)

        for gm in gpu_meshes:
            glUniform1f(glGetUniformLocation(pbr_shader, "metallic"), gm["metallic"])
            glUniform1f(glGetUniformLocation(pbr_shader, "roughness"), gm["roughness"])
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, gm["tex"])
            glUniform1i(glGetUniformLocation(pbr_shader, "albedoMap"), 0)
            glBindVertexArray(gm["vao"])
            glDrawElements(GL_TRIANGLES, gm["count"], GL_UNSIGNED_INT, None)

        # --- 绘制天空盒 ---
        glDepthFunc(GL_LEQUAL)
        glUseProgram(skybox_shader)
        glUniformMatrix4fv(glGetUniformLocation(skybox_shader, "projection"), 1, GL_FALSE, glm.value_ptr(projection))
        glUniformMatrix4fv(glGetUniformLocation(skybox_shader, "view"), 1, GL_FALSE, glm.value_ptr(view))
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_CUBE_MAP, env_cubemap)
        glUniform1i(glGetUniformLocation(skybox_shader, "environmentMap"), 0)
        glBindVertexArray(cube_vao)
        glDrawArrays(GL_TRIANGLES, 0, 36)
        glDepthFunc(GL_LESS)

        # --- 绘制光源标记 ---
        glUseProgram(light_shader)
        light_colors = [[1.0, 1.0, 0.0], [1.0, 0.5, 0.0]]
        for i, l in enumerate(lights_data):
            p = l["position"]
            m_light = glm.translate(glm.mat4(1.0), glm.vec3(p[0], p[1], p[2]))
            mvp = projection * view * model * m_light
            glUniformMatrix4fv(glGetUniformLocation(light_shader, "mvp"), 1, GL_FALSE, glm.value_ptr(mvp))
            lc = light_colors[i % len(light_colors)]
            glUniform3f(glGetUniformLocation(light_shader, "lightColor"), lc[0], lc[1], lc[2])
            glBindVertexArray(sphere_vao)
            glDrawElements(GL_TRIANGLES, sphere_count, GL_UNSIGNED_INT, None)

        glfw.swap_buffers(window)

    glfw.terminate()


if __name__ == "__main__":
    main()
