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

# 读取相机
cam = scene.camera
cam_transform = scene.camera_transform
cam_pos = cam_transform[:3, 3].copy()
cam_fov = float(cam.fov[1])
cam_res = cam.resolution

# 读取光源
lights_data = []
for light in scene.lights:
    color = np.array(light.color[:3], dtype=np.float32) / 255.0
    transform, _ = scene.graph.get(light.name)
    pos = transform[:3, 3].copy()
    lights_data.append({"color": color, "intensity": light.intensity, "position": pos})

# 读取网格
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

# ============ GLSL 着色器 ============

VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aUV;

out vec3 FragPos;
out vec3 Normal;
out vec2 UV;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat3 normalMatrix;

void main() {
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = normalize(normalMatrix * aNormal);
    UV = aUV;
    gl_Position = projection * view * vec4(FragPos, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330 core
out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;
in vec2 UV;

uniform sampler2D albedoMap;
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

float DistributionGGX(vec3 N, vec3 H, float r) {
    float a = r * r;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    float denom = NdotH2 * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

float GeometrySchlickGGX(float NdotV, float r) {
    float k = (r + 1.0) * (r + 1.0) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float r) {
    return GeometrySchlickGGX(max(dot(N, V), 0.0), r)
         * GeometrySchlickGGX(max(dot(N, L), 0.0), r);
}

vec3 FresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

void main() {
    vec3 albedo = pow(texture(albedoMap, UV).rgb, vec3(2.2));
    vec3 N = normalize(Normal);
    vec3 V = normalize(camPos - FragPos);

    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    float r = max(roughness, 0.05);

    vec3 Lo = vec3(0.0);
    for (int i = 0; i < numLights; i++) {
        vec3 L = normalize(lights[i].position - FragPos);
        vec3 H = normalize(V + L);
        float dist = length(lights[i].position - FragPos);
        float attenuation = 1.0 / (dist * dist);
        vec3 radiance = lights[i].color * lights[i].intensity * attenuation;

        float NDF = DistributionGGX(N, H, r);
        float G = GeometrySmith(N, V, L, r);
        vec3 F = FresnelSchlick(max(dot(H, V), 0.0), F0);

        vec3 numerator = NDF * G * F;
        float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
        vec3 specular = numerator / denominator;

        vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);

        float NdotL = max(dot(N, L), 0.0);
        Lo += (kD * albedo / PI + specular) * radiance * NdotL;
    }

    vec3 ambient = vec3(0.03) * albedo;
    vec3 color = ambient + Lo;

    // tone mapping + gamma
    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0 / 2.2));

    FragColor = vec4(color, 1.0);
}
"""

LIGHT_VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 aPos;

uniform mat4 mvp;

void main() {
    gl_Position = mvp * vec4(aPos, 1.0);
}
"""

LIGHT_FRAGMENT_SHADER = """
#version 330 core
out vec4 FragColor;
uniform vec3 lightColor;

void main() {
    FragColor = vec4(lightColor, 1.0);
}
"""

# ============ 交互状态 ============

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


# ============ OpenGL 辅助函数 ============

def create_shader_program(vs_src, fs_src):
    vs = shaders.compileShader(vs_src, GL_VERTEX_SHADER)
    fs = shaders.compileShader(fs_src, GL_FRAGMENT_SHADER)
    return shaders.compileProgram(vs, fs)


def upload_texture(img):
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

    stride = 8 * 4  # 3+3+2 floats * 4 bytes
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(24))
    glEnableVertexAttribArray(2)

    glBindVertexArray(0)
    return vao, len(faces) * 3


def create_sphere_vao(radius=0.015, slices=16, stacks=16):
    verts = []
    for i in range(stacks + 1):
        phi = np.pi * i / stacks
        for j in range(slices + 1):
            theta = 2.0 * np.pi * j / slices
            x = radius * np.sin(phi) * np.cos(theta)
            y = radius * np.cos(phi)
            z = radius * np.sin(phi) * np.sin(theta)
            verts.append([x, y, z])
    verts = np.array(verts, dtype=np.float32)

    indices = []
    for i in range(stacks):
        for j in range(slices):
            a = i * (slices + 1) + j
            b = a + slices + 1
            indices.extend([a, b, a + 1, b, b + 1, a + 1])
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


# ============ 主程序 ============

def main():
    if not glfw.init():
        print("GLFW 初始化失败")
        sys.exit(1)

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    w, h = int(cam_res[0]), int(cam_res[1])
    window = glfw.create_window(w, h, "Chair Viewer (PBR)", None, None)
    if not window:
        glfw.terminate()
        print("窗口创建失败")
        sys.exit(1)

    glfw.make_context_current(window)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_pos_callback)
    glfw.set_scroll_callback(window, scroll_callback)
    glfw.set_key_callback(window, key_callback)

    glEnable(GL_DEPTH_TEST)
    glClearColor(0.15, 0.15, 0.15, 1.0)

    # 编译着色器
    pbr_shader = create_shader_program(VERTEX_SHADER, FRAGMENT_SHADER)
    light_shader = create_shader_program(LIGHT_VERTEX_SHADER, LIGHT_FRAGMENT_SHADER)

    # 上传网格数据
    gpu_meshes = []
    for m in mesh_data:
        vao, count = create_mesh_vao(m["vertices"], m["normals"], m["uvs"], m["faces"])
        tex = upload_texture(m["tex_img"])
        gpu_meshes.append({
            "vao": vao, "count": count, "tex": tex,
            "metallic": m["metallic"], "roughness": m["roughness"],
        })

    # 光源球体
    sphere_vao, sphere_count = create_sphere_vao()

    print(f"相机位置: {cam_pos}")
    print(f"相机FOV: {cam_fov}")
    print(f"光源数量: {len(lights_data)}")
    for i, l in enumerate(lights_data):
        print(f"  光源{i+1}: 位置={l['position']}, 颜色={l['color']}, 强度={l['intensity']}")
    print("操作: 左键拖拽旋转, 右键/滚轮缩放, Esc退出")

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

        # 绘制椅子
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

        for gm in gpu_meshes:
            glUniform1f(glGetUniformLocation(pbr_shader, "metallic"), gm["metallic"])
            glUniform1f(glGetUniformLocation(pbr_shader, "roughness"), gm["roughness"])
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, gm["tex"])
            glUniform1i(glGetUniformLocation(pbr_shader, "albedoMap"), 0)
            glBindVertexArray(gm["vao"])
            glDrawElements(GL_TRIANGLES, gm["count"], GL_UNSIGNED_INT, None)

        # 绘制光源标记
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
