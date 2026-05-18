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

mesh_data = []
for name, geom in scene.geometry.items():
    mat = geom.visual.material
    mesh_data.append({
        "vertices": np.array(geom.vertices, dtype=np.float32),
        "normals": np.array(geom.vertex_normals, dtype=np.float32),
        "uvs": np.array(geom.visual.uv, dtype=np.float32),
        "faces": np.array(geom.faces, dtype=np.uint32),
        "tex_img": mat.baseColorTexture,
    })

combined = trimesh.util.concatenate(scene.dump())
bounds = combined.bounds
center = (bounds[0] + bounds[1]) / 2.0

# ============ GLSL 着色器 ============

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

void main(){
    vec3 color = texture(albedoMap, UV).rgb;
    FragColor = vec4(color, 1.0);
}
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

# ============ 主程序 ============

def main():
    if not glfw.init():
        sys.exit(1)

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    w, h = int(cam_res[0]), int(cam_res[1])
    window = glfw.create_window(w, h, "Chair Viewer (PBR)", None, None)
    if not window:
        glfw.terminate()
        sys.exit(1)

    glfw.make_context_current(window)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_pos_callback)
    glfw.set_scroll_callback(window, scroll_callback)
    glfw.set_key_callback(window, key_callback)

    glEnable(GL_DEPTH_TEST)
    glClearColor(0.15, 0.15, 0.15, 1.0)

    pbr_shader = compile_shader(PBR_VS, PBR_FS)

    gpu_meshes = []
    for m in mesh_data:
        vao, count = create_mesh_vao(m["vertices"], m["normals"], m["uvs"], m["faces"])
        tex = upload_texture_2d(m["tex_img"])
        gpu_meshes.append({"vao": vao, "count": count, "tex": tex})

    print(f"相机位置: {cam_pos}, FOV: {cam_fov}")
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

        glUseProgram(pbr_shader)
        glUniformMatrix4fv(glGetUniformLocation(pbr_shader, "model"), 1, GL_FALSE, glm.value_ptr(model))
        glUniformMatrix4fv(glGetUniformLocation(pbr_shader, "view"), 1, GL_FALSE, glm.value_ptr(view))
        glUniformMatrix4fv(glGetUniformLocation(pbr_shader, "projection"), 1, GL_FALSE, glm.value_ptr(projection))
        glUniformMatrix3fv(glGetUniformLocation(pbr_shader, "normalMatrix"), 1, GL_FALSE, glm.value_ptr(normal_mat))

        for gm in gpu_meshes:
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, gm["tex"])
            glUniform1i(glGetUniformLocation(pbr_shader, "albedoMap"), 0)
            glBindVertexArray(gm["vao"])
            glDrawElements(GL_TRIANGLES, gm["count"], GL_UNSIGNED_INT, None)

        glfw.swap_buffers(window)

    glfw.terminate()


if __name__ == "__main__":
    main()
