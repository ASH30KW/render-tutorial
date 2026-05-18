import trimesh
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from PIL import Image
import sys

scene = trimesh.load("/home/ai/Desktop/chair.glb")

meshes = []
for name, geom in scene.geometry.items():
    verts = np.array(geom.vertices, dtype=np.float32)
    norms = np.array(geom.vertex_normals, dtype=np.float32)
    faces = np.array(geom.faces, dtype=np.uint32)
    uvs = np.array(geom.visual.uv, dtype=np.float32)
    tex_img = geom.visual.material.baseColorTexture
    meshes.append({
        "name": name,
        "vertices": verts,
        "normals": norms,
        "faces": faces,
        "uvs": uvs,
        "tex_img": tex_img,
        "tex_id": None,
    })

combined = trimesh.util.concatenate(scene.dump())
bounds = combined.bounds
center = (bounds[0] + bounds[1]) / 2.0
scale = np.max(bounds[1] - bounds[0])

light_pos = [0.5, 1.0, 0.5]

rotation_x = 20.0
rotation_y = 0.0
zoom = 2.0
last_mouse = [0, 0]
mouse_button = None


def upload_texture(img):
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    img_data = img.convert("RGBA").tobytes()
    tex_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.width, img.height, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, img_data)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glGenerateMipmap(GL_TEXTURE_2D)
    return tex_id


def init():
    glClearColor(0.15, 0.15, 0.15, 1.0)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_TEXTURE_2D)

    glLightfv(GL_LIGHT0, GL_POSITION, [light_pos[0], light_pos[1], light_pos[2], 1.0])
    glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
    glLightfv(GL_LIGHT0, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
    glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
    glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION, 1.0)
    glLightf(GL_LIGHT0, GL_LINEAR_ATTENUATION, 0.05)
    glLightf(GL_LIGHT0, GL_QUADRATIC_ATTENUATION, 0.01)

    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [0.3, 0.3, 0.3, 1.0])
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 30.0)

    for m in meshes:
        m["tex_id"] = upload_texture(m["tex_img"])


def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    gluLookAt(0, 0, zoom, 0, 0, 0, 0, 1, 0)
    glRotatef(rotation_x, 1, 0, 0)
    glRotatef(rotation_y, 0, 1, 0)
    glScalef(1.0 / scale * 2, 1.0 / scale * 2, 1.0 / scale * 2)
    glTranslatef(-center[0], -center[1], -center[2])

    glEnable(GL_TEXTURE_2D)
    glEnable(GL_LIGHTING)

    for m in meshes:
        glBindTexture(GL_TEXTURE_2D, m["tex_id"])

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        glEnableClientState(GL_TEXTURE_COORD_ARRAY)

        glVertexPointer(3, GL_FLOAT, 0, m["vertices"])
        glNormalPointer(GL_FLOAT, 0, m["normals"])
        glTexCoordPointer(2, GL_FLOAT, 0, m["uvs"])

        glDrawElements(GL_TRIANGLES, len(m["faces"]) * 3, GL_UNSIGNED_INT, m["faces"])

        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)
        glDisableClientState(GL_TEXTURE_COORD_ARRAY)

    glDisable(GL_TEXTURE_2D)
    glDisable(GL_LIGHTING)
    glColor3f(1.0, 1.0, 0.0)
    glPushMatrix()
    glTranslatef(light_pos[0], light_pos[1], light_pos[2])
    glutSolidSphere(0.03, 16, 16)
    glPopMatrix()

    glutSwapBuffers()


def reshape(w, h):
    if h == 0:
        h = 1
    glViewport(0, 0, w, h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, w / h, 0.01, 100.0)
    glMatrixMode(GL_MODELVIEW)


def mouse(button, state, x, y):
    global mouse_button, last_mouse
    if state == GLUT_DOWN:
        mouse_button = button
        last_mouse = [x, y]
    else:
        mouse_button = None


def motion(x, y):
    global rotation_x, rotation_y, zoom, last_mouse
    dx = x - last_mouse[0]
    dy = y - last_mouse[1]
    if mouse_button == GLUT_LEFT_BUTTON:
        rotation_y += dx * 0.5
        rotation_x += dy * 0.5
    elif mouse_button == GLUT_RIGHT_BUTTON:
        zoom += dy * 0.01
        zoom = max(0.5, min(10.0, zoom))
    last_mouse = [x, y]
    glutPostRedisplay()


def keyboard(key, x, y):
    if key == b'\x1b':
        sys.exit(0)


glutInit(sys.argv)
glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
glutInitWindowSize(1024, 768)
glutCreateWindow(b"Chair Viewer")
init()
glutDisplayFunc(display)
glutReshapeFunc(reshape)
glutMouseFunc(mouse)
glutMotionFunc(motion)
glutKeyboardFunc(keyboard)
print("Controls: Left-drag to rotate, Right-drag to zoom, Esc to quit")
glutMainLoop()
