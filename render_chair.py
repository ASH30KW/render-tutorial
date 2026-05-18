import trimesh
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import sys

scene = trimesh.load("/home/ai/Desktop/chair.glb")

if isinstance(scene, trimesh.Scene):
    mesh = trimesh.util.concatenate(scene.dump())
else:
    mesh = scene

vertices = np.array(mesh.vertices, dtype=np.float32)
faces = np.array(mesh.faces, dtype=np.uint32)
normals = np.array(mesh.vertex_normals, dtype=np.float32)

if mesh.visual and hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
    colors = np.array(mesh.visual.vertex_colors[:, :3], dtype=np.float32) / 255.0
else:
    colors = None

bounds = mesh.bounds
center = (bounds[0] + bounds[1]) / 2.0
scale = np.max(bounds[1] - bounds[0])

light_pos = [0.5, 1.0, 0.5]

rotation_x = 20.0
rotation_y = 0.0
zoom = 2.0
last_mouse = [0, 0]
mouse_button = None

def init():
    glClearColor(0.15, 0.15, 0.15, 1.0)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

    glLightfv(GL_LIGHT0, GL_POSITION, [light_pos[0], light_pos[1], light_pos[2], 1.0])
    glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
    glLightfv(GL_LIGHT0, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
    glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
    glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION, 1.0)
    glLightf(GL_LIGHT0, GL_LINEAR_ATTENUATION, 0.05)
    glLightf(GL_LIGHT0, GL_QUADRATIC_ATTENUATION, 0.01)


    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_NORMAL_ARRAY)
    glVertexPointer(3, GL_FLOAT, 0, vertices)
    glNormalPointer(GL_FLOAT, 0, normals)
    if colors is not None:
        glEnableClientState(GL_COLOR_ARRAY)
        glColorPointer(3, GL_FLOAT, 0, colors)

def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    gluLookAt(0, 0, zoom, 0, 0, 0, 0, 1, 0)
    glRotatef(rotation_x, 1, 0, 0)
    glRotatef(rotation_y, 0, 1, 0)
    glScalef(1.0 / scale * 2, 1.0 / scale * 2, 1.0 / scale * 2)
    glTranslatef(-center[0], -center[1], -center[2])

    if colors is None:
        glColor3f(0.7, 0.55, 0.35)

    glDrawElements(GL_TRIANGLES, len(faces) * 3, GL_UNSIGNED_INT, faces)

    glDisable(GL_LIGHTING)
    glColor3f(1.0, 1.0, 0.0)
    glPushMatrix()
    glTranslatef(light_pos[0], light_pos[1], light_pos[2])
    glutSolidSphere(0.03, 16, 16)
    glPopMatrix()
    glEnable(GL_LIGHTING)

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
