from multiprocessing import Process, Queue
from typing import List
import pypangolin as pangolin
import OpenGL.GL as gl
import numpy as np
import pygame
from pygame.locals import DOUBLEBUF


class Display2D(object):
    def __init__(self, W, H):
        pygame.init()
        self.screen = pygame.display.set_mode((W, H), DOUBLEBUF)
        self.surface = pygame.Surface(self.screen.get_size()).convert()

    def paint(self, img):
        for event in pygame.event.get():
            pass

        pygame.surfarray.blit_array(self.surface, img.swapaxes(0, 1)[:, :, [0, 1, 2]])
        self.screen.blit(self.surface, (0, 0))
        pygame.display.flip()


def draw_camera(pose, w=1.0, h_ratio=0.75, z_ratio=0.6):
    h = w * h_ratio
    z = w * z_ratio

    gl.glPushMatrix()
    gl.glMultTransposeMatrixd(pose)

    gl.glBegin(gl.GL_LINES)
    gl.glVertex3f(0, 0, 0)
    gl.glVertex3f(w, h, z)
    gl.glVertex3f(0, 0, 0)
    gl.glVertex3f(w, -h, z)
    gl.glVertex3f(0, 0, 0)
    gl.glVertex3f(-w, -h, z)
    gl.glVertex3f(0, 0, 0)
    gl.glVertex3f(-w, h, z)

    gl.glVertex3f(w, h, z)
    gl.glVertex3f(w, -h, z)

    gl.glVertex3f(-w, h, z)
    gl.glVertex3f(-w, -h, z)

    gl.glVertex3f(-w, h, z)
    gl.glVertex3f(w, h, z)

    gl.glVertex3f(-w, -h, z)
    gl.glVertex3f(w, -h, z)
    gl.glEnd()

    gl.glPopMatrix()


class Display3D(object):
    def __init__(self):
        self.state = None
        self.q = Queue()
        self.vp = Process(target=self.viewer_thread, args=(self.q,))
        self.vp.daemon = True
        self.vp.start()

    def viewer_thread(self, q):
        self.viewer_init(1024, 768)
        while 1:
            self.viewer_refresh(q)

    def viewer_init(self, w, h):
        pangolin.CreateWindowAndBind("Map Viewer", w, h)
        gl.glEnable(gl.GL_DEPTH_TEST)

        self.scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(w, h, 420, 420, w // 2, h // 2, 0.2, 10000),
            pangolin.ModelViewLookAt(0, -10, -8, 0, 0, 0, 0, -1, 0),
        )
        self.handler = pangolin.Handler3D(self.scam)

        # Create Interactive View in window
        self.dcam = pangolin.CreateDisplay()
        self.dcam.SetBounds(
            pangolin.Attach(0),
            pangolin.Attach(1),
            pangolin.Attach(0),
            pangolin.Attach(1),
            w / h,
        )
        self.dcam.SetHandler(self.handler)
        # hack to avoid small Pangolin, no idea why it's *2
        self.dcam.Resize(pangolin.Viewport(0, 0, w * 2, h * 2))
        self.dcam.Activate()

    def viewer_refresh(self, q):
        while not q.empty():
            self.state = q.get()

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        self.dcam.Activate(self.scam)

        if self.state is not None:
            if len(self.state[0]) > 1:
                # draw poses
                gl.glColor3f(0.0, 1.0, 0.0)
                for pose in self.state[0][:-1]:
                    draw_camera(pose)

            if len(self.state[0]) > 0:
                # draw current pose as yellow
                gl.glColor3f(1.0, 1.0, 0.0)
                draw_camera(self.state[0][-1])

            if self.state[1] is not None and len(self.state[1]) > 0:
                # draw keypoints
                gl.glPointSize(5)
                gl.glColor3f(1.0, 0.0, 0.0)
                pangolin.DrawPoints(self.state[1])

        pangolin.FinishFrame()

    def paint(self, poses, pts):
        if self.q is None:
            return

        self.q.put((poses, pts))
