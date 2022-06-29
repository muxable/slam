import sys
import time
from typing import List
import cv2
import numpy as np
from matplotlib import pyplot as plt

from display import Display3D


class AnnotatedFrame:
    def __init__(self, frame, K):
        self.frame = frame
        self.K = K

        orb = cv2.ORB_create(nfeatures=3000)
        self.keypoints, self.descriptors = orb.detectAndCompute(frame, None)

    def render(self):
        # draw only keypoints location, not size and orientation
        img2 = cv2.drawKeypoints(
            self.frame, self.keypoints, None, color=(0, 255, 0), flags=0
        )
        plt.imshow(img2)
        plt.show()

    @staticmethod
    def reconstruct(f1: "AnnotatedFrame", f2: "AnnotatedFrame"):
        """
        Computes the 3D transform between two images.

        See https://docs.opencv.org/4.x/da/de9/tutorial_py_epipolar_geometry.html for more details.
        """
        assert (f1.K == f2.K).all(), "TODO: reconstruction only valid for same camera"

        # Find and match interest points.
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(f1.descriptors, f2.descriptors, k=2)

        # Perform Lowe's ratio test to filter out low-signal points.
        pts1, pts2 = [], []

        for i, (m, n) in enumerate(matches):
            if m.distance < 0.75 * n.distance:
                pts2.append(f2.keypoints[m.trainIdx].pt)
                pts1.append(f1.keypoints[m.queryIdx].pt)

        pts1 = np.float64(pts1)
        pts2 = np.float64(pts2)

        K = f1.K

        E, mask = cv2.findEssentialMat(pts1, pts2, K)
        pts1 = pts1[mask.ravel() == 1]
        pts2 = pts2[mask.ravel() == 1]

        _, R, t, _ = cv2.recoverPose(E, pts1, pts2)

        print(R, t)

        P_l = f1.K @ np.hstack((np.eye(3, 3), np.zeros((3, 1))))
        P_r = f2.K @ np.hstack((R, t))

        pts = cv2.triangulatePoints(P_l, P_r, pts1.T, pts2.T)
        pts /= pts[3]  # convert to euclidean

        Rt = np.eye(4)
        Rt[:3, :3] = R
        Rt[:3, 3] = t.ravel()

        return Rt, pts


class SLAM:
    def __init__(self):
        self.frames = []
        self.cameras = []
        self.points = None

    def localize(self, f2: List[AnnotatedFrame]):
        if len(self.frames) == 0:
            self.frames = [f2]
            self.cameras = [np.eye(4)]
            return np.eye(4)

        Rt, pts = AnnotatedFrame.reconstruct(self.frames[-1], f2)

        self.frames.append(f2)
        self.cameras.append(Rt @ self.cameras[-1])
        if self.points is None:
            self.points = self.cameras[-1] @ pts
        else:
            self.points = np.hstack((self.points, self.cameras[-1] @ pts))
        return self.cameras[-1]


if __name__ == "__main__":
    slam = SLAM()
    display = Display3D()
    cap = cv2.VideoCapture("production.mp4")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        K = np.array(
            [[525, 0, frame.shape[1] // 2], [0, 525, frame.shape[0] // 2], [0, 0, 1]]
        )
        frame = AnnotatedFrame(frame, K)
        # frame.render()
        p = slam.localize(frame)
        display.paint(slam.cameras, slam.points)
        print(p)
        cv2.waitKey(0)

    # time.sleep(10000)
    # slam = SLAM()
    # source = cv2.VideoCapture(sys.argv[1])
    # while source.isOpened():
    #     ret, frame = source.read()
    #     if not ret:
    #         break
    #     slam.localize(AnnotatedFrame(frame))
