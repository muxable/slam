import sys
import time
from typing import List
import cv2
import numpy as np
from matplotlib import pyplot as plt

from display import Display2D, Display3D


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
        kp1, kp2 = [], []

        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                kp2.append(f2.keypoints[m.trainIdx].pt)
                kp1.append(f1.keypoints[m.queryIdx].pt)

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

        return Rt, pts, kp1, kp2


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
        return self.cameras[-1], pts


if __name__ == "__main__":
    slam = SLAM()
    disp2d = Display2D()
    disp3d = Display3D()
    cap = cv2.VideoCapture("video_color.mp4")
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break
        K = np.array(
            [[707.0912, 0, img.shape[1] // 2], [0, 707.0912, img.shape[0] // 2], [0, 0, 1]]
        )
        frame = AnnotatedFrame(img, K)
        # frame.render()
        cam, pts = slam.localize(frame)
        # draw the keypoints on the image
        for keypoint in frame.keypoints:
            cv2.circle(img, keypoint, color=(0,0,0), radius=3)
        # draw the correspondence lines from the previous image
        for pt in pts:
            if len(self.pts[i1].frames) >= 5:
                cv2.circle(img, (u1, v1), color=(0,255,0), radius=3)
            else:
            cv2.circle(img, (u1, v1), color=(0,128,0), radius=3)
            # draw the trail
            pts = []
            lfid = None
            for f, idx in zip(self.pts[i1].frames[-9:][::-1], self.pts[i1].idxs[-9:][::-1]):
            if lfid is not None and lfid-1 != f.id:
                break
            pts.append(tuple(map(lambda x: int(round(x)), f.kpus[idx])))
            lfid = f.id
            if len(pts) >= 2:
                cv2.polylines(img, np.array([pts], dtype=np.int32), False, myjet[len(pts)]*255, thickness=1, lineType=16)
        disp2d.paint(frame)
        disp3d.paint(slam.cameras, slam.points)
        print("localized points", len(pts))
        cv2.waitKey(0)

    # time.sleep(10000)
    # slam = SLAM()
    # source = cv2.VideoCapture(sys.argv[1])
    # while source.isOpened():
    #     ret, frame = source.read()
    #     if not ret:
    #         break
    #     slam.localize(AnnotatedFrame(frame))
