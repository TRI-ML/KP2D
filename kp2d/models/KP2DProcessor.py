from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt
from features.matchProcessing import MatchProcessor
from features.homography import homographyFilter, visualizeHomography

from kp2d.networks.keypoint_net import KeypointNet
import torch

class KP2DProcessor:

    def __init__(self, model_path, device = "cpu"):

        self._detector = self._load_model_from_path(model_path)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self._matchProc = MatchProcessor()

        self.device = device

    def processBatch(self, imageBatch, startImageIdx, matchCutoff=0.5):
        n = len(imageBatch)
        if (n < 2):
            return None
        self._matchProc.newBatch()
        trainImage = cv2.imread(imageBatch[0], cv2.IMREAD_GRAYSCALE)
        train_kp, train_des = self._detect_keypoints(trainImage)


        for i, image in enumerate(imageBatch[1:]):
            img = cv2.imread(image)

            query_kp, query_des = self._detect_keypoints(trainImage)

            matches = self._matcher.match(query_des, train_des)
            # visualizeHomography(matches, query_kp, train_kp, img, trainImage)
            matches = self._filterMatches(matches, matchCutoff, query_kp, train_kp)
            self._matchProc.addMatches(matches, train_kp, query_kp, startImageIdx, startImageIdx + i + 1)

        # Show matches and halt execution
        """
        train_kp_image = cv2.drawKeypoints(trainImage, train_kp, None, color=(0,255,0), flags=0)
        plt.figure()
        plt.imshow(train_kp_image)
        match_img = cv2.drawMatches(img, query_kp, trainImage, train_kp, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.figure()
        plt.imshow(match_img)
        plt.show()
        """
        return self._matchProc.getMatchedKeypoints()

    def _filterMatches(self, matches, bestPercent, keypoints1, keypoints2):
        numBestMatches = int(len(matches) * bestPercent)
        bestMatches = sorted(matches, key=lambda x: x.distance)[:numBestMatches]
        bestMatches = homographyFilter(bestMatches, keypoints1, keypoints2)
        return bestMatches

    def _load_model_from_path(self, model_path):
        checkpoint = torch.load(model_path)
        model_args = checkpoint['config']['model']['params']

        keypoint_net = KeypointNet(use_color=True,
                                   do_upsample=False,
                                   do_cross=True)

        keypoint_net.load_state_dict(checkpoint['state_dict'])
        if self.device == "cuda":
            keypoint_net = keypoint_net.cuda()
        keypoint_net.eval()
        return keypoint_net

    def _img_to_torch(self, img):
        img_ten = torch.as_tensor(img)
        img_ten = img_ten.unsqueeze(0)
        img_ten = img_ten.permute(0, 3, 1, 2)
        return (img_ten.float / 255.).to(self.device)

    def _top_k_keypoints(self, scores, keypoints, features , k = 300):
        _, top_k = scores.view(1, -1).topk(k, dim=1)
        k_kps = keypoints.view(1, 2, -1)[:, :, top_k[0].squeeze()]
        k_feat = features.view(1, 2, -1)[:, :, top_k[0].squeeze()]
        return k_kps, k_feat

    def _detect_keypoints(self, img): #TODO: make use of batch
        img_torch = self._img_to_torch(img)
        source_score, source_uv_pred, source_feat = self._detector(img_torch)
        return self._top_k_keypoints(source_score, source_uv_pred, source_feat)