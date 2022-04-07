import torch
from kp2d.networks.keypoint_net import KeypointNet
import cv2

import kp2d.datasets.augmentations as aug
from kp2d.utils.keypoints import draw_keypoints
import glob
import kp2d.utils.tt_image as tt

model_path = r"D:\PycharmProjects\KP2D\data\models\kp2d/model.ckpt"
picture_path = r"D:\MachineLearning\SonarData\SonarDataSets\Real\FoldingChair3m\frame0029.jpg"

#load model
checkpoint = torch.load(model_path)
model_args = checkpoint['config']['model']['params']

keypoint_net = KeypointNet(use_color=True,
                           do_upsample=False,
                           do_cross=True)

keypoint_net.load_state_dict(checkpoint['state_dict'])
keypoint_net = keypoint_net.cuda()
keypoint_net.eval()



eval_params = [{'res': (320, 240), 'top_k': 300, }]
eval_params += [{'res': (640, 480), 'top_k': 1000, }]

img = cv2.imread(picture_path, cv2.IMREAD_COLOR)

data = aug.ha_augment_sample({'image': tt.img_to_torch(img).permute( 2, 0, 1)})

source_score, source_uv_pred, source_feat = keypoint_net(data['image'].unsqueeze(0).to('cuda'))
source_score_2, source_uv_pred_2, source_feat_2 = keypoint_net(data['image_aug'].unsqueeze(0).to('cuda'))
_, top_k = source_score.view(1, -1).topk(300, dim=1)
_, top_k_2 = source_score_2.view(1, -1).topk(300, dim=1)

vis_ori = draw_keypoints(data['image'].permute( 1, 2, 0).numpy(), source_uv_pred.view(1, 2, -1)[:, :, top_k[0].squeeze()], (255, 0, 255))
vis_ori_2 = draw_keypoints(data['image_aug'].permute( 1, 2, 0).numpy(), source_uv_pred_2.view(1, 2, -1)[:, :, top_k_2[0].squeeze()], (255, 0, 255))

# show picture
cv2.imshow("img", vis_ori)
cv2.imshow("img_2", vis_ori_2)
cv2.waitKey(0)