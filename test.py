import torch
from kp2d.networks.keypoint_net import KeypointNet
import cv2
from kp2d.utils.keypoints import draw_keypoints
import glob
model_path = r"D:\PycharmProjects\KP2D\data\models\kp2d/model.ckpt"
model_path_2 = r"D:\PycharmProjects\KP2D\data\models\kp2d/v4.ckpt"
picture_path = r"D:\MachineLearning\SonarData\SonarDataSets\Real\FoldingChair3m"

#load model
checkpoint = torch.load(model_path)
model_args = checkpoint['config']['model']['params']

keypoint_net = KeypointNet(use_color=True,
                           do_upsample=False,
                           do_cross=True)

keypoint_net.load_state_dict(checkpoint['state_dict'])
keypoint_net = keypoint_net.cuda()
keypoint_net.eval()

checkpoint_2 = torch.load(model_path_2)
model_args_2 = checkpoint['config']['model']['params']

keypoint_net_2 = KeypointNet(use_color=True,
                           do_upsample=False,
                           do_cross=True)

keypoint_net_2.load_state_dict(checkpoint_2['state_dict'])
keypoint_net_2 = keypoint_net_2.cuda()
keypoint_net_2.eval()


eval_params = [{'res': (320, 240), 'top_k': 300, }]
eval_params += [{'res': (640, 480), 'top_k': 1000, }]

for filename in glob.glob(picture_path + '/*.jpg'):
#load picture

    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    img_ten = torch.as_tensor(img)
    img_ten = img_ten.unsqueeze(0)
    img_ten =img_ten.permute(0,3,1,2)



    #compute points

    source_score, source_uv_pred, source_feat = keypoint_net((img_ten.float()/250.).to('cuda'))
    source_score_2, source_uv_pred_2, source_feat_2 = keypoint_net_2((img_ten.float()/250.).to('cuda'))
    _, top_k = source_score.view(1,-1).topk(300, dim=1)
    _, top_k_2 = source_score_2.view(1,-1).topk(300, dim=1)

    #plot points on picture
    vis_ori = draw_keypoints(img, source_uv_pred.view(1,2,-1)[:,:,top_k[0].squeeze()],(255,0,255))
    vis_ori_2 = draw_keypoints(img, source_uv_pred_2.view(1,2,-1)[:,:,top_k_2[0].squeeze()],(255,0,255))
    #show picture
    cv2.imshow("img", vis_ori)
    cv2.imshow("img_2", vis_ori_2)
    cv2.waitKey(0)
cv2.destroyAllWindows()

# def _normalize_uv_coordinates(uv_pred, H, W):
#     uv_norm = uv_pred.clone()
#     uv_norm[:, 0] = (uv_norm[:, 0] / (float(W - 1) / 2.)) - 1.
#     uv_norm[:, 1] = (uv_norm[:, 1] / (float(H - 1) / 2.)) - 1.
#     uv_norm = uv_norm.permute(0, 2, 3, 1)
#     return uv_norm
#
# def _denormalize_uv_coordinates(uv_norm, H, W):
#     uv_pred = uv_norm.clone()
#     uv_pred[:, :, :, 0] = (uv_pred[:, :, :, 0] + 1) * (float(W - 1) / 2.)
#     uv_pred[:, :, :, 1] = (uv_pred[:, :, :, 1] + 1) * (float(H - 1) / 2.)
#     uv_pred = uv_pred.permute(0, 3, 1, 2)
#     return uv_pred