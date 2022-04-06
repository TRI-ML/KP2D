# This is Tomtoms image library because he is sick of all the confusing indexing stuff
import torch

def img_to_torch(img):
    t_img = torch.as_tensor(img)
    return t_img.float()/255.