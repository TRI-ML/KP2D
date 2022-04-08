from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
import numpy as np
import cv2
import os

imgs = []
data_path = r"D:\MachineLearning\SonarData\SonarDataSets\Real"
rosbag_path = r'C:\Users\Dr. Paul von Immel\Downloads\rosbag2_2022_04_06-11-41_16-20220408T162332Z-001\rosbag2_2022_04_06-11-41_16'


folderName = os.path.basename(rosbag_path)
data_path_extracted = os.path.join(data_path, folderName)





def rosbag_to_jpg(rosbag_path, data_path_extracted):
    if not os.path.exists(data_path_extracted):
        os.mkdir(data_path_extracted)
    i = 0
    with Reader(rosbag_path) as reader:
        for connection, timestamp, rawdata in reader.messages(connections=reader.connections.values()):
            msg = deserialize_cdr(rawdata, connection.msgtype)
            img_reshaped = np.reshape(msg.data, [msg.height,msg.width])
            filename = "frame{:>05}.jpg".format(i)
            cv2.imwrite(os.path.join(data_path_extracted, filename), img_reshaped)
            i += 1


def load_all_imgs(data_path_extracted):
    images = []
    for filename in os.listdir(data_path_extracted):
        img = cv2.imread(os.path.join(data_path_extracted, filename))
        if img is not None:

            images.append(img)
    return np.asarray(images)

imgs = load_all_imgs(r"D:\MachineLearning\SonarData\SonarDataSets\Real\rosbag2_2022_04_06-11-27_20")

mean = np.mean(imgs,axis=0)
for i in imgs:
    cv2.imshow("minus mean", (i - mean*2)/255.)
    cv2.imshow("i", i/255.)
    cv2.imshow("mean", mean*4/255.)
    cv2.waitKey(0)