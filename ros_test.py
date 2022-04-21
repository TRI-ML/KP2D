from rosbags.rosbag2 import Reader
from rosbags.rosbag1 import Reader as Reader1
from rosbags.serde import deserialize_cdr
import numpy as np
import cv2
import os
from rosbags.serde import deserialize_cdr, ros1_to_cdr
imgs = []
data_path = r"D:\MachineLearning\SonarData\SonarDataSets\Real"
rosbag_path = r'C:\Users\Dr. Paul von Immel\Downloads\rosbag2_2022_04_06-11-27_20-20220408T125835Z-001\rosbag2_2022_04_06-11-27_20'

folderName = os.path.basename(rosbag_path)
data_path_extracted = os.path.join(data_path, folderName)


def get_xyz():
    with Reader1(r"C:\Users\Dr. Paul von Immel\Downloads\2022-04-06-11-25-30.bag") as reader:
        # topic and msgtype information is available on .connections dictionary
        for connection in reader.connections.values():
            print(connection.topic,'|', connection.msgtype)

        for connection, timestamp, rawdata in reader.messages():
            if connection.topic == '/waterlinked/acoustic_position/raw':
                 msg = deserialize_cdr(ros1_to_cdr(rawdata, connection.msgtype), connection.msgtype)
                 print(msg)


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

def rosbag_inv(rosbag_path):

    with Reader(rosbag_path) as reader:
        for connection in reader.connections.values():
            print(connection.topic,'|', connection.msgtype)

        for connection, timestamp, rawdata in reader.messages(connections=reader.connections.values()):
            msg = deserialize_cdr(rawdata, connection.msgtype)
            print(msg.header.stamp.sec)


def load_all_imgs(data_path_extracted):
    images = []
    for filename in os.listdir(data_path_extracted):
        img = cv2.imread(os.path.join(data_path_extracted, filename))
        if img is not None:

            images.append(img)
    return np.asarray(images)

def resize_imgs(data_path_extracted):

    for filename in os.listdir(data_path_extracted):
        img = cv2.imread(os.path.join(data_path_extracted, filename))
        if img is not None:
            print(img.shape)
            if img.shape[1]== 256:
                size = (img.shape[1]*2,img.shape[0])
                img = cv2.resize(img,size)
                print(img.shape)
                cv2.imwrite(os.path.join(data_path_extracted, filename), img)

#imgs = load_all_imgs(r"D:\MachineLearning\SonarData\SonarDataSets\Real\rosbag2_2022_04_06-11-27_20")
imgs = resize_imgs(r"D:\MachineLearning\SonarData\SonarDataSets\Real\rosbag2_2022_04_06-11-41_16")
#get_xyz()
#rosbag_inv(rosbag_path)
# for i in imgs:
#     cv2.imshow("minus mean", (i - mean*2)/255.)
#     cv2.imshow("i", i/255.)
#     cv2.imshow("mean", mean*4/255.)
#     cv2.waitKey(0)