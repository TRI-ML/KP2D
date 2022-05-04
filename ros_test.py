from rosbags.rosbag2 import Reader
from rosbags.rosbag1 import Reader as Reader1
from rosbags.serde import deserialize_cdr
import numpy as np
import cv2
import os
from rosbags.serde import deserialize_cdr, ros1_to_cdr
import math
import pandas as pd

import csv

imgs = []
data_path = r"D:\MachineLearning\SonarData\SonarDataSets\Real"
rosbag_paths= [r'C:\Users\Dr. Paul von Immel\Downloads\rosbag2_2022_04_06-11-41_16-20220408T162332Z-001\rosbag2_2022_04_06-11-41_16',
               r'C:\Users\Dr. Paul von Immel\Downloads\rosbag2_2022_04_06-11-45_44-20220408T162334Z-001\rosbag2_2022_04_06-11-45_44',
               r'C:\Users\Dr. Paul von Immel\Downloads\rosbag2_2022_04_06-11-27_20-20220408T125835Z-001\rosbag2_2022_04_06-11-27_20'
               ]

folderNames = [os.path.basename(f) for f in rosbag_paths]
data_path_extracted = [os.path.join(data_path, f) for f in folderNames]

def get_xyz():
    with Reader1(r"C:\Users\Dr. Paul von Immel\Downloads\2022-04-06-11-40-04.bag") as reader:
        csvfile =  open('ros_data.csv', 'w', newline='')
        spamwriter = csv.writer(csvfile, delimiter=',',quotechar = '|', quoting = csv.QUOTE_MINIMAL)
        spamwriter.writerow(['s', 'n','x','y','z','roll','pitch','yaw'])
        # topic and msgtype information is available on .connections dictionary
        for connection in reader.connections.values():
            print(connection.topic,'|', connection.msgtype)

        for connection, timestamp, rawdata in reader.messages():
            if connection.topic == '/waterlinked/robot_pose/filtered':
                msg = deserialize_cdr(ros1_to_cdr(rawdata, connection.msgtype), connection.msgtype)
                print(msg.pose.pose.orientation)
                sec = msg.header.stamp.sec
                nanosec = msg.header.stamp.nanosec
                x = msg.pose.pose.position.x
                y = msg.pose.pose.position.y
                z = msg.pose.pose.position.z
                q = msg.pose.pose.orientation
                r,p,ya = q_to_e(q)
                spamwriter.writerow([sec, nanosec, x, y, z, r, p, ya])
    csvfile.close()

def q_to_e(q):
    x = q.x
    y = q.y
    z = q.z
    w = q.w

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)*180/math.pi

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)*180/math.pi

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)*180/math.pi

    return roll_x, pitch_y, yaw_z  # in radians

def rosbag_to_jpg(rosbag_paths, data_path):

    for rosbag_path in rosbag_paths:
        folderName = os.path.basename(rosbag_path)
        data_path_extracted = os.path.join(data_path, folderName)

        if not os.path.exists(data_path_extracted):
            os.mkdir(data_path_extracted)
        i = 0
        csvfile =  open(os.path.join(data_path_extracted,'time.csv'), 'w', newline='')
        spamwriter = csv.writer(csvfile, delimiter=',',quotechar = '|', quoting = csv.QUOTE_MINIMAL)
        spamwriter.writerow(['i', 's', 'n'])
        with Reader(rosbag_path) as reader:
            for connection, timestamp, rawdata in reader.messages(connections=reader.connections.values()):
                msg = deserialize_cdr(rawdata, connection.msgtype)
                img_reshaped = np.reshape(msg.data, [msg.height,msg.width])
                spamwriter.writerow([i, msg.header.stamp.sec, msg.header.stamp.nanosec])
                #filename = "frame{:>05}.jpg".format(i)
                #cv2.imwrite(os.path.join(data_path_extracted, filename), img_reshaped)
                i += 1
        csvfile.close()

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

def match_data(data_path_extracted):
    xyz_data = pd.read_csv('ros_data.csv')
    min_time = xyz_data.s.min()
    max_time = xyz_data.s.max()
    for d in data_path_extracted:
        print(d)
        pic_times = pd.read_csv(os.path.join(d, 'time.csv'))
        min_pic_time = pic_times.s.min()
        max_pic_time = pic_times.s.max()
        if (min_time<min_pic_time and max_time > min_pic_time) and (min_time<max_pic_time and max_time > max_pic_time):

            start = (pic_times.s[0], pic_times.n[0])
            end = (pic_times.s[pic_times.s.size-1], pic_times.n[pic_times.n.size-1])
            dF = xyz_data[pic_times.s.min() < xyz_data['s']]
            dF = dF[dF['s'] < pic_times.s.max()]
            print(start)
            print(end)


            return pic_times, dF



#imgs = load_all_imgs(r"D:\MachineLearning\SonarData\SonarDataSets\Real\rosbag2_2022_04_06-11-27_20")
#imgs = resize_imgs(r"D:\MachineLearning\SonarData\SonarDataSets\Real\rosbag2_2022_04_06-11-41_16")
#sec,nanosec,x,y,z = get_xyz()
#rosbag_to_jpg(rosbag_paths, data_path)
p, x = match_data(data_path_extracted)
#get_xyz()
#rosbag_to_jpg(rosbag_path, data_path_extracted)
# for i in imgs:
#     cv2.imshow("minus mean", (i - mean*2)/255.)
#     cv2.imshow("i", i/255.)
#     cv2.imshow("mean", mean*4/255.)
#     cv2.waitKey(0)