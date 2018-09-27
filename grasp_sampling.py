import cv2
import os
import numpy as np
import random
from foxarm.common import image
import matplotlib.pyplot as plt
from scipy import misc

from cfgs.config import cfg

# cx, cy, fx, fy = 322.7521286, 257.23471352, 578.04136686, 580.96473333 
# cx, cy, fx, fy = 298.55645619, 242.75084457, 596.24139992, 594.00542779
cx, cy, fx, fy = 297.33638738, 242.0229963, 595.274641, 593.10999329
GRIPPER_WIDTH_IN_PIXEL = 30

def grasp_sample(raw_data, grasp_num, sub_dir):
    # calculate the gradients
    data = (raw_data - raw_data.min()) / (raw_data.max() - raw_data.min()) * 255
    data = data.astype(np.uint8)

    sobelx = cv2.Sobel(data, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(data, cv2.CV_64F, 0, 1, ksize=5)
    gd = np.sqrt(np.abs(sobelx) ** 2 + np.abs(sobely) ** 2)

    ret, binary = cv2.threshold(gd, 1000, 255, cv2.THRESH_BINARY)

    save_path = os.path.join(cfg.save_dir, sub_dir, 'binary.jpg')
    cv2.imwrite(save_path, binary)

    # find points with high gradients and calculate gradient directions
    gd_direction = np.arctan2(sobely, sobelx) * 180 / np.pi
    nonzero_coord = np.nonzero(binary)
    candidate_points = []
    point_num = nonzero_coord[0].shape[0]
    candidate_directions = gd_direction[nonzero_coord]
    for idx in range(point_num):
        candidate_points.append([nonzero_coord[0][idx],
                                 nonzero_coord[1][idx],
                                 candidate_directions[idx]])

    # sample antipodal points, and draw
    candidate_grasps = []
    for i in range(grasp_num):
        while True:
            sample = random.sample(candidate_points, 2)
            angle = np.abs(sample[0][2] - sample[1][2])
            distance = np.linalg.norm(np.array(sample[0][:-1]) - np.array(sample[1][:-1]))
            if np.abs(angle - 180) < cfg.antipodal_th and distance <= GRIPPER_WIDTH_IN_PIXEL:
                candidate_grasps.append(sample)
                break

    return candidate_grasps, binary

def visualize_grasp(samples, binary, sub_dir):
    # visulize grasp
    color = cv2.cvtColor(binary.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    for i, sample in enumerate(samples):
        color_vis = np.copy(color)
        delta = 1
        for pt in sample:
            y = pt[0]
            x = pt[1]
            cv2.rectangle(color_vis,
                          (x - delta, y - delta),
                          (x + delta, y + delta),
                          (255, 0, 0),
                          2)
        save_path = os.path.join(cfg.save_dir, sub_dir, 'color_vis_%02d.jpg' % i)
        cv2.imwrite(save_path, color_vis)

def depth2cloud(sample, raw_data):
    p1 = np.array(sample[0][:2])
    p2 = np.array(sample[1][:2])

    p = ((p1 + p2) / 2).astype(np.int64)
    z = raw_data[p[0], p[1]]

    y1 = ((p1[0] + cfg.crop_y_start) * z - z * cy) / fy
    x1 = ((p1[1] + cfg.crop_x_start) * z - z * cx) / fx
    y2 = ((p2[0] + cfg.crop_y_start) * z - z * cy) / fy
    x2 = ((p2[1] + cfg.crop_x_start) * z - z * cx) / fx

    p1_3d = np.array([x1, y1, z])
    p2_3d = np.array([x2, y2, z])

    return p1_3d, p2_3d

def align(raw_data, samples, table_height, sub_dir):
    # rotate, crop and resize
    gqcnn_ims = []
    gqcnn_depth = []
    depth_image = image.DepthImage(raw_data.astype(np.float))
    for i, sample in enumerate(samples):
        p1 = np.array(sample[0][:2])
        p2 = np.array(sample[1][:2])
        gp_center = (p1 + p2) / 2
        img_center = np.array(raw_data.shape) / 2
        trans = img_center - gp_center
        diff = p1 - p2
        angle = np.arctan2(diff[0], diff[1])
        trans_image = depth_image.transform(trans, angle)
        crop_image = trans_image.crop(80, 80)
        resize_image = crop_image.resize((32, 32))
        img = np.expand_dims(resize_image.data, -1)
        gp_center_int = gp_center.astype(np.int64)
        obj_height = raw_data[gp_center_int[0], gp_center_int[1]]
        candidate_z = np.linspace(obj_height, table_height, cfg.z_num)
        for z in candidate_z:
            gqcnn_ims.append(img)
            gqcnn_depth.append(z)

        save_path = os.path.join(cfg.save_dir, sub_dir, 'resize_image_%02d.png' % i)
        misc.imsave(save_path, resize_image.data)
    return np.array(gqcnn_ims), np.array(gqcnn_depth)


if __name__ == '__main__':
    depth_img = np.load('../data/100.pkl')
    samples, binary_img = grasp_sample(depth_img, 3)
    grasp_pos = depth2cloud(samples[0], depth_img)
    visualize_grasp(samples, binary_img)
    gqcnn_im, gqcnn_depth = align(depth_img, samples)

