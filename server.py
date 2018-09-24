import socket
import time
import struct
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorpack import *

from code_gqcnn.train import GQCNN
from recv_img import *
from grasp_sampling import *
from cfgs.config import cfg

# Load GQCNN model and predict_func
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
sess_init = SaverRestore(cfg.model_path)
model = GQCNN() 
predict_config = PredictConfig(session_init=sess_init,
                               model=model,
                               input_names=["input", "pose"],
                               output_names=["Softmax"])
predict_func = OfflinePredictor(predict_config)


# Connect with depth camera
img_conn = construct_conn()
recv_img_thread = RecvImgThread(img_conn)
recv_img_thread.start()

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
addr = (cfg.local_ip, cfg.arm_port)
sock.bind(addr)
sock.listen(1)
print('Waiting for robot connection...')
conn, addr = sock.accept()
print('robot connected with %s:%s' % addr)

while True:
    msg = conn.recv(1024)
    time.sleep(1)
    # msg = b'c'
    print(msg.decode('utf-8'))

    if msg.decode('utf-8') == 'c':
        depth_img = recv_img_thread.get_img()
        # crop the img
        crop_img = depth_img[cfg.crop_y_start:cfg.crop_y_start+cfg.crop_height,
                             cfg.crop_x_start:cfg.crop_x_start+cfg.crop_width]
        # sample grasps
        samples, binary_img = grasp_sample(crop_img,
                                           grasp_num=10)
        # save grasps as images
        visualize_grasp(samples, binary_img)

        # align the grasps and run the model
        gqcnn_imgs, gqcnn_depths = align(crop_img,
                                         samples,
                                         table_height=cfg.table_height,
                                         z_num=3, save_dir=cfg.save_dir)
        gqcnn_imgs = (gqcnn_imgs + cfg.height_bias) / 1000
        gqcnn_depths = (gqcnn_depths + cfg.height_bias) / 1000
        prediction = predict_func(gqcnn_imgs, gqcnn_depths)

        # choose the best sample, and transform it and send through tcp socket
        best_sample = samples[np.argmax(prediction[0][:, -1])]
        p1_3d, p2_3d = depth2cloud(best_sample, depth_img)

        sock.send('ffffff', p1_3d[0], p1_3d[1], p1_3d[2], p2_3d[0], p2_3d[1], p2_3d[2])
    elif msg.decode('utf-8') == 'q':
        break

recv_img_thread.stop()
recv_img_thread.join()
conn.close()
