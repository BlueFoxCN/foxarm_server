import socket
import datetime
import argparse
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

parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true', help='In debug mode, user input is used instead of the remote robot app.')
args = parser.parse_args()


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

if args.debug == False:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    addr = (cfg.arm_ip, cfg.arm_port)
    sock.bind(addr)
    sock.listen(1)
    print('Waiting for robot connection...')
    conn, addr = sock.accept()
    print('robot connected with %s:%s' % addr)

while True:
    if args.debug == False:
        msg = conn.recv(1024)
        msg = msg.decode('utf-8')
        time.sleep(1)
    else:
        msg = input('Continue(c) or Quit(q)?')

    if msg == 'c':
        sub_dir = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        os.mkdir(os.path.join(cfg.save_dir, sub_dir))
        depth_img = recv_img_thread.get_img(sub_dir)
        # crop the img
        crop_img = depth_img[cfg.crop_y_start:cfg.crop_y_start+cfg.crop_height,
                             cfg.crop_x_start:cfg.crop_x_start+cfg.crop_width]
        # sample grasps
        samples, binary_img = grasp_sample(crop_img,
                                           grasp_num=10,
                                           sub_dir=sub_dir)
        # save grasps as images
        visualize_grasp(samples, binary_img, sub_dir)

        # align the grasps and run the model
        gqcnn_imgs, gqcnn_depths = align(crop_img,
                                         samples,
                                         table_height=cfg.table_height,
                                         sub_dir=sub_dir)
        gqcnn_imgs = (gqcnn_imgs + cfg.height_bias) / 1000
        gqcnn_depths = (gqcnn_depths + cfg.height_bias) / 1000
        prediction = predict_func(gqcnn_imgs, gqcnn_depths)

        # choose the best sample, and transform it and send through tcp socket
        best_sample = samples[np.argmax(prediction[0][:, -1]) // cfg.z_num]
        result_img = cv2.cvtColor(np.expand_dims(crop_img, -1).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        p1 = best_sample[0][0:2]
        p2 = best_sample[1][0:2]
        cv2.rectangle(result_img,
                      (p1[1], p1[0]),
                      (p1[1], p1[0]),
                      (0, 0, 255),
                      3)
        cv2.rectangle(result_img,
                      (p2[1], p2[0]),
                      (p2[1], p2[0]),
                      (0, 0, 255),
                      3)
        cv2.imwrite(os.path.join(cfg.save_dir, sub_dir, 'final.jpg'), result_img)
        print(best_sample)
        p1_3d, p2_3d = depth2cloud(best_sample, crop_img)

        if args.debug == False:
            sock.send('ffffff', p1_3d[0], p1_3d[1], p1_3d[2], p2_3d[0], p2_3d[1], p2_3d[2])
        else:
            print("[%.2f, %.2f, %.2f]" % (p1_3d[0], p1_3d[1], p1_3d[2]))
            print("[%.2f, %.2f, %.2f]" % (p2_3d[0], p2_3d[1], p2_3d[2]))
    elif msg == 'q':
        break

recv_img_thread.stop()
recv_img_thread.join()
if args.debug == False:
    conn.close()
