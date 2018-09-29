import struct
import datetime
import copy
import sys
import numpy as np
from scipy import misc
import time
import socket
import pickle
import os
from threading import Thread

from cfgs.config import cfg

def construct_conn():

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    addr = (cfg.img_ip, cfg.img_port)
    sock.bind(addr)
    sock.listen(1)
    print('Waiting for image connection...')
    conn, addr= sock.accept()
    print('image connected with %s:%s' % addr)

    return conn

class RecvImgThread(Thread):

    def __init__(self, conn):
        Thread.__init__(self)
        self.conn = conn
        self.imgs = [None] * cfg.img_list_len
        self.recv = True

    def run(self):
        frame_data = []
        img_size = cfg.img_h * cfg.img_w
        cur_frame = [None] * img_size
        remain_value = ()
        while self.recv:
            line_num = 1
            # for i in range(cfg.img_h // line_num):
            cur_frame[:len(remain_value)] = remain_value
            cur_idx = len(remain_value)
            while True:
                data = self.conn.recv(cfg.img_w * line_num * 2)
                value = struct.unpack('<%dH' % (len(data) / 2), data)
                if cur_idx + len(value) >= img_size:
                    cur_frame[cur_idx:img_size] = value[:img_size - cur_idx]
                    remain_value = value[img_size - cur_idx:]
                    break
                else:
                    cur_frame[cur_idx:cur_idx + len(value)] = value
                    cur_idx += len(value)

            cur_frame_ary = np.array(cur_frame).reshape(cfg.img_h, cfg.img_w)

            # insert data to the head of the list
            # print('receive one frame')
            for i in range(cfg.img_list_len - 1, 0, -1):
                self.imgs[i] = self.imgs[i-1]
            self.imgs[0] = cur_frame_ary

            frame_data = frame_data[img_size:]

    def stop(self):
        self.recv = False

    def get_img(self, sub_dir):
        dir_path = os.path.join(cfg.save_dir, sub_dir)
        imgs_buffer = copy.deepcopy(self.imgs)

        imgs_buffer = np.array(imgs_buffer)
        img_process = np.zeros((cfg.img_h, cfg.img_w))

        mask = np.where(imgs_buffer==0, 0, 1)
        mask = np.sum(mask, 0)
        mask[mask==0] = 1
        img_process = np.sum(imgs_buffer, 0) / mask

        done_avg = time.time()

        c = 0
        while((img_process==0).any() and c < 2):
            c += 1
            pad_arg = np.argwhere(img_process==0)
            for i, j in pad_arg:
                grid = img_process[i-1:i+2, j-1:j+2]
                num = np.argwhere(grid!=0).shape[0]
                if num:
                    img_process[i, j] = int(np.sum(grid) / num)

        print('pad time: %g' % (time.time() - done_avg))

        save_path = os.path.join(dir_path, 'depth.jpg')
        misc.imsave(save_path, img_process)
        f = open(os.path.join(dir_path, 'depth.pkl'), 'wb')
        pickle.dump(img_process, f)
        print('Done process imgs')
        return img_process


