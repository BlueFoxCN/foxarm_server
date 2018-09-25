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
        self.list_len = 10
        self.imgs = [None] * self.list_len
        self.recv = True

    def run(self):
        while self.recv:
            frame_data = []
            line_num = 1
            for i in range(cfg.img_h // line_num):
                data = self.conn.recv(cfg.img_w * line_num * 2)
                value = struct.unpack('<%dH' % (cfg.img_w * line_num), data)
                frame_data.append(value)

            frame_data = np.array(frame_data)
            frame_data = frame_data.reshape(cfg.img_h, cfg.img_w)

            # insert data to the head of the list
            for i in range(self.list_len - 1, 0, -1):
                self.imgs[i] = self.imgs[i-1]
            self.imgs[0] = frame_data

    def stop(self):
        self.recv = False

    def get_img(self, sub_dir):
        dir_path = os.path.join(cfg.save_dir, sub_dir)
        imgs_buffer = copy.deepcopy(self.imgs)

        img_process = imgs_buffer[0]
        for img in imgs_buffer[1:]:
            arg = np.where(img_process == 0)
            img_process[arg] = img[arg] 
        c = 0
        while((img_process==0).any() and c < 3):
            c += 1
            pad_arg = np.argwhere(img_process==0)
            for i, j in pad_arg:
                grid = img_process[i-1:i+2, j-1:j+2]
                num = np.argwhere(grid!=0).shape[0]
                if num:
                    img_process[i, j] = int(np.sum(grid) / num)

        save_path = os.path.join(dir_path, 'depth.jpg')
        misc.imsave(save_path, img_process)
        print('Done process imgs')
        return img_process


