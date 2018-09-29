import pickle
from easydict import EasyDict as edict

cfg = edict()

cfg.img_w = 640
cfg.img_h = 480

cfg.img_ip = '127.0.0.1'
cfg.arm_ip = '192.168.5.40'
cfg.img_port = 9003
cfg.arm_port = 9005

cfg.model_path = "model_files/gqcnn_model"

cfg.crop_x_start = 330
cfg.crop_y_start = 150
cfg.crop_width = 240
cfg.crop_height = 240

cfg.table_height = 580
cfg.height_bias = 145

cfg.depth_bias = 25

cfg.antipodal_th = 30

cfg.img_list_len = 2

cfg.z_num = 3

cfg.save_dir = 'whole_pro_data'

cfg.gd_th = 700

f = open('cali.pkl', 'rb')
cali = pickle.load(f)
cfg.cx = cali['cx']
cfg.cy = cali['cy']
cfg.fx = cali['fx']
cfg.fy = cali['fy']

