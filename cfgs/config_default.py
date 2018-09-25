from easydict import EasyDict as edict

cfg = edict()

cfg.img_w = 640
cfg.img_h = 480

cfg.img_ip = '127.0.0.1'
cfg.arm_ip = '192.168.5.42'
cfg.img_port = 9003
cfg.arm_port = 9005

cfg.model_path = "model_files/gqcnn_model"

cfg.crop_x_start = 250
cfg.crop_y_start = 100
cfg.crop_width = 200
cfg.crop_height = 280

cfg.table_height = 580
cfg.height_bias = 145

cfg.antipodal_th = 30

cfg.z_num = 3

cfg.save_dir = 'whole_pro_data'
