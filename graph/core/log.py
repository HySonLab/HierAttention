# Create a simple logger that can log training curves and final performance
# from torch.utils.tensorboard import SummaryWriter  # tensorboard
import logging
import os
import sys
import shutil
import datetime
from datetime import date

def config_logger(cfg, OUT_PATH="results/", time=False):
    # time option is used for debugging different model architecture.
    data_name = cfg.dataset

    # # generate config_string
    # os.makedirs(OUT_PATH, exist_ok=True)
    # if cfg.logfile is None:
    #     config_string = cfg.model.name + '_' + cfg.model.gnn_type
    #     if cfg.depth > 0:
    #         config_string = config_string + '_depth' + str(cfg.depth)
    # else:
    #     config_string = cfg.logfile

    # # setup tensorboard writer
    # writer_folder = os.path.join(OUT_PATH, data_name, config_string)
    # if time:
    #     writer_folder = os.path.join(
    #         writer_folder, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    # if os.path.isdir(writer_folder):
    #     shutil.rmtree(writer_folder)  # reset the folder, can also not reset
    # writer = SummaryWriter(writer_folder)

    # # setup logging
    # logger = logging.getLogger()
    # logger.setLevel(logging.INFO)
    # logger_filer = os.path.join(OUT_PATH, data_name, 'summary.log')
    # fh = logging.FileHandler(logger_filer)
    # fh.setLevel(logging.INFO)
    # fh.setFormatter(logging.Formatter('%(message)s'))
    # logger.addHandler(fh)

    # redirect stdout print, better for large scale experiments
    log_folder_name = cfg.expname + '-' + str(date.today())
    os.makedirs(f'logs/{data_name}/{log_folder_name}', exist_ok=True)

    if not cfg.debug:
        sys.stdout = open(f'logs/{data_name}/{log_folder_name}/log.txt', 'w')

    # log configuration
    print("-"*50)
    print(cfg)
    print("-"*50)
    print('Time:', datetime.datetime.now().strftime("%Y/%m/%d - %H:%M"))

    # return writer, logger
    return None, None