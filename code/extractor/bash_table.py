import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
from tqdm import tqdm
import base64
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
import pickle
from predictor import VisualizationDemo
from pdf2image import convert_from_path
import numpy as np
import csv
import os
from multiprocessing import Process, Queue
import multiprocessing as mp
# constants
WINDOW_NAME = "COCO detections"

CONFIG_PATH = "./models/All_X152.yaml"
OPT_PATH = "./models/model_final.pth"

PDF_DIR = "./pdfs"
OUTPUT_DIR = "./detect_output"

myconfidence_threshold = 0.9

def all_files_path(rootDir):                       
    for root, dirs, files in os.walk(rootDir):         
        for file in files:                        
            file_path = os.path.join(root, file)
            filepaths.append(file_path)            
        for dir in dirs:                                      
            dir_path = os.path.join(root,dir)                              
            all_files_path(dir_path)               

filepaths = []
all_files_path(PDF_DIR)
filepaths = sorted(list(set(filepaths)))
print(len(filepaths))

mopts = ["MODEL.WEIGHTS", OPT_PATH]

def setup_cfg():
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(CONFIG_PATH)
    cfg.merge_from_list(mopts)
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = myconfidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = myconfidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = myconfidence_threshold
    cfg.freeze()
    return cfg
    
    
def base64toCv(base64_src):
    img_b64decode = base64.b64decode(base64_src)  # base64解码
    img_array = np.fromstring(img_b64decode, np.uint8)  # 转换np序列
    img_cv = cv2.imdecode(img_array, cv2.COLOR_BGR2RGB)  # 转换OpenCV格式
    return img_cv

cfg = setup_cfg()

demo = VisualizationDemo(cfg)

def parse_pdf(pdf_id,pdf_path):
    try:
        img_pages = convert_from_path(pdf_path)
        finalresult = {}
        for i,img in enumerate(img_pages):
            img = np.array(img)
            # img = cv2.imdecode(img, cv2.COLOR_BGR2RGB)


            predictions, visualized_output,final_res = demo.run_on_image(img)
            print(final_res)
            finalresult[f'{pdf_id}_%03d'%int(i)] = final_res

        return finalresult
    except:
        print('error')
        return {}

def pred_img(img):
    # cfg = setup_cfg()

    # demo = VisualizationDemo(cfg)
    predictions, visualized_output,final_res = demo.run_on_image(img)
    print(final_res)
    return final_res
    

for file in tqdm(filepaths):
    if '.pdf' in file:
        output_file = os.path.join(OUTPUT_DIR, file.split("/")[-1].replace(".pdf", ".txt"))
        if os.path.exists(output_file):
            continue
        else:
            fresult = open(output_file,'a')
            fresult.close()
        resdict = parse_pdf(file,file)

        for key in resdict:
            print(str(key) + ":" + str(resdict[key]))
            fresult = open(output_file,'a')
            fresult.write(str(key) + ':' + str(resdict[key]) + "\n")
            fresult.close()
