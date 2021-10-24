import cv2
import sys
import datetime
import glob
import time
import numpy as np
from detectron2 import config

import _pickle as cPickle
import os
from detectron2.engine import DefaultTrainer
from numpy.lib.arraysetops import unique
from detectron2.structures.boxes import Boxes
from detectron2.structures.masks import BitMasks
from sys import meta_path, path
import detectron2.data.transforms as T
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog
from detectron2.data import DatasetMapper   # the default mapper
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import(
     DefaultPredictor
)
from detectron2.structures import Instances, instances
import torch
import scipy.ndimage
import copy
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from PIL import Image
def load_mask(image,dataset_dict):
    """Generate instance masks for the objects in the image with the given ID.
    """
    #masks, coords, class_ids, scales, domain_label = None, None, None, None, None
    
    image = image.copy()
    id = dataset_dict["id"]
    image_id = dataset_dict["image_id"]
    gt_dir = os.path.join("/data2","qiweili","cat","gt",str(id),image_id+'.png')
    #print(gt_dir)
    mask = cv2.imread(gt_dir )[:, :, :3]#本来就二维，第三个2的参数可以去掉 

    return image , mask 

def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.
    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[0], 4], dtype=np.int32)
    for i in range(mask.shape[0]):
        m = mask[i, :, :]
        #对于一张照片中的不同类的物体分别处理
        # Bounding box.
        #返回mask中物体所在的行坐标
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        #返回mask中物体所在的列坐标
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)

j = 0
def file_name(file_dir):
    list_dir = []
    for _, _, files in os.walk(file_dir):  
        list_dir =files #当前路径下所有非目录子文件 
    return list_dir
def cat_train_function( ):
    """Load a subset of the CAMERA dataset.
    dataset_dir: The root directory of the CAMERA dataset.
    subset: What to load (train, val)
    if_calculate_mean: if calculate the mean color of the images in this dataset
    """
    global j 
    print('begin load cat')
    dataset_dir = os.path.join("/data2","qiweili","cat","train")
    folder_list = [name for name in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, name))]
    list = []
    for i in range(1,len(folder_list)+1):
        cat_id = i 
        cat_dir = os.path.join(dataset_dir,str(i))
        image_path_all =file_name(cat_dir)
        for i in range(len(image_path_all)) :
            image_path = os.path.join(cat_dir,image_path_all[i])
            t = image_path_all[i]
            t = t.split('.')[0]
            list.append({'image_id':t,'image_path':image_path,
                            'id':cat_id})
            j=j+1
    
    print(j)
    print('load cat successfully')
    return list

def mapper(dataset_dict):
    #print(dataset_dict)

    #time.sleep(1)
    #print('use mapper')
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

    image_path = dataset_dict["image_path"]
    #print(image_dir)
    #print(image_dir)

    image = cv2.imread(image_path)[:, :, :3]
    
    #image = image[:, :, ::-1]
    #image_d = image[:, :, ::-1].copy()
    # If grayscale. Convert to RGB for consistency.
    #cv2.imwrite('origin.jpg',image)
    if image.ndim != 3:
        image =  cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

  
    image ,mask= load_mask(image,dataset_dict)

    return image,mask 

if __name__ == "__main__":
    name = {"1":"杜若",
        "2":"小宝",
        "3":"雪风"}
    
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    cfg = get_cfg()

    cfg.DATALOADER.ASPECT_RATIO_GROUPING = False
    cfg.merge_from_file(
        "configs/mask_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.DATALOADER.NUM_WORKERS = 0


    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    

    cfg.MODEL.WEIGHTS = os.path.join("logs", "model_0009999.pth") 

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # 设置一个阈值
    predictor = DefaultPredictor(cfg)
    print('model built')
    image =cv2.imread("3.png")
        
    out = predictor(image)
    bbox=out["instances"].pred_boxes.tensor.int().cpu().numpy()
    #print(bbox)
    #print('\n\n\n\n\n\n\n\n\n\n')
    print('cat is',name[str(out["instances"].pred_classes.cpu().numpy()[0])])
    print('scores:',out["instances"].scores.cpu().numpy()[0])
    #image = image[:,:,::-1].copy()
    for i in range(len(bbox)):
        cv2.rectangle(image, (bbox[i][0], bbox[i][1]), (bbox[i][2], bbox[i][3]),
                    color=(0,0,255), thickness = 10 )
    cv2.imwrite('after.jpg',image)
    #input()
    '''
    for i in cat_train_function():
        image , mask = mapper(i)
        
        out = predictor(image)
        bbox=out["instances"].pred_boxes.tensor.int().cpu().numpy()
        #print(bbox)
        print(out["instances"].pred_classes)
        print(out["instances"].scores)
        #image = image[:,:,::-1].copy()
        for i in range(len(bbox)):
            cv2.rectangle(image, (bbox[i][0], bbox[i][1]), (bbox[i][2], bbox[i][3]),
                        color=(0,0,255), thickness = 10 )
        cv2.imwrite('after.jpg',image)
        input()
    '''



