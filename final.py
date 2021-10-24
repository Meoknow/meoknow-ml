from numpy.lib.arraysetops import unique
from numpy.lib.function_base import angle
from torch.utils.data import dataloader, dataset
from detectron2.structures.boxes import Boxes
from detectron2.structures.masks import BitMasks
from sys import meta_path, path
import detectron2.data.transforms as T
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog
from detectron2.data import DatasetMapper   # the default mapper
from detectron2.data.common import AspectRatioGroupedDataset, DatasetFromList, MapDataset
from detectron2.data import detection_utils as utils
from detectron2.data import (
    MetadataCatalog)
from detectron2.data.build import (
    get_detection_dataset_dicts,
    _train_loader_from_config,
    _test_loader_from_config,
    build_batch_data_loader,
    trivial_batch_collator
)
from detectron2.evaluation import DatasetEvaluator
import logging
from detectron2.utils.logger import setup_logger
from detectron2.modeling import build_model
import scipy.ndimage
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import (DefaultTrainer ,hooks)
from detectron2.engine.hooks import (CallbackHook,EvalHook)
from detectron2.structures import Instances, instances
from detectron2.data.samplers import InferenceSampler, RepeatFactorTrainingSampler, TrainingSampler
import torch
import copy
import os
import numpy as np
import cv2
import time
import glob
import random
from PIL import Image
from collections.abc import Iterable
from pycocotools.coco import COCO
from detectron2.config import configurable
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)


def load_mask(image,dataset_dict):
    """Generate instance masks for the objects in the image with the given ID.
    """
    #masks, coords, class_ids, scales, domain_label = None, None, None, None, None
    
    image = image.copy()
    id = dataset_dict["id"]
    image_id = dataset_dict["image_id"]
    gt_dir = os.path.join("/data2","qiweili","cat","gt",str(id),image_id+'.png')
    #print(gt_dir)
    #print('mask',gt_dir)
    mask = cv2.imread(gt_dir )[:, :, :3]#本来就二维，第三个2的参数可以去掉 

    return image , mask 

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
    #print(folder_list)
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

def cat_val_function( ):
    """Load a subset of the CAMERA dataset.
    dataset_dir: The root directory of the CAMERA dataset.
    subset: What to load (train, val)
    if_calculate_mean: if calculate the mean color of the images in this dataset
    """
    global j 
    print('begin load cat')
    dataset_dir = os.path.join("/data2","qiweili","cat","test")
    folder_list = [name for name in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, name))]
    list = []
    #print(folder_list)
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
    
    #画出aug前,没有reszie的图
    #mask_before, coord_, class_ids_, scales_, domain_label_ = load_mask(dataset_dict)
    

    #mask_before=mask
    #bbox_before = extract_bboxes(mask_before)
    augs = T.AugmentationList([
        T.RandomFlip(prob=0.5),
        T.RandomRotation(angle=[-5,5],expand=False)
        #T.RandomCrop("absolute", (640, 640))
    ])  # type: T.Augmentation
    augs2 = T.AugmentationList([
        T.RandomBrightness(0.9, 1.1),
    ])
    # Define the augmentation input ("image" required, others optional):
    input1 = T.AugInput(image)
    # Apply the augmentation:
    transform = augs(input1)  # type: T.Transform
    image = input1.image  # new image
    #print(image_transformed.shape)
    #cv2.imwrite("before.jpg",image[:,:,::-1])
    #cv2.imwrite("image_after1.jpg",image_transformed[:,:,::-1])
    mask = transform.apply_image(mask)
    #cv2.imwrite("mask_after.jpg",mask_transformed[:,:,::-1])
    #print(image2_transformed-image_transformed)
    transform2 = augs2(input1) 
    image = input1.image  # new image
    #cv2.imwrite("image_after2.jpg",image_transformed2[:,:,::-1])
    class_ids =np.array( [dataset_dict["id"]])
    mask= mask[np.newaxis,:,:,0]
    #print(class_ids)
    #input()
    #上需要改动
    #print(image.shape)
    #if(mask.shape[0]==0) :
    #    exit(0)
    #print(image_path)
    #print(image.shape)
    
    image=torch.from_numpy(image.transpose(2, 0, 1))
    
    instances = Instances(tuple([image.shape[1],image.shape[2]]))
    instances.gt_classes = torch.from_numpy(class_ids).long()
    instances.gt_boxes = BitMasks(mask).get_bounding_boxes()
    instances.gt_masks =BitMasks(mask)
    #print(Boxes(bbox))
    
    #print('mask',mask.shape)
    #print('--------------')
    instances.gt_masks =BitMasks(mask)
    if len(class_ids) == 0:
        return None
    #print(BitMasks(mask).get_bounding_boxes())
    #print('--------------')
    #instances.scales = scales
    #print(instances)
    return {
       # create the format that the model expects
        "image": image,
        "height" :image.shape[1],
        "width":image.shape[0]  ,
        "instances" : instances ,
        #"coord" :coord,
        #"scales" :scales
    }


class Trainer(DefaultTrainer):

    def build_train_loader(cls, cfg):
        dataloader = build_detection_train_loader(cfg, mapper=mapper)
        return dataloader
    @classmethod
    def mytest(cls, cfg, model,dataset_name):
        #model.eval()
        loss = {}
        re ={}
        with torch.no_grad():
            l = len( dataset_name)
            for i in dataset_name :
                a = model(i)
                if not loss :
                    loss = a
                else :
                    for key ,value in a.items():
                        loss[key]=loss[key]+value
            for key ,value in loss.items():
                re['val_'+key] = loss[key].item()/l
            #print(re)
        model.train()
        return re





if __name__ == "__main__":

    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    cfg = get_cfg()
    #cfg = get_nocsrcnn_cfg_defaults(cfg)
    cfg.DATALOADER.ASPECT_RATIO_GROUPING = False
    cfg.merge_from_file(
        
        "configs/mask_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.DATALOADER.NUM_WORKERS = 0
    DatasetCatalog.register("cat_train_dataset", cat_train_function)
    DatasetCatalog.register("cat_val_dataset", cat_val_function)

    cfg.SOLVER.BASE_LR = 0.001
    cfg.DATASETS.TRAIN = ("cat_train_dataset",)
    cfg.DATASETS.TEST = ("cat_val_dataset",)
    #cfg.DATASETS.WEIGHT = [3,1,1]
    
    #os.environ['CUDA_LAUNCH_BLOCKING'] = str(1)
    cfg.SOLVER.IMS_PER_BATCH = 2
    #cfg.SOLVER.MAX_ITER = 400
    #cfg.SOLVER.STEPS = (200, 300)
    

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    
    cfg.OUTPUT_DIR = '/data2/qiweili/logs/cat'
    cfg.MODEL.WEIGHTS = os.path.join("/data2/qiweili/logs", "model_final_280758.pkl") 
    #cfg.MODEL.BACKBONE.FREEZE_AT = 6
    cfg.SOLVER.MAX_ITER = 2000
    cfg.SOLVER.STEPS = (1500, 1800)
    trainer = Trainer(cfg)
    model  =  trainer.build_model(cfg)
    val_dataloader = build_detection_test_loader(cfg, mapper=mapper,dataset_name=cfg.DATASETS.TEST)
    trainer.register_hooks([EvalHook(30,lambda:trainer.mytest(cfg,trainer.model,val_dataloader))])
    trainer.resume_or_load(resume=False)
    trainer.train()
