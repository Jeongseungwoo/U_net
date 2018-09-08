import numpy as np
import cv2
import os
from config import *

def preprocess() :
    train_dir = os.path.join(data_dir,"train")
    dir_path = []
    for directory in os.listdir(train_dir) :
        if os.path.exists(os.path.join(data_dir,directory+".npz")) is False :
            path = os.path.join(train_dir,directory)
            img_list = os.listdir(path)
            data = []
            for img in img_list :
                img = cv2.imread(os.path.join(path,img),1)
                
                # Data augmentation & Resize 추가 !! 

                data.append(img)
            np.savez(os.path.join(data_dir,directory+".npz"),img = data)
        dir_path.append(os.path.join(data_dir,directory+".npz"))
    return dir_path

if __name__ == "__main__" :
    print("Preprocessing")
    preprocess()