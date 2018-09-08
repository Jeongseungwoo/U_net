import cv2
import os 
from Model import Model
from config import *

def test() :
    if os.path.exists(outputs_dir) is False :
        os.mkdir(outputs_dir)
    model = Model()
    model.load(os.path.join(model_dir,"U_net"))
    file_list = os.listdir(os.path.join(data_dir,"test"))
    for file in file_list :
        print("Filename : " + file)
        seg = model.test(file)
        cv2.imwrite(os.path.join(outputs_dir,file),seg)
        print("Done.")
        
if __name__ == "__main__" :
    test()
    print("Segmentation complete!")