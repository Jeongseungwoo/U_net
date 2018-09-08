import numpy as np
import cv2
import os, random
from Model import Model
from preprocess import preprocess
from config import *

def train() :
    
    # make directory 
    if os.path.exists(log_dir) is False :
        os.mkdir(log_dir)
    if os.path.exists(model_dir) is False :
        os.mkdir(model_dir)
    if os.path.exists(os.path.join(data_dir,"test")) is False :
        os.mkdir(os.path.join(data_dir,"test"))
    
    # preprocessing datasets (Construct datasets : augmentation & resize)
    print("Construct datasets")
    input_datasets,label_datasets = preprocess()
    inputs = np.load(input_datasets)["img"]
    labels = np.load(label_datasets)["img"]
    print("Done!")
    
    n_sample = inputs.shape[0]
    
    model = Model()
    
    print("Training...")
    for epoch in range(n_epochs) :
        start_time = time.time()
        
        # random sampling 
        shuffle_idx = np.arange(n_samle)
        random.shuffle(shuffle_idx)
        inputs = np.array([inputs[idx] for idx in shuffle_idx])
        labels = np.array([labels[idx] for idx in shuffle_idx])
        
        for i in range(n_samples // mini_batch_size):
            num_iterations = n_samples // mini_batch_size * epoch + i
            start = i * mini_batch_size
            end = (i + 1) * mini_batch_size
            
            U_net_loss = model.train(inputs[start:end],labels[start:end])
        model.save(model_dir,"U_net")
        end_time = time.time()
        epoch_time = end_time-start_time
        print("Epoch : %d, Loss : %f, Time : %02d:%02d" % (epoch,U_net_loss,(epoch_time % 3600 // 60),(epoch_time % 60 // 1)))
        
if __name__ == "__main__" :
    train()
    print("Finish")
