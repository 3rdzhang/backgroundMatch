from transformers import AutoImageProcessor, ViTModel
import torch
from PIL import Image
import  os
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd

class FeatsExtracter():
    def __init__(self):
        self.image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
    
    def extract_feature(self, img_dir, max_len = 10):
        feats = {}
        files = os.listdir(img_dir)
        
        for f in files[:max_len]:
            f_path = os.path.join(img_dir,f)
            image = Image.open(f_path)
            print("extracting feats of {}".format(f_path))
            inputs = self.image_processor(image, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
                last_hidden_states = outputs.last_hidden_state[:,0]
                feats[f] = last_hidden_states
        return feats

    def save_feats(self, feats, feats_file):
        with open(feats_file, 'wb') as f:
            pickle.dump(feats, f)

    def load_feats(self, feats_file):
        with open(feats_file, 'rb') as f:
            feats = pickle.load(f)
            return feats
  

def get_one_feats(img_name, feats_dict):
    return {img_name: feats_dict[img_name]}

def get_group_feats(img_dir, feats_dict):
    files = os.listdir(img_dir)
    feats = { k : feats_dict[k] for k in files}
    return feats

def get_result(ori_feats, back_feats, topn):
    result = {}
    for ori_f in ori_feats.keys():
        dist = {}
        for back_f in back_feats.keys():
            d = torch.nn.functional.cosine_similarity(ori_feats[ori_f], back_feats[back_f])
            dist[back_f] = d
        dist = sorted(dist.items(), key=lambda d:d[1], reverse = True)
        result[ori_f] = dist[:topn]
    return result  

def combine_result(back_result, profile_result):
    for ori in back_result.keys():
        back_result[ori].append(profile_result[ori])
    return back_result

def show_result(back_result, profile_result,  ori_path, back_path, profile_path):
    plt.figure()
    row_num = len(back_result)
    fig, axs = plt.subplots(row_num, 5, figsize=(25, 15))
    i = 1
    for r in back_result.keys():
        image = Image.open(os.path.join(ori_path, r))
        plt.subplot(row_num, 5, i)
        i = i + 1
        plt.imshow(image)
        for rr in back_result[r]:
            image = Image.open(os.path.join(back_path, rr[0]))
            plt.subplot(row_num, 5, i)
            i = i + 1
            plt.imshow(image)
         
        image = Image.open(os.path.join(profile_path, profile_result[r][0][0]))
        plt.subplot(row_num, 5, i)
        i = i + 1
        plt.imshow(image)

