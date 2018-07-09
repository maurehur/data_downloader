import os
import urllib.request
import sys
import zipfile
import pandas as pd
import shutil
import gzip
import numpy as np
from scipy.misc import imsave
from PIL import Image

TRAIN_URL = "https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip"

def extract_training_dataset(filename):
    """
    This function create a dataframe from train-dataset
    
    Args:
        filename (str): name of dataset
    
    return:
        test-dataframe
    """
    # filename = filename
    real_path = os.path.realpath('data')
    extract_path = os.path.join(real_path, 'unzip_folder')
    
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)
    print('- Extracting {}'.format(filename))
    
    with zipfile.ZipFile(os.path.join(real_path, filename), "r") as zip_ref:
        zip_ref.extractall(extract_path)
    
    feature = pd.DataFrame()
    print('- Moving Files and built csv')
    dataset = os.path.join(real_path, 'udacity-dataset')
    if not os.path.exists(dataset):
        os.makedirs(dataset)
    
    df = pd.read_csv(os.path.join(extract_path, 'data', 'driving_log.csv'), delimiter=',')
    data = get_dataframe(df)
    data["filename"] = data["filename"].str.replace('IMG/', os.path.join(dataset, 'IMG/')).str.strip()
    data.to_pickle(os.path.join(extract_path, 'data', 'data.csv'))
    if not os.path.exists(os.path.join(dataset, 'IMG')):
        shutil.move(os.path.join(extract_path, 'data', 'IMG'), dataset)
        shutil.move(os.path.join(extract_path, 'data', 'data.csv'), dataset)
    return None

def get_dataframe(df):
    
    df_L = df.copy()
    df_C = df.copy()
    df_R = df.copy()

    df_L["camera"] = 0
    df_C["camera"] = 1
    df_R["camera"] = 2

    df_L["filename"] = df_L["left"]
    df_C["filename"] = df_C["center"]
    df_R["filename"] = df_R["right"]

    df_L = df_L.drop(["left", "center", "right"], axis=1)
    df_C = df_C.drop(["left", "center", "right"], axis=1)
    df_R = df_R.drop(["left", "center", "right"], axis=1)

    df = pd.concat([df_L, df_C, df_R])

    return df