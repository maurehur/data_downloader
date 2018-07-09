import os
import urllib.request
import sys
import zipfile
import pandas as pd
import shutil
import numpy as np
from PIL import Image

TRAINING_SET_URL = "http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip"
TEST_SET_URL = "http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_Images.zip"
TEST_CSV_URL = "http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_GT.zip" 


def extract_training(filename):
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
    for file in zip_ref.namelist():
        if file.endswith(".csv"):
            structure = file.rsplit("/", 2)
            
            basename_dir = structure[0]
            dir_class = structure[1]
            filename_csv = structure[2]
            
            filein_csv = os.path.join(extract_path, basename_dir, dir_class, filename_csv)
            df = pd.read_csv(filein_csv, delimiter=';')
            df.columns = map(str.lower, df.columns)
            dataset = os.path.join(real_path, 'german-traffic-signs', 'training', dir_class)
            if not os.path.exists(dataset):
                os.makedirs(dataset)
            filename = df.filename.apply(lambda x: shutil.copy(os.path.join(extract_path, basename_dir, dir_class, x), dataset))
            df = pd.concat([feature, df])
            df['filename'] = df.filename.apply(lambda x: os.path.join(real_path, 'german-traffic-signs', 'training', dir_class, x))
            
            
            feature = df
    df.rename(columns={'classid': 'class_id'}, inplace=True)
    df['image'] = df.filename.apply(read_images)
    df = df.sample(frac=1)
    return df.to_pickle(os.path.join(real_path, 'german-traffic-signs', 'training-data.csv'))


def extract_test(filename):
    """
    This function create a dataframe from test-dataset
    
    Args:
        filename (str): name of dataset
    
    return:
        test-dataframe
    """
    # filename = filename+'.zip'
    real_path = os.path.realpath('data')
    extract_path = os.path.join(real_path, 'unzip_folder')
    
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)
    print('- Extracting {}'.format(filename))
    
    with zipfile.ZipFile(os.path.join(real_path, filename), "r") as zip_ref:
        zip_ref.extractall(extract_path)
        feature = pd.DataFrame()
        
    print('- Moving Files and built csv')
    for file in zip_ref.namelist():
        if file.endswith(".csv"):
            structure = file.rsplit("/", 1)
            
            basename_dir = structure[0]
            filename_csv = structure[1]
            
            filein_csv = os.path.join(extract_path, basename_dir, filename_csv)
            df = pd.read_csv(filein_csv, delimiter=';')
            df.columns = map(str.lower, df.columns)
            dataset = os.path.join(real_path, 'german-traffic-signs', 'test')
            if not os.path.exists(dataset):
                os.makedirs(dataset)
                
            filename = df.filename.apply(lambda x: shutil.copy(os.path.join(extract_path, basename_dir, x), dataset))
    return None

def extract_csv(filename):
    """
    This function create a dataframe from test-dataset
    
    Args:
        filename (str): name of dataset
    
    return:
        test-dataframe
    """
    # filename = filename+'.zip'
    real_path = os.path.realpath('data')
    extract_path = os.path.join(real_path, 'unzip_folder')
    
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)
    print('- Extracting {}'.format(filename))
    
    with zipfile.ZipFile(os.path.join(real_path, filename), "r") as zip_ref:
        zip_ref.extractall(extract_path)
        feature = pd.DataFrame()
        
    print('- Moving Files and built csv')
    for file in zip_ref.namelist():
        if file.endswith(".csv"):
            structure = file.rsplit("/", 1)
            
            filename_csv = structure[0]
            
            filein_csv = os.path.join(extract_path, filename_csv)
            df = pd.read_csv(filein_csv, delimiter=';')
            df.columns = map(str.lower, df.columns)
            dataset = os.path.join(real_path, 'german-traffic-signs')
            df['filename'] = df.filename.apply(lambda x: os.path.join(real_path, 'german-traffic-signs', 'test', x))
    
    df.rename(columns={'classid': 'class_id'}, inplace=True)
    df['image'] = df.filename.apply(read_images)
    return df.to_pickle(os.path.join(real_path, 'german-traffic-signs', 'test-data.csv'))


def remove_files(filename, rm):
    path = os.path.realpath('data')
    shutil.rmtree(os.path.join(path, 'unzip_folder'))
    if rm:
        os.remove(os.path.realpath(os.path.join(path, filename)))
        return print('- Removing {} and {}\n'.format(filename, 'unzip_folder'))
    else:
        return print('- Removing {}\n'.format('unzip_folder'))

def read_images(filename):
    try:
        img = np.array(Image.open(filename))
        return img
    except Exception as inst:
        print(inst)

