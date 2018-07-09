import os
import urllib.request
import sys
import zipfile
import pandas as pd
import shutil
import numpy as np


def print_progress(count, blocksize, total):
    # Percentage completion.
    pct_complete = min(int(float((count * blocksize *100 ) / total)), 100)
    # Status-message.
    # Note the \r which means the line should overwrite itself.
    msg = "\r- Progress: {}%".format(pct_complete)

    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()
    
def download_data(url, filename, expected_bytes):
    """
    This function download all the files of a dataset
    
    Args:
        url (str): network object
        filename (str): name of dataset
        expected_bytes (int): size of the url-file

    """
    
    filename = filename
    try:
        if not os.path.exists('data'):
            os.makedirs('data')
        real_path = os.path.realpath('data')
        
        if not os.path.exists(os.path.join(real_path, filename)):
            print("- Opening url: {}".format(url))
            download, _ = urllib.request.urlretrieve(url, os.path.join(real_path, filename), reporthook=print_progress)
            print('\n- Download Complete!')

        statinfo = os.stat(os.path.join(real_path, filename))
        if statinfo.st_size != expected_bytes:
            print("- Opening url: {}".format(url))
            download, _ = urllib.request.urlretrieve(url, os.path.join(real_path, filename), reporthook=print_progress)
            print('\n- Download Complete!')
            
        elif statinfo.st_size == expected_bytes:
            print('- Found and verified: {}'.format(filename))
        
    except Exception as e:
        print(str(e))
        
        
def get_data(url, filename):
    """
    This function download all the files of a dataset
    
    Args:
        url (str): network object
        filename (str): name of dataset
    
    return:
        Data from a url

    """
    open_url = urllib.request.urlopen(url)
    total = int(open_url.getheader("Content-Length"))
    
    return download_data(url, filename, total)
    
