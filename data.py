from data_downloader.utils import get_data
from data_downloader.datasets.mnist import *
from data_downloader.datasets.german_traffic_signs import *
from data_downloader.datasets.udacity_dataset import *


class DataDownloader:
    
    def __init__(self, dataset, rm=False):
        self.dataset = dataset
        self.rm = False
        
        if self.dataset ==  'mnist':
            print('###################################################')
            print('Attempting to download: {}'.format(self.dataset))
            print('###################################################')
            
            feature_train = 'mnist-train.gz'
            label_train = 'mnist-train-label.gz'

            feature_test = 'mnist-test.gz'
            label_test = 'mnist-test-label.gz'

            get_data(TRAIN_FEATURES_URL, feature_train)
            get_data(TRAIN_LABELS_URL, label_train)

            get_data(TEST_FEATURES_URL, feature_test)
            get_data(TEST_LABELS_URL, label_test)

            train_mnist_extract = extract_data(feature_train, 60000)
            train_labels = extract_labels(label_train, 60000)

            test_mnist_extract = extract_data(feature_test, 10000)
            test_labels = extract_labels(label_test, 10000)

            process(train_mnist_extract, train_labels, 'training')
            process(test_mnist_extract, test_labels, 'test')
            remove_files_mnist(rm)

        elif self.dataset == 'german-traffic-signs':
            
            print('###################################################')
            print('Attempting to download: {}'.format(self.dataset+'-dataset'))
            print('###################################################')
            
            filename_train = 'german-traffic-signs-train.zip'
            filename_test = 'german-traffic-signs-test.zip'
            filename_final_test = 'german-final-test.zip'

            get_data(TRAINING_SET_URL, filename_train)
            extract_training(filename_train)
            remove_files(filename_train, rm)

            get_data(TEST_SET_URL, filename_test)
            extract_test(filename_test)
            remove_files(filename_test, rm)
            
            get_data(TEST_CSV_URL, filename_final_test)
            extract_csv(filename_final_test)

        elif self.dataset == 'udacity-dataset':
            
            print('###################################################')
            print('Attempting to download: {}'.format(self.dataset))
            print('###################################################')
            ''
            filename_train = 'udacity-dataset.zip'
            get_data(TRAIN_URL, filename_train)
            extract_training_dataset(filename_train)
            remove_files(filename_train, rm)

