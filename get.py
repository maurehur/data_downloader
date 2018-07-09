from data_downloader.data import DataDownloader


def dado(dataset, rm=False):
    return DataDownloader(dataset, rm)