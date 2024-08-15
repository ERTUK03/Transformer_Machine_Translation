import requests
import os
import tarfile

def download_dataset(url, path):
    if os.path.isfile(path):
        print('File already exists. Skipping download.')
    else:
        response = requests.get(url)
        with open(f'{path}.tgz', 'wb') as file:
            file.write(response.content)

    if os.path.isdir(path):
        print('Directory already exists. Skipping extraction.')
    else:
        with tarfile.open(f'{path}.tgz', 'r') as tar_ref:
            tar_ref.extractall(path)
