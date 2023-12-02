from time import time
import multiprocessing as mp
import os
import numpy as np
import torch
from torchvision import datasets
from tqdm import tqdm


data_path = "/home/sk138/data/cifar-100-python-segmented/cifar-196-64/"
is_train = False

def segmented_loader(file_path):
    loaded = np.load(file_path)
    return torch.from_numpy(loaded['data'])

root = os.path.join(data_path, 'train' if is_train else 'test')
dataset = datasets.DatasetFolder(root, loader=segmented_loader, extensions='npz')

print(mp.cpu_count())
for num_workers in range(2, mp.cpu_count(), 2):  
    train_loader = torch.utils.data.DataLoader(dataset, shuffle=True, num_workers=num_workers, batch_size=256, pin_memory=True)
    start = time()
    for i, data in tqdm(enumerate(train_loader, 0)):
        pass
    end = time()
    print("Finish with:{} second, num_workers={}".format(end - start, num_workers))


'''
Previous database (using test set)
64
Finish with:96.0620448589325 second, num_workers=10
Finish with:92.5387864112854 second, num_workers=12 --- best
Finish with:95.98159790039062 second, num_workers=14
Finish with:96.43429636955261 second, num_workers=16
Finish with:95.92833399772644 second, num_workers=18
Finish with:101.1809720993042 second, num_workers=20
Finish with:99.13950085639954 second, num_workers=22
Finish with:98.59757828712463 second, num_workers=24
Finish with:98.05948281288147 second, num_workers=26
Finish with:97.34843945503235 second, num_workers=28


New database (using test set)
Finish with:51.086848735809326 second, num_workers=10
Finish with:48.959205865859985 second, num_workers=12 --- best
Finish with:51.03466200828552 second, num_workers=14
Finish with:51.015719413757324 second, num_workers=16
Finish with:51.85728430747986 second, num_workers=18
Finish with:53.1455500125885 second, num_workers=20
Finish with:53.41371440887451 second, num_workers=22
Finish with:53.08710956573486 second, num_workers=24
Finish with:52.976211071014404 second, num_workers=26
Finish with:52.531614542007446 second, num_workers=28
Finish with:52.288734674453735 second, num_workers=30
Finish with:52.01108741760254 second, num_workers=32
Finish with:51.71964383125305 second, num_workers=34
Finish with:54.958446741104126 second, num_workers=36
Finish with:55.915868282318115 second, num_workers=38
Finish with:55.980263233184814 second, num_workers=40
Finish with:58.02855730056763 second, num_workers=42
Finish with:56.0040557384491 second, num_workers=44
Finish with:56.984166622161865 second, num_workers=46
Finish with:56.03009223937988 second, num_workers=48
Finish with:56.02543902397156 second, num_workers=50
Finish with:56.14415264129639 second, num_workers=52
Finish with:55.97662401199341 second, num_workers=54
Finish with:56.21009826660156 second, num_workers=56
Finish with:56.1126389503479 second, num_workers=58
Finish with:56.17449474334717 second, num_workers=60
Finish with:56.14838433265686 second, num_workers=62

'''