from torch.utils.data import DataLoader
#from dataset.dataloader import DeformDatasetSampler
#from dataset.deform_dataset import DeformDataset
import time
from dataloader import DeformDatasetSampler
from deform_dataset import DeformDataset

dataset = DeformDataset('/root/futuregan/data/DEFORM_depth', 12)
sampler = DeformDatasetSampler(dataset, None, 12)
loader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=7, pin_memory=True)
tic = time.time()
for batch in loader:
    toc = time.time()
    print(batch.shape)
    print("Time: {}".format(toc-tic))
    tic = time.time()
    time.sleep(0.5)

