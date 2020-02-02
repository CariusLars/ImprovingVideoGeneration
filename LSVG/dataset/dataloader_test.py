from torch.utils.data import DataLoader
from dataset.moving_mnist import MovingMnist
from dataset.moving_mnist_sampler import MnistSampler
import matplotlib.pyplot as plt
import numpy as np
import time
#np.random.seed(42)

data = MovingMnist('/Users/michaelgentnermac/Documents/ADL4CV')
sampler = MnistSampler(data)

loader = DataLoader(data, 100, sampler=sampler)

iterator = iter(loader)

batch0 = next(iterator)

img = batch0['img']
#seq = batch0['seq']

plt.imshow(img[2, :, :].squeeze())
#plt.imshow(seq[1, 10, :, :])
plt.show()
tic = 0
toc = 0
diff = 0
for batch in loader:

    toc = time.time()
    #print(batch['img'].shape)
    #print(batch['seq'].shape)
    diff = toc-tic
    tic = time.time()
    #time.sleep(1)

