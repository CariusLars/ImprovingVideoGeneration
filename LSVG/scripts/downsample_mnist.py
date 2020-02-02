import numpy as np
import torchvision


data = np.load('/Users/michaelgentnermac/Documents/ADL4CV/mnist_test_seq.npy')

Resize = torchvision.transforms.Resize(28)
ToPil = torchvision.transforms.ToPILImage()
ToTensor = torchvision.transforms.ToTensor()

seq_len = data.shape[0]
seqs = data.shape[1]

res = []

for i in range(seq_len):
    seq = []
    for j in range(seqs):

        img = ToPil(data[i, j, :, :])
        img = Resize(img)
        seq.append(ToTensor(img).numpy().squeeze())
    res.append(seq)
    print(i)

d_data = np.array(res)

np.save('./mnist_test_seq_28.npy', d_data)




