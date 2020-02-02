from torch.utils.data import Dataset
import os
import yaml
from PIL import Image
import numpy as np
import torch, torchvision


class DeformDataset(Dataset):

    def __init__(self, data_root_path, snippet_length, transform=None, channels=3):

        print('Initializing DeformDataset')
        super(DeformDataset, self).__init__()
        self.data_root_path = data_root_path
        self.snippet_length = snippet_length
        self.valid, self.test = self._load_test_and_valid_set()
        #self.resolution = None
        #self.transformation = torchvision.transforms.Resize(4)
        self.transform = transform
        self.channels = channels

        sequences = [directory for directory in os.listdir(data_root_path) if (directory.startswith('seq')
                                                                           and directory not in self.valid
                                                                           and directory not in self.test)]
        sequences.sort()
        #print('Sequences:')
        #print(sequences)
        # Create dictionary containing the dataset structure and check if .jpg and .pgm exist
        self.train_data_dict = {}
        for seq in sequences:

            frames = os.listdir(os.path.join(data_root_path, seq, 'color'))
            frames.sort()
            for frame in frames:

                if frame.startswith('frame'):
                    frame_name = os.path.splitext(frame)[0]
                    frame_path = os.path.join(self.data_root_path, seq, 'color', frame_name+'.jpg')
                    if self.channels == 4:
                        depth_path = os.path.join(self.data_root_path, seq, 'depth', frame_name+'.pgm')
                        assert(os.path.exists(frame_path) and os.path.exists(depth_path))
                    else:
                        assert (os.path.exists(frame_path))

            # Append sequence only if it is complete
            self.train_data_dict[seq] = [os.path.splitext(frm)[0] for frm in frames if frm.startswith('frame')]
            #print(self.train_data_dict)
    def __getitem__(self, idx):

        img_dpts_list = []
        #img_dpts = np.zeros((12, 128, 128, 4))

        for i in range(self.snippet_length):

            img_dpts_list.append(self.load_image_and_depth(idx[0], self.train_data_dict[idx[0]][idx[1]+i]))

        img_dpts = torch.stack(img_dpts_list, dim=0)
        #print(img_dpts)
        #print(type(img_dpts))
            #img_dpts[i, :, :, :] = self.load_image_and_depth(idx[0], self.train_data_dict[idx[0]][idx[1]+i])
        #print("Self.transform: ")
        #print(self.transform)
        #if self.transform is not None:

            # Make pixel range [-1,1]
        #    img_dpts = [self.transform(frame).mul(2).add(-1) for frame in img_dpts]
            # Make depth range [0,1]
        #    img_dpts[:, :, :, 3] = ((img_dpts[:, :, :, 3] + 1) / 10000) - 1
        #    print(type(img_dpts))
        #    print(img_dpts.shape)
        #shape now: [D,C, H, W]
        #return np.moveaxis(np.array(img_dpts),-1,0) #shape: [C,D,H,W] (C: nimg_channels, D: nframes, H: img_h, W: img_w)
        #return img_dpts.permute(3,0,1,2)  # shape: [C,D,H,W] (C: nimg_channels, D: nframes, H: img_h, W: img_w)
        return img_dpts.permute(1,0,2,3)
    def __len__(self):
        return sum([len(seq)-11 for seq in self.train_data_dict.values()])

    def _load_test_and_valid_set(self):

        with open(os.path.join(self.data_root_path, 'valid_test_config.yaml'), 'r') as yamlfile:

            content = yaml.load(yamlfile)
            return content['valid'], content['test']

    def set_resolution(self, width, height):
        self.resolution = (width, height)

    def image_transformation(self, image):
        if self.transform is not None:
            return self.transform(image).mul(2).add(-1)
        else:
            return image.mul(2).add(-1)

    def depth_transformation(self, depth):
        if self.transform is not None:
            return self.transform(depth).mul(0.0002).add(-1)
        else:
            return depth.mul(0.0002).add(-1)

    def load_image_and_depth(self, seq, frame):

        img_path = os.path.join(self.data_root_path, seq, 'color', frame + '.jpg')
        if self.channels == 4:
            depth_path = os.path.join(self.data_root_path, seq, 'depth', frame+ '.pgm')
        img = Image.open(img_path).convert('RGB')
        img_transformed = self.image_transformation(img)
        if self.channels == 4:
            depth = Image.open(depth_path)
            depth_transformed = self.depth_transformation(depth)
            return torch.cat((img_transformed, depth_transformed), 0)
        else:
            return img_transformed
        #return np.concatenate((img_transformed, np.expand_dims(depth_transformed, axis=2)), axis=2)


if __name__ == '__main__':

    dat = DeformDataset('/Users/michaelgentnermac/Documents/ADL4CV/Data/test_set')
    #print(dat['seq00000', 10])
