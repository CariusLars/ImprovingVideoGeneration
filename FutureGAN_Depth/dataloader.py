from torch.utils.data import Sampler
import numpy as np


class DeformDatasetSampler(Sampler):

    def __init__(self, deform_dataset, test_set, snippet_length):
        super(DeformDatasetSampler, self).__init__(deform_dataset)
        self.dataset = deform_dataset
        self.seq_lenghts = {seq: len(self.dataset.train_data_dict[seq])-snippet_length for
                            seq in self.dataset.train_data_dict.keys()}
        self.snippet_length = snippet_length

        assert deform_dataset.snippet_length == self.snippet_length

    def __iter__(self):
        #print("Train data dict:")
        #print(self.dataset.train_data_dict)
        sequences = list(self.dataset.train_data_dict.keys())
        #print("sequences:")
        #print(sequences)
        # sample sequence
        rand_seqs = list(np.random.choice(sequences, size=len(self.dataset)))
        #print("rand seqs:")
        #print(rand_seqs)
        # sample startpoint
        rand_frames = [np.random.randint(0, self.seq_lenghts[seq], size=1) if self.seq_lenghts[seq]>0 else 0 for seq in rand_seqs]
        #print("rand_frames:")
        #print(rand_frames)
        #print("seq_lengths:")
        #print(self.seq_lenghts)

        for i, seq in enumerate(rand_seqs):
            #print(seq, int(rand_frames[i]))
            yield (seq, int(rand_frames[i]))

    def __len__(self):
        return len(self.dataset)

# ToDO write collate_fn