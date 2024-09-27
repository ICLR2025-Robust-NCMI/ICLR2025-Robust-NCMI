import torch
import numpy as np

class ClassBalancedSampler(torch.utils.data.Sampler):
    """
    Balanced sampling from the labeled and unlabeled data.
    """
    def __init__(self, inds, batch_size_per_class, num_batches=None):
        self.inds = np.array(inds)
        self.n_classes = len(inds)

        self.batch_size = batch_size_per_class

        if num_batches is not None:
            self.num_batches = num_batches
        else:
            self.num_batches = int(np.ceil(self.inds.shape[1] / self.batch_size))
        super().__init__(None)

    def __iter__(self):
        for i in range(self.n_classes):
            np.random.shuffle(self.inds[i])
        batch_counter = 0
        while batch_counter < self.num_batches:
            inds_shuffled = torch.randperm(self.inds.shape[1]) # [self.sup_inds[i] for i in torch.randperm(len(self.sup_inds))]
            for k in range(0, len(inds_shuffled), self.batch_size):
                if batch_counter >= self.num_batches:
                    break
                batch = self.inds[:, inds_shuffled[k:min(k + self.batch_size,len(inds_shuffled))]].flatten().tolist()

                np.random.shuffle(batch)
                yield batch
                batch_counter += 1

    def __len__(self):
        return self.num_batches
