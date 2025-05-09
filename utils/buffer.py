import torch
import numpy as np
from typing import Tuple
from torchvision import transforms

def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    """
    Reservoir sampling algorithm.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :return: the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1

class Buffer:
    """
    The memory buffer
    """
    def __init__(self, buffer_size, device, n_tasks=1, mode='reservoir'):
        assert mode in ['ring', 'reservoir']
        self.buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0
        self.functional_index = eval(mode)
        if mode == 'ring':
            assert n_tasks is not None
            self.task_number = n_tasks
            self.buffer_portion_size = buffer_size // n_tasks
        self.attributes = ['examples', 'labels', 'x_mark', 'y_mark']

    def init_tensors(self, examples: torch.Tensor, labels: torch.Tensor,
                     x_mark: torch.Tensor, y_mark: torch.Tensor) -> None:
        """
        Initializes just the required tensors.
        :param examples: tensor containing the observations
        :param labels: tensor containing the labels
        :param x_mark: tensor containing external features of observations
        :param y_mark: tensor containing external features of labels
        """
        for attr_str in self.attributes:
            attr = eval(attr_str)
            if attr is not None and not hasattr(self, attr_str):
                typ = torch.float32
                setattr(self, attr_str, torch.zeros((self.buffer_size,
                                                     *attr.shape[1:]), dtype=typ, device=self.device))

    def add_data(self, examples, labels=None, x_mark=None, y_mark=None):
        """
        Adds the data to the memory buffer according to the reservoir strategy.
        :param examples: tensor containing the observations
        :param labels: tensor containing the labels
        :param x_mark: tensor containing external features of observations
        :param y_mark: tensor containing external features of labels
        :return:
        """
        if not hasattr(self, 'examples'):
            self.init_tensors(examples, labels, x_mark, y_mark)

        for i in range(examples.shape[0]):
            index = reservoir(self.num_seen_examples, self.buffer_size)
            self.num_seen_examples += 1
            if index >= 0:
                self.examples[index] = examples[i].to(self.device)
                if labels is not None:
                    self.labels[index] = labels[i].to(self.device)
                if x_mark is not None:
                    self.x_mark[index] = x_mark[i].to(self.device)
                if y_mark is not None:
                    self.y_mark[index] = y_mark[i].to(self.device)

    def get_data(self, size: int, transform: transforms=None) -> Tuple:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        if size > min(self.num_seen_examples, self.examples.shape[0]):
            size = min(self.num_seen_examples, self.examples.shape[0])

        choice = np.random.choice(min(self.num_seen_examples, self.examples.shape[0]),
                                  size=size, replace=False)
        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee.cpu())
                                  for ee in self.examples[choice]]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr[choice],)

        return ret_tuple

    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.num_seen_examples == 0:
            return True
        else:
            return False

    def empty(self) -> None:
        """
        Set all the tensors to None.
        """
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                delattr(self, attr_str)
        self.num_seen_examples = 0
