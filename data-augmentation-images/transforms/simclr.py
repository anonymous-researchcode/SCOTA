import torchvision
from torchvision.transforms import Compose
import random
from transforms.rand_augment import augmentations
import numpy as np

class CompositeTransform:

    def __init__(self, transform_names=[], ratios=[], pre_transforms=[], post_transforms=[], 
                 probs=None, tree_idxes=None, **kwargs):
        '''
        A generic class for simclr data augmentation that takes in a list of transform names and ratios
        '''
        self.pre_transforms = pre_transforms
        self.post_transforms = post_transforms
        self.transforms = []
        ''' Implement all other augmentations '''
        ops = [augmentations[name] for name in transform_names]
        for i, (op, minval, maxval) in enumerate(ops):
            assert ratios[i] in [0.2, 0.4, 0.6, 0.8, 1.0]
            tmp_min, tmp_max = minval + (ratios[i]-0.2)*(maxval - minval), minval + (ratios[i])*(maxval - minval)
            self.transforms.append(op(minval=tmp_min, maxval=tmp_max))
        # self.transforms.append(torchvision.transforms.ToTensor())

        self.probs = probs

        self.max_idx = np.max(tree_idxes)+1 if tree_idxes is not None else None
        self.tree_idxes = tree_idxes
        self.idx_to_transform = self.generate_idx_to_transforms(tree_idxes)
    
    def generate_idx_to_transforms(self, tree_idxes):
        if tree_idxes is None:
            return None
        transformed_idxes = np.ones(self.max_idx)*-1
        for i, idx in enumerate(tree_idxes):
            transformed_idxes[idx] = i
        return transformed_idxes.astype(int)
    
    def add_pre_and_post(self, selected_transforms):
        final_transforms = self.pre_transforms[:]
        final_transforms += selected_transforms
        final_transforms.append(torchvision.transforms.ToTensor())
        final_transforms += self.post_transforms
        # print(final_transforms)
        return Compose(final_transforms)

    def apply_transform(self, data):
        if self.probs is None:
            return self.add_pre_and_post(self.transforms)(data)
        elif self.tree_idxes is None:
            tem_transform = []
            for prob, transform in zip(self.probs, self.transforms):
                if random.random() < prob:
                    tem_transform.append(transform)
                    # data = self.transform(data)
            return self.add_pre_and_post(tem_transform)(data)
        else: 
            cur_idx = 0
            # appy transforms based on the tree
            tem_transform = []
            while cur_idx < self.max_idx:
                if self.idx_to_transform[cur_idx] == -1:
                    break
                cur_transform = self.transforms[self.idx_to_transform[cur_idx]]
                cur_prob = self.probs[self.idx_to_transform[cur_idx]]
                if random.random() < cur_prob:
                    tem_transform.append(cur_transform)
                    # data = cur_transform(data)
                # print(cur_idx, cur_transform, cur_prob)
                if cur_idx*2+1 >= self.max_idx or self.idx_to_transform[cur_idx*2+1] == -1:
                    break
                next_prob = self.probs[self.idx_to_transform[cur_idx*2+1]]
                if random.random() < next_prob:
                    cur_idx = cur_idx*2 + 1
                else:
                    cur_idx = cur_idx*2 + 2
            return self.add_pre_and_post(tem_transform)(data)
    
    def __call__(self, data):
        return self.apply_transform(data)

class SimCLRTransfrom:

    def __init__(self, transform_names=[], ratios=[], pre_transforms=[], post_transforms=[], 
                 probs=None, tree_idxes=None, **kwargs):
        '''
        A generic class for simclr data augmentation that takes in a list of transform names and ratios
        '''
        self.pre_transforms = pre_transforms
        self.post_transforms = post_transforms
        self.transforms = []
        ''' Implement all other augmentations '''
        ops = [augmentations[name] for name in transform_names]
        for i, (op, minval, maxval) in enumerate(ops):
            # assert ratios[i] in [0.2, 0.4, 0.6, 0.8, 1.0]
            tmp_min, tmp_max = minval + (ratios[i]-0.2)*(maxval - minval), minval + (ratios[i])*(maxval - minval)
            self.transforms.append(op(minval=tmp_min, maxval=tmp_max))
        # self.transforms.append(torchvision.transforms.ToTensor())

        self.probs = probs

        self.max_idx = np.max(tree_idxes)+1 if tree_idxes is not None else None
        self.tree_idxes = tree_idxes
        self.idx_to_transform = self.generate_idx_to_transforms(tree_idxes)
    
    def generate_idx_to_transforms(self, tree_idxes):
        if tree_idxes is None:
            return None
        transformed_idxes = np.ones(self.max_idx)*-1
        for i, idx in enumerate(tree_idxes):
            transformed_idxes[idx] = i
        return transformed_idxes.astype(int)
    
    def add_pre_and_post(self, selected_transforms):
        final_transforms = self.pre_transforms[:]
        final_transforms += selected_transforms
        final_transforms.append(torchvision.transforms.ToTensor())
        final_transforms += self.post_transforms
        # print(final_transforms)
        return Compose(final_transforms)

    def apply_transform(self, data):
        if self.probs is None:
            return self.add_pre_and_post(self.transforms)(data)
        elif self.tree_idxes is None:
            tem_transform = []
            for prob, transform in zip(self.probs, self.transforms):
                if random.random() < prob:
                    tem_transform.append(transform)
                    # data = self.transform(data)
            return self.add_pre_and_post(tem_transform)(data)
        else: 
            cur_idx = 0
            # appy transforms based on the tree
            tem_transform = []
            while cur_idx < self.max_idx:
                if self.idx_to_transform[cur_idx] == -1:
                    break
                cur_transform = self.transforms[self.idx_to_transform[cur_idx]]
                cur_prob = self.probs[self.idx_to_transform[cur_idx]]
                if random.random() < cur_prob:
                    tem_transform.append(cur_transform)
                    # data = cur_transform(data)
                # print(cur_idx, cur_transform, cur_prob)
                if cur_idx*2+1 >= self.max_idx or self.idx_to_transform[cur_idx*2+1] == -1:
                    break
                next_prob = self.probs[self.idx_to_transform[cur_idx*2+1]]
                if random.random() < next_prob:
                    cur_idx = cur_idx*2 + 1
                else:
                    cur_idx = cur_idx*2 + 2
            return self.add_pre_and_post(tem_transform)(data)
    
    def __call__(self, data):
        return self.apply_transform(data), self.apply_transform(data)

class SimCLR:
    """
    A stochastic data augmentation module that transforms any given data example randomly
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(self, pre_transforms=[], post_transforms=[], **kwargs):
        s = 1
        color_jitter = torchvision.transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        self.train_transform = torchvision.transforms.Compose(
            pre_transforms +
            [
                torchvision.transforms.RandomResizedCrop(size=224),
                torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
                torchvision.transforms.RandomApply([color_jitter], p=0.8),
                torchvision.transforms.RandomGrayscale(p=0.2),
                torchvision.transforms.ToTensor(),
            ]
            + post_transforms
        )

        self.test_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size=224),
                torchvision.transforms.ToTensor(),
            ]   
        )

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)