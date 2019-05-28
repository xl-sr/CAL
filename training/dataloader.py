import numpy as np
import pandas as pd
from PIL import Image
from ipdb import set_trace
import bisect

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import imgaug as ia
import imgaug.augmenters as iaa

### data augmentation

def get_augmentations():
    # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
    # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    # Define our sequence of augmentation steps that will be applied to every image
    # All augmenters with per_channel=0.5 will sample one value _per image_
    # in 50% of all cases. In all other cases they will sample new values
    # _per channel_.
    seq = iaa.Sequential([
            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((0, 5),
                [
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                        iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                        iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                    ]),
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                    iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                    # search either for all edges or for directed edges,
                    # blend the result with the original image using a blobby mask
                    iaa.SimplexNoiseAlpha(iaa.OneOf([
                        iaa.EdgeDetect(alpha=(0.5, 1.0)),
                        iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                    ])),
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                    iaa.OneOf([
                        iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                        iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                    ]),
                    iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                    iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                    # either change the brightness of the whole image (sometimes
                    # per channel) or change the brightness of subareas
                    iaa.OneOf([
                        iaa.Multiply((0.5, 1.5), per_channel=0.5),
                        iaa.FrequencyNoiseAlpha(
                            exponent=(-4, 0),
                            first=iaa.Multiply((0.5, 1.5), per_channel=True),
                            second=iaa.ContrastNormalization((0.5, 2.0))
                        )
                    ]),
                    iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                    sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                ],
                random_order=True
            )
        ],
        random_order=True
    )
    return seq

### data transforms
class Resize(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, inp_size):
        if isinstance(inp_size, int):
            inp_size = (inp_size, inp_size)
        self.inp_size = inp_size

    def __call__(self, sample):
        sample['image'] = transforms.Resize((self.inp_size))(sample['image'])
        return sample

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, scalar):
        self.scalar = scalar

    def __call__(self, im):
        w, h = [int(s*self.scalar) for s in im.size]
        return transforms.Resize((h, w))(im)

class Crop(object):
    def __init__(self, box):
        assert len(box) == 4
        self.box = box

    def __call__(self, im):
        return im.crop(self.box)

class Augment(object):
    """
    add data augmentation, takes PIL image and returns numpy array
    """
    def __init__(self, seq):
        self.seq = seq

    def __call__(self, im):
        return Image.fromarray(self.seq.augment_images([np.array(im)])[0])

class ToTensor(object):
    def __call__(self, im):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return normalize(transforms.functional.to_tensor(im))

def get_data_transforms(t='train'):
    data_transforms = {
        'train': transforms.Compose([
            Augment(get_augmentations()),
            Crop((0,120,800,480)),
            Rescale(0.4),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            Crop((0,120,800,480)),
            Rescale(0.4),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms[t]

### helper functions

def onehot(x, n_values='auto'):
    return np.array(OneHotEncoder(n_values=n_values).fit_transform(x.reshape(-1,1)).todense())

def onehot_modified(vals, possible_vals):
    if not isinstance(possible_vals, list): raise TypeError("provide possible_vals as a list")
    enc_vals = np.zeros([len(vals), len(possible_vals)])
    for i, value in enumerate(vals):
        if isinstance(possible_vals[0], float):
            enc = np.where(abs(possible_vals-value)<1e-3)
        else:
            enc = np.where(possible_vals==value)
        enc_vals[i,enc] = 1
    return enc_vals

def roll(x, n):
    return torch.cat((x[-n:], x[:-n]))

### dataset
LABEL_KEYS = ['red_light', 'hazard_stop', 'speed_sign',
              'relative_angle', 'center_distance', 'veh_distance']

def collate_seq(batch):
    inp_batch = [item[0] for item in batch]
    label_batch = [item[1] for item in batch]
    return [inp_batch, label_batch]

    inputs, labels = {}, {}

    inputs['sequence'] = torch.cat([item[0]['sequence'] for item in batch])
    inputs['direction'] = torch.cat([item[0]['sequence'] for item in batch])

    for k in LABEL_KEYS:
        labels[k] = torch.cat([item[1][k] for item in batch])

    return [inputs, labels]

class CAL_Dataset(Dataset):
    def __init__(self, root_dir, t, seq_len, subset_len=0):
        assert t in ['train', 'val']
        self.transform = get_data_transforms(t)

        # load im paths and labels
        df_all = pd.read_csv(root_dir + 'annotations.csv')
        is_val = np.load(root_dir + 'is_val.npy')
        df = df_all[is_val] if t=='val' else df_all[~is_val]
        df = df.reset_index(drop=True); del df['Unnamed: 0']
        if subset_len:
            df = df[:subset_len]
        self.total_frames = len(df)

        # inputs
        self.im_paths = root_dir + df['im_name']
        self.direction = df['direction']

        # output
        self.labels = {}
        # transform reg affordances
        reg_keys = ['relative_angle', 'center_distance', 'veh_distance']
        reg_norm = df[reg_keys] / abs(df[reg_keys]).max()
        self.labels.update({k: torch.Tensor(reg_norm[k]) for k in reg_keys})

        # transform cls affordances
        self.labels['red_light'] = torch.Tensor(onehot_modified(np.array(df['red_light']), [False, True]))
        self.labels['hazard_stop'] = torch.Tensor(onehot_modified(np.array(df['hazard_stop']), [False, True]))
        self.labels['speed_sign'] = torch.Tensor(onehot_modified(np.array(df['speed_sign']), [-1, 30, 60, 90]))

        # setup for sequence reading
        self.start_idx = [0] + list(np.squeeze(np.where(np.diff(df.seq_id)) + np.array(1)))
        self.seq_len = seq_len
        test_im = self.transform(Image.open(self.im_paths[0]))
        self.frame_buffer = []

    def __len__(self):
        return self.total_frames

    def __getitem__(self, idx):
        inputs, labels = {}, {}

        idx = idx % self.total_frames
        # we want bisect_right here so that the first frame in a file gets the
        # file, not the previous file
        next_file_idx = bisect.bisect_right(self.start_idx, idx)
        if next_file_idx < len(self.start_idx):
            start = self.start_idx[next_file_idx]
        else:
            start = self.total_frames - self.seq_len

        if start < idx + self.seq_len:
            idx = start - self.seq_len

        frames = []
        for i in range(self.seq_len):
            im = Image.open(self.im_paths[idx+i])
            im = self.transform(im)
            frames.append(im.unsqueeze(0))
        inputs['sequence'] = torch.cat(frames)

        # get label of the last image
        last_idx = idx + self.seq_len
        inputs['direction'] = np.array(self.direction[last_idx])
        for k in self.labels.keys():
            labels[k] = self.labels[k][last_idx]

        # inputs['sequence'] = idx
        return inputs, labels

def get_data(data_path, seq_len, batch_size):
    train_ds = CAL_Dataset(data_path, 'train', seq_len=seq_len)
    val_ds = CAL_Dataset(data_path, 'val', seq_len=seq_len)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=14),
        DataLoader(val_ds, batch_size=batch_size * 2, pin_memory=True, num_workers=14),
    )

def get_mini_data(data_path, seq_len, batch_size=32, l=4000):
    train_ds = CAL_Dataset(data_path, 'train', seq_len=seq_len, subset_len=l)
    val_ds = CAL_Dataset(data_path, 'train', seq_len=seq_len, subset_len=l)
    return (
        DataLoader(train_ds, batch_size=batch_size, num_workers=10),
        DataLoader(val_ds, batch_size=batch_size, num_workers=10)
    )
