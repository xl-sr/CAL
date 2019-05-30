import numpy as np
import os, sys
from PIL import Image
import json
import torch
from torchvision import transforms
from .net import get_model

BASE_PATH = os.path.abspath(os.path.join('.', '.'))
MODEL_PATH = BASE_PATH + "/training/models/"

# classes of the categorical affordances
CAT_DICT = {
    'red_light': [False, True],
    'hazard_stop': [False, True],
    'speed_sign': [-1, 30, 60, 90],
}

# normalizing constants of the continuous affordances
REG_DICT = {
    'center_distance': 1.6511945645500001,
    'veh_distance': 50.0,
    'relative_angle': 0.759452569632
}

### helper functions

def load_json(path):
    with open(path + '.json', 'r') as json_file:
        f = json.load(json_file)
    return f

def to_np(t):
    return np.array(t.cpu())

def softmax(x):
    return np.exp(x)/sum(np.exp(x))

### data transforms

class Rescale(object):
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

def get_transform():
    return transforms.Compose([
        Crop((0, 120, 800, 480)),
        Rescale(0.4),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

### network

class CAL_network(object):
    def __init__(self, name='gru'):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._transform = get_transform()

        # get the model
        params = load_json(MODEL_PATH + name)
        self.model, _ = get_model(params)
        self.model.load_state_dict(torch.load(MODEL_PATH + f"{name}.pth"))
        self.model.eval().to(self.device);

    def predict(self, sequence, direction):
        inputs = {
            'sequence': torch.cat(sequence).to(self.device),
            'direction': np.array(direction),
        }
        preds = self.model(inputs)
        preds = {k: to_np(v) for k,v in preds.items()}

        out = {}
        out.update({k: self.cat_process(k, preds[k]) for k in CAT_DICT})
        out.update({k: self.reg_process(k, preds[k]) for k in REG_DICT})
        return out

    def preprocess(self, arr):
        im = self._transform(Image.open(arr))
        return im.unsqueeze(0)

    @staticmethod
    def cat_process(cl, arr):
        arr = softmax(arr)
        max_idx = np.argmax(arr)
        pred_class = CAT_DICT[cl][max_idx]
        pred_prob = np.max(arr)
        return (pred_class, pred_prob)

    @staticmethod
    def reg_process(cl, arr):
        arr = np.clip(arr, -1, 1)
        return arr*REG_DICT[cl]
