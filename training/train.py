import numpy as np
import torch
import torch.nn.functional as F
import os, copy, time
from tqdm import tqdm
from ipdb import set_trace

### helper functions

def to_np(t):
    return np.array(t.cpu())

### Losses

def calc_class_weight(x, fac=2):
    """calculate inverse normalized count, multiply by given factor"""
    _, counts = np.unique(x, return_counts=True)
    tmp = 1/counts/sum(counts)
    tmp /= max(tmp)
    return tmp*fac

def get_class_weights():
    # class weights, calculated on the training set
    df_all = pd.read_csv(data_path + 'annotations.csv')
    return {
       'red_light': torch.Tensor(calc_class_weight(df_all['red_light'])),
       'hazard_stop': torch.Tensor(calc_class_weight(df_all['hazard_stop'])),
       'speed_sign': torch.Tensor(calc_class_weight(df_all['speed_sign'])),
       'relative_angle': torch.Tensor([1]),
       'center_distance': torch.Tensor([1]),
       'veh_distance': torch.Tensor([1]),
    }

WEIGHTS = {
    'red_light': torch.Tensor([0.1109, 10.0]),
    'hazard_stop': torch.Tensor([0.0266, 2.0]),
    'speed_sign': torch.Tensor([0.0203, 0.8945, 1.8224, 2.0]),
    'relative_angle': torch.Tensor([1]),
    'center_distance': torch.Tensor([1]),
    'veh_distance': torch.Tensor([1]),
}

def WCE(x, y, w):
    """weighted mean average"""
    t = F.cross_entropy(x, torch.argmax(y, dim=1), weight=w)
    return t

def MAE(x, y, w):
    return F.l1_loss(x.squeeze(), y)*w

def custom_loss(y_pred, y_true, dev='cuda'):
    loss = torch.Tensor([0]).to(dev)
    for k in y_pred:
        func = MAE if y_pred[k].shape[1]==1 else WCE
        loss += func(y_pred[k], y_true[k], WEIGHTS[k].to(dev))

    return loss

def loss_batch(model, loss_func, preds, labels, opt=None):
    loss = loss_func(preds, labels)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item()

### training wrappers

def train(model, data_loader, loss_func, opt):
    device = next(model.parameters()).device
    for inputs, labels in tqdm(data_loader):
        inputs['sequence'] = inputs['sequence'].to(device)
        preds = model(inputs)
        labels = {k: v.to(device) for k,v in labels.items()}
        loss_batch(model, loss_func, preds=preds, labels=labels, opt=opt)
    return model

def validate(model, data_loader, loss_func):
    device = next(model.parameters()).device
    all_preds, all_labels = {}, {}

    with torch.no_grad():
        val_loss = 0
        for inputs, labels in tqdm(data_loader):
            inputs['sequence'] = inputs['sequence'].to(device)
            preds = model(inputs)
            labels = {k: v.to(device) for k,v in labels.items()}
            l = loss_batch(model, loss_func, preds=preds, labels=labels)
            val_loss += l/data_loader.batch_size

            # concatenate for inspection
            if not all_preds:
                all_preds = {k: to_np(v) for k,v in preds.items()}
                all_labels = {k: to_np(v) for k,v in labels.items()}
            else:
                all_preds = {k: np.concatenate([all_preds[k], to_np(v)]) for k,v in preds.items()}
                all_labels = {k: np.concatenate([all_labels[k], to_np(v)]) for k,v in labels.items()}

        val_loss /= len(data_loader)
        print(val_loss)

    return val_loss, all_preds, all_labels

def fit(epochs, model, loss_func, opt, train_dl, valid_dl, dev='cuda', val_hist=None):
    since = time.time()

    val_hist = [] if val_hist is None else val_hist
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = np.inf if not val_hist else min(val_hist)

    for epoch in range(epochs):
        model.train()
        model = train(model, train_dl, loss_func, opt)
        model.eval()
        val_loss, _, _ = validate(model, valid_dl, loss_func)
        val_hist.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            PATH = f"./models/{model.params.name}.pth"
            torch.save(model.state_dict(), PATH)
            best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_val_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_hist
