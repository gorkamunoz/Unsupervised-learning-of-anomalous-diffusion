import numpy as np    
import torch
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.loss import _Loss

def normalize(trajs):
    ''' Normalizes trajectories by substracting average and dividing by sqrt of msd
    Arguments:
	- traj: trajectory or ensemble of trajectories to normalize. 
    - dimensions: dimension of the trajectory.
	return: normalized trajectory'''
    if len(trajs.shape) == 1:
        trajs = np.reshape(trajs, (1, trajs.shape[0]))
    trajs = trajs - trajs.mean(axis=1, keepdims=True)
    displacements = (trajs[:,1:] - trajs[:,:-1]).copy()    
    variance = np.std(displacements, axis=1)
    variance[variance == 0] = 1
    new_trajs = np.cumsum((displacements.transpose()/variance).transpose(), axis = 1)
    return np.concatenate((np.zeros((new_trajs.shape[0], 1)), new_trajs), axis = 1)

def prepare_dataset(dataset, return_labels=False, n_labels=2, norm=True, shuffle=True, lw=False, to_tensor=True):
    """ Normalizes trajectories from dataset, shuffles them and converts them from np arrays to torch tensors """
    if shuffle: np.random.shuffle(dataset)
    labels = dataset[:,:n_labels]
    dataset = dataset[:,n_labels:]
    if norm: dataset = normalize(dataset)
    if lw: dataset, labels = (dataset[(np.prod((abs(dataset[:,1:])<10e10), axis=1)!=0)], labels[(np.prod((abs(dataset[:,1:])<10e10), axis=1)!=0)])
    if to_tensor: dataset = torch.from_numpy(dataset).reshape(dataset.shape[0],1,dataset.shape[1]).float()
    if return_labels: return dataset, labels
    return dataset

def do_inference(dataset, model, decode=True):
    """ Does inference with given a data set and an autoencoder model 
    return: numpy.array, inferred data set"""
    if isinstance(dataset, np.ndarray):
        raise ValueError('You must convert the data set to a tensor before running inference. Check prepare_dataset')
    model.decode = decode
    return model(dataset).detach().numpy().reshape(dataset.shape[0],-1)

def do_PCA(dataset, model, decode=False, n_components=2):
    """ Does PCA on a data set after inference with an autoencoder model
    return: numpy.array
    """
    inference = do_inference(dataset=dataset, model=model, decode=decode)
    return PCA(n_components=n_components).fit_transform(inference)

def do_TSNE(dataset, model, decode=False, n_components=2):
    """ Does TSNE on a data set after inference with an autoencoder model
    return: numpy.array
    """
    inference = do_inference(dataset=dataset, model=model, decode=decode)
    return TSNE(n_components=n_components).fit_transform(inference)

def do_UMAP(dataset, model, decode=False, n_components=2):
    """ Does UMAP on a data set after inference with an autoencoder model
    return: numpy.array
    """
    inference = do_inference(dataset=dataset, model=model, decode=decode)
    return umap.UMAP(n_components=n_components).fit_transform(inference)

def mrae(prediction, target):
    """ Computes mean relative absolute error between a predictor tensor and a tensor with the target values
    return: float
    """
    return ((prediction-target).abs_()/(target+1)).mean().item()

def swsMSE(prediction, target, scale=1):
    """ Computes the sample-wise scaled MSE between a predictor tensor and a tensor with the target values 
    return: float
    """
    scaling_factor = scale * target.abs().max(dim=-1).values.unsqueeze(-1)
    return F.mse_loss(torch.div(prediction, scaling_factor), torch.div(target, scaling_factor)).item()

def relative_entropy(tensor_a, tensor_b):
    tensor_a = torch.div(tensor_a,sum(tensor_a))
    tensor_b = torch.div(tensor_b,sum(tensor_b))
    return sum(tensor_a*np.log(torch.div(tensor_a,tensor_b)))

"""def eb_parameter(trajectories, msd_ratio=0.6):"""
    
def get_msd(trajectories, msd_ratio=0.6):
    n = round(trajectories.shape[-1]*msd_ratio)
    return np.array([np.mean((trajectories[:,i:]-trajectories[:,:-i])**2, axis=1) for i in range (1,n)]).T

def get_variance(msd):
    return np.mean(np.mean(msd**2, axis=0) - np.mean(msd, axis=0)**2)

class Dataset_Loader(Dataset):
    """ Data loader class for the supervised model """
    def __init__(self, trainset, transform=None):
        self.data = trainset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
class swsMSELoss(nn.MSELoss):
    def __init__(self, scale=1):
        super(swsMSELoss, self).__init__()
        self.scale = scale
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        scaling_factor = self.scale * target.abs().max(dim=-1).values.unsqueeze(-1)
        return F.mse_loss(torch.div(input, scaling_factor), torch.div(target, scaling_factor), reduction=self.reduction)

class MAPELoss(_Loss):
    def __init__(self, scale=True, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(MAPELoss, self).__init__(size_average, reduce, reduction)
        self.scale = scale

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        scaling_factor = self.scale_tensor(target)
        return torch.div((target-input).abs(),(input.abs()+scaling_factor)).mean()
    
    def scale_tensor(self, target: Tensor):
        if self.scale:
            return target.abs().max(dim=-1).values.unsqueeze(-1)
        return 1