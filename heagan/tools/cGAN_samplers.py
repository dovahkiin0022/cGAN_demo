import os
from joblib import load
import numpy as np
from importlib import resources

with resources.files('heagan.saved_cGAN').joinpath('scale_kde_pipe.joblib') as fname:
    scale_sampler = load(fname)

def prop_sampler(n_samples,prop_dim, default = 'normal'):
  if default == 'kde':
    return scale_sampler['KDE'].sample(n_samples).astype('float32')
  elif default == 'normal':
    return np.random.normal(size=[n_samples,prop_dim]).astype('float32')
  
def noise_sampler(N, z_dim):
    return np.random.normal(size=[N, z_dim]).astype('float32')