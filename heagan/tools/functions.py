import os
import pymatgen.core as mg
import numpy as np
import pandas as pd
import torch

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '../..'))


periodic_df = pd.read_csv(os.path.join(ROOT_DIR,'dataset/periodic_table.csv'))
atomic_number_order = periodic_df['Symbol'].values[:103]

def check_cuda():
  if torch.cuda.is_available():
    cuda = True
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
  else:
    cuda = False
  return cuda

def pymatgen_comp(comp_list):
  if type(comp_list) == list or type(comp_list)== np.ndarray:
    return [mg.Composition(x) for x in comp_list]
  else:
    return mg.Composition(comp_list)
  

def decode(vec, elem_list, thresh=0.00):
    comp = ''
    for i, x in enumerate(vec):
        if x > thresh:
            comp += elem_list[i] + '{:.2f} '.format(x)
    return mg.Composition(comp)

class data_generator(object):
    def __init__(self, comps, use_all_eles = True):

        #with open(csv_file, 'r') as fid:
            #l = fid.readlines()
        #data = [x.strip().split(',')[1] for x in l]
        #data.remove('Composition')

        #remove single elements from dataset, want only HEAs. Also keep unqiue compositions
        if use_all_eles:
            fixed_list = atomic_number_order
        else:
            fixed_list = []

        if len(fixed_list) == 0:
          all_eles = []
          for c in comps:
            all_eles += list(c.get_el_amt_dict().keys())
        else:
          all_eles = fixed_list
        eles = np.array(sorted(list(set(all_eles))))

        self.elements = eles
        self.size = len(eles)
        self.length = len(comps)

        all_vecs = np.zeros([len(comps), len(self.elements)])
        for i, c in enumerate(comps):
            for k, v in c.get_el_amt_dict().items():
                j = np.argwhere(eles == k)
                all_vecs[i, j] = v
        all_vecs = all_vecs / np.sum(all_vecs, axis=1).reshape(-1, 1)
        self.real_data = np.array(all_vecs, dtype=np.float32)

    def sample(self, N):
        idx = np.random.choice(np.arange(self.length), N, replace=False)
        data = self.real_data[idx]

        return np.array(data, dtype=np.float32),idx
    
    def elements(self):
      return eles
    
def calculate_entropy_mixing(comp):
  delta = 0
  for v in comp.get_el_amt_dict().values():
    if v>0:
      delta += v*np.log(v)
  return delta