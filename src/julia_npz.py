import sys
import numpy as np
from os.path import join
import h5py
import pickle as pkl
import scipy
import chumpy   # only needed for smpl pkl

src_path = sys.argv[1]
out_npz_path = sys.argv[2]
# npz_path = "/home/nitin/Downloads/smplx/SMPLX_MALE.npz"

# load source
if src_path.endswith("npz"):
    data = np.load(src_path, allow_pickle = True)
elif src_path.endswith("pkl"):
    data = pkl.load(open(src_path,"rb"),encoding="latin1")

h = h5py.File(join(out_npz_path[:-3]+"hdf5"),"w")

outnpz = {}

for k,v in data.items():
    if type(v) == scipy.sparse._csc.csc_matrix:
        v = v.toarray()
    elif type(v) == str:
        continue
    elif type(v) == chumpy.ch.Ch:
        v = v.r
    try:
        if (v.dtype == np.float32) or \
            (v.dtype == np.float64):
            outnpz[k] = v.astype(np.float32)
            h.create_dataset(k, 
                        data= v.astype(np.float32))
        elif (v.dtype == np.int32) or \
            (v.dtype == np.int64):
            outnpz[k] = v.astype(np.int32)
            h.create_dataset(k, 
                        data=v.astype(np.int32))
        elif (v.dtype == np.uint64) or \
            (v.dtype == np.uint32):
            outnpz[k] = v.astype(np.uint32)
            h.create_dataset(k, 
                        data=v.astype(np.uint32))
        else:
            print(f"{k} : {v.dtype}")
    except:
        import ipdb;ipdb.set_trace()
        
h.close()

np.savez(out_npz_path, **outnpz)