from smplx.body_models import SMPL
import torch

device = torch.device("cpu")
smpl = SMPL("/trainman-mount/trainman-k8s-storage-8e3fd382-2f06-466b-8bde-ca27be865ff5/projects/smplx/smplx/models/smpl/SMPL_MALE.pkl").to(device)
betas = torch.zeros(1,10).float().to(device)
global_orient = torch.zeros(1,3).float().to(device)
pose = torch.zeros(1,69).float().to(device)
trans = torch.zeros(1,3).float().to(device)

%timeit smplout = smpl.forward(betas=betas,body_pose=pose,global_orient=global_orient,transl=trans)

