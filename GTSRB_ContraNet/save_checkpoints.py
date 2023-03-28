import os

import torch
import encoder
import glob
from os.path import dirname, abspath, exists, join
from models import invencoder_resnet as module


def load_checkpoint(model, optimizer, filename, metric=False, ema=False):
    start_step = 0
    if isinstance(model, encoder.VAE):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
        return model 
    if ema:
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
        return model
    else:
        checkpoint = torch.load(filename)
        seed = checkpoint['seed']
        run_name = checkpoint['run_name']
        start_step = checkpoint['step']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        ada_p = checkpoint['ada_p']
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

        if metric:
            best_step = checkpoint['best_step']
            best_fid = checkpoint['best_fid']
            best_fid_checkpoint_path = checkpoint['best_fid_checkpoint_path']
            return model, optimizer, seed, run_name, start_step, ada_p, best_step, best_fid, best_fid_checkpoint_path
    return model, optimizer, seed, run_name, start_step, ada_p










checkpoint_dir = '../pertrain_model'
when='best'
g_checkpoint_dir = glob.glob(join('../pertrain_model',"model=G-{when}-weights-step*.pth".format(when=when)))[0]
d_checkpoint_dir = glob.glob(join('../pertrain_model',"model=D-{when}-weights-step*.pth".format(when=when)))[0]
local_rank='cuda'
Gen = module.Generator(80, 128, 32, 96, True, True,
                       2, 'ReLU', 'ProjGAN', 10,
                       'ortho', 'N/A', False).to(local_rank)

Dis = module.Discriminator(32, 96, True, True, 1,
                           'ReLU', 'ProjGAN', 'N/A', 10, False,
                           False, 'ortho', 'N/A', False).to(local_rank)
G_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, Gen.parameters()), 0.0001, [0.00001, 0.00001], eps=1e-6)

D_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, Dis.parameters()), 0.0001, [0.00001, 0.00001], eps=1e-6)


Gen, G_optimizer, trained_seed, run_name, step, prev_ada_p = load_checkpoint(Gen, G_optimizer, g_checkpoint_dir)
    
Dis, D_optimizer, trained_seed, run_name, step, prev_ada_p, best_step, best_fid, best_fid_checkpoint_path =\
        load_checkpoint(Dis, D_optimizer, d_checkpoint_dir, metric=True)


g_states = {'seed': trained_seed, 'run_name': run_name, 'step': step, 'best_step': best_step,
                    'state_dict': Gen.state_dict(), 'optimizer': G_optimizer.state_dict(), 'ada_p': prev_ada_p}

d_states = {'seed': trained_seed, 'run_name': run_name, 'step': step, 'best_step': best_step,
                    'state_dict': Dis.state_dict(), 'optimizer': D_optimizer.state_dict(), 'ada_p': prev_ada_p,
                    'best_fid': best_fid, 'best_fid_checkpoint_path': best_fid_checkpoint_path}

d_checkpoint_output_path = join('./', "model=D-{when}-weights-step={step}.pth".format(when='best', step=str(step)))
g_checkpoint_output_path = join('./', "model=G-{when}-weights-step={step}.pth".format(when='best', step=str(step)))
torch.save(d_states, d_checkpoint_output_path, _use_new_zipfile_serialization=False)
torch.save(g_states, g_checkpoint_output_path, _use_new_zipfile_serialization=False)