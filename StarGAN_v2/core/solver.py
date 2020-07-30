import os
from os.path import join as ospj
import time
import datetime
from munch import Munch
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from StarGAN_v2.core.model import build_model
from StarGAN_v2.core.checkpoint import CheckpointIO
from StarGAN_v2.core.data_loader import InputFetcher
import StarGAN_v2.core.utils as utils

class Solver(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.nets_ema = build_model(args) 

        for name, module in self.nets_ema.items():
            setattr(self, name + '_ema', module)

        self.ckptios = [CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets_ema.ckpt'), **self.nets_ema)]
        self.to(self.device)

        for name, network in self.named_children():
            # Do not initialize the FAN parameters
            if ('ema' not in name) and ('fan' not in name):
                print('Initializing %s...' % name)
                network.apply(utils.he_init)

    def _load_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.load(step)

    def using_reference(self, loaders):
        args = self.args
        nets_ema = self.nets_ema
        os.makedirs(args.result_dir, exist_ok=True)
        self._load_checkpoint(args.resume_iter)

        src = next(InputFetcher(loaders.src, None, args.latent_dim, 'test'))
        ref = next(InputFetcher(loaders.ref, None, args.latent_dim, 'test'))

        fname = ospj(args.result_dir, 'reference')
        print('Working on {}...'.format(fname))
        self.translate_using_reference(nets_ema, args, src.x, ref.x, ref.y, fname)

    def translate_using_reference(self, nets, args, x_src, x_ref, y_ref, fname):
        N, C, H, W = x_src.size() 
        wb = torch.ones(1, C, H, W).to(x_src.device)
        x_src_with_wb = torch.cat([wb, x_src], dim=0)

        masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None
        s_ref = nets.style_encoder(x_ref, y_ref) 
        s_ref_list = s_ref.unsqueeze(1).repeat(1, N, 1) 
        x_concat = [x_src_with_wb]

        for i, s_ref in enumerate(s_ref_list):
            x_fake = nets.generator(x_src, s_ref, masks=masks)
            utils.save_image(x_fake,1,f'{fname}_{i+1}.jpg')
            x_fake_with_ref = torch.cat([x_ref[i:i+1], x_fake], dim=0)
            x_concat += [x_fake_with_ref]

        x_concat = torch.cat(x_concat, dim=0)
        #utils.save_image(x_concat, N+1, f'{fname}.jpg')

    def using_latent(self,loaders):
        args = self.args
        nets = self.nets_ema
        os.makedirs(args.result_dir, exist_ok=True)
        self._load_checkpoint(args.resume_iter)

        src = next(InputFetcher(loaders.src, None, args.latent_dim, 'test'))

        fname = ospj(args.result_dir, 'using_latent')
        print(f'Working on {fname}...')

        target_domain = args.target_domain

        self.translate_using_latent(nets,args,src,target_domain,fname)

    def translate_using_latent(self,nets,args,src,trg_domain,fname):
        x_src = src.x
        device = x_src.device
        N = x_src.size(0) # 1

        y_trg_list = [torch.tensor(trg_domain)]
        z_trg_list = torch.randn(args.num_outs_per_domain, 1, args.latent_dim).repeat(1, N, 1).to(device)  # num_outs_per_domain arg 활용

        # utils.translate_using_latent
        N,C,H,W = x_src.size()
        latent_dim = z_trg_list[0].size(1)
        x_concat = [x_src]
        masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None

        for y_trg in y_trg_list:
            for i,z_trg in enumerate(z_trg_list):
                s_trg = nets.mapping_network(z_trg,y_trg)
                x_fake = nets.generator(x_src,s_trg,masks=masks) # torch.tensor
                utils.save_image(x_fake,1,f'{fname}_{i+1}.jpg')

                x_concat += [x_fake]

        x_concat = torch.cat(x_concat,dim=0)
        #utils.save_image(x_concat,args.num_outs_per_domain+1,f'{fname}.jpg')

    '''
    def vector_interpolation(self,loaders):
        args = self.args
        nets_ema = self.nets_ema
        os.makedirs(args.result_dir,exist_ok=True)
        self._load_checkpoint(args.resume_iter)

        start = next(InputFetcher(loaders.start, None, args.latent_dim, 'test'))
        end = next(InputFetcher(loaders.end, None, args.latent_dim, 'test'))
        src = next(InputFetcher(loaders.src, None, args.latent_dim, 'test'))

        fname = ospj(args.result_dir, 'vector_interpolation')
        print('Working on {}...'.format(fname))
        self.interpolate_vector(nets_ema, args, start.x, end.x, start.y, end.y, src.x, fname)
        
    def interpolate_vector(self, nets, args, x_start, x_end, y_start, y_end, x_src, fname):
        N, C, H, W = x_src.size()
        
        masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None
        s_start = nets.style_encoder(x_start, y_start)
        s_end = nets.style_encoder(x_end, y_end)
        s_trg_list = [torch.lerp(s_start,s_end,weight) for weight in np.linspace(0,1,args.num_outs_per_domain)] # 0 , 025, 0.5, 0.75, 1
        x_concat = [x_start]
        
        for i, s_trg in enumerate(s_trg_list):
            x_fake = nets.generator(x_src, s_trg, masks= masks)
            utils.save_image(x_fake,1,f'{fname}_{i+1}.jpg')

            x_concat += [x_fake]

        x_concat += [x_end]
        x_concat = torch.cat(x_concat, dim=0)
        utils.save_image(x_concat,args.num_outs_per_domain+2,f'{fname}.jpg')
    '''