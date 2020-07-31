import torch

import argparse

from Face_parsing.test import parsing
from StarGAN_v2.test import make_img
from SEAN.test import reconstruct

def main(args):
    print(args)
    torch.manual_seed(args.seed)

    if args.mode == 'dyeing':
        # Parsing > SEAN
        parsing(respth='./results/label/src' ,dspth='./data/src/src') # parsing src_image
        parsing(respth='./results/label/others', dspth='./data/dyeing') # parsing ref_image
        reconstruct(args.mode)
        
    elif args.mode == 'styling_ref':
        # StarGAN > Parsing > SEAN
        make_img(args) 
        parsing(respth='./results/label/src', dspth='./data/src/src') # parsing src_image
        parsing(respth='./results/label/others', dspth='./results/img') # parsing fake_image
        reconstruct(args.mode)

    elif args.mode == 'styling_rand':
        # StarGAN > Parsing > SEAN
        make_img(args)
        parsing(respth='./results/label/src', dspth='./data/src/src')
        parsing(respth='./results/label/others', dspth='./results/img') # parsing fake_image
        reconstruct(args.mode)
    
    else:
        raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # implement
    parser.add_argument('--mode', type=str, required=True,
                        choices=['dyeing','styling_ref','styling_rand'], help='set mode')
    parser.add_argument('--seed', type=int, default=777,
                        help='Seed for random number generator')

    # StarGAN_v2
    parser.add_argument('--img_size', type=int, default=256, help='Image resolution')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers used in DataLoader')

    parser.add_argument('--num_domains', type=int, default=7, help='Number of domains')
    parser.add_argument('--latent_dim', type=int, default=16, help='Latent vector dimension')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension of mapping network')
    parser.add_argument('--style_dim', type=int, default=64,help='Style code dimension')
    parser.add_argument('--w_hpf', type=float, default=1, help='weight for high-pass filtering')

    parser.add_argument('--resume_iter', type=int, default=100000,help='Iterations to resume training/testing')
    parser.add_argument('--checkpoint_dir', type=str, default='pretrained_network/StarGAN')
    parser.add_argument('--wing_path', type=str, default='pretrained_network/StarGAN/wing.ckpt')

    parser.add_argument('--src_dir', type=str, default='./data/src')
    parser.add_argument('--result_dir', type=str, default='./results/img')
    
    # for styling_ref
    parser.add_argument('--ref_dir', type=str, default='./data/ref')

    # for styling_rand
    parser.add_argument('--target_domain', type=int, default=0)
    parser.add_argument('--num_outs_per_domain', type=int, default=3)
    
    args = parser.parse_args()
    main(args)
