
from munch import Munch
import torch
from torch.backends import cudnn
from StarGAN_v2.core.data_loader import get_test_loader
from StarGAN_v2.core.solver import Solver

def make_img(args):
    cudnn.benchmark = True
    torch.manual_seed(args.seed)

    solver = Solver(args)

    if args.mode == 'styling_ref':
        loaders = Munch(src=get_test_loader(root=args.src_dir,
                                            img_size = args.img_size,
                                            batch_size = args.batch_size,
                                            shuffle = False,
                                            num_workers = args.num_workers),
                        ref=get_test_loader(root=args.ref_dir,
                                            img_size = args.img_size,
                                            batch_size = args.batch_size,
                                            shuffle=False,
                                            num_workers= args.num_workers))

        solver.using_reference(loaders)

    elif args.mode =='styling_rand':
        loaders = Munch(src=get_test_loader(root=args.src_dir,
                                            img_size = args.img_size,
                                            batch_size = args.batch_size,
                                            shuffle = False,
                                            num_workers = args.num_workers))

        solver.using_latent(loaders)

    else:
        raise NotImplementedError
