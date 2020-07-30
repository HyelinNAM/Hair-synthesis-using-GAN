import torch
import SEAN.models.networks as networks
import SEAN.util.util as util

class Pix2PixModel(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor

        self.netG, self.netD, self.netE = self.initialize_networks(opt)

    def initialize_networks(self, opt):
        netG = networks.define_G(opt)
        netD = networks.define_D(opt) if opt.isTrain else None
        netE = networks.define_E(opt) if opt.use_vae else None

        if not opt.isTrain or opt.continue_train:
            netG = util.load_network(netG, 'G', opt.which_epoch, opt) 
            if opt.isTrain:
                netD = util.load_network(netD, 'D', opt.which_epoch, opt)

        return netG, netD, netE
        
    def forward(self,src_data,oth_data,mode):
        src_semantics, src_image = self.preprocess_input(src_data)
        oth_semantics, oth_image = self.preprocess_input(oth_data)
        #obj_dic = data['path']

        if mode == 'dyeing':
            with torch.no_grad():
                fake_image = self.netG.dyeing(src_semantics,src_image,oth_semantics,oth_image,src_data['path'])
            return fake_image

        elif mode == 'styling':
            with torch.no_grad():
                fake_image = self.netG.styling(src_semantics,src_image,oth_semantics,oth_data['path'])
            return fake_image
        
        else:
            raise ValueError("|mode| is invalid")
        
    def preprocess_input(self, data):
        # move to GPU and change data types
        data['label'] = data['label'].long() 
        if self.use_gpu():
            data['label'] = data['label'].cuda(non_blocking=True)
            data['instance'] = data['instance'].cuda(non_blocking=True)
            data['image'] = data['image'].cuda(non_blocking=True)

        # create one-hot label map
        label_map = data['label']
        bs, _, h, w = label_map.size()
        nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label \
            else self.opt.label_nc
        input_label = self.FloatTensor(bs, nc, h, w).zero_()
        input_semantics = input_label.scatter_(1, label_map, 1.0)

        # concatenate instance map if it exists
        if not self.opt.no_instance:
            inst_map = data['instance']
            instance_edge_map = self.get_edges(inst_map)
            input_semantics = torch.cat((input_semantics, instance_edge_map), dim=1)

        return input_semantics, data['image']

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0
    
    
