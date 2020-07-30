import os
from collections import OrderedDict
from itertools import cycle

from SEAN import data
from SEAN.options.test_options import TestOptions
from SEAN.models.pix2pix_model import Pix2PixModel
from SEAN.util.visualizer import Visualizer

def reconstruct(mode):
    opt = TestOptions().parse()
    opt.status = 'test'
    opt.contain_dontcare_label = True
    opt.no_instance = True

    model = Pix2PixModel(opt)
    model.eval()

    visualizer = Visualizer(opt)

    # make dataloader for source image
    src_dataloader = data.create_dataloader(opt)

    # make dataloader for ref/generated image
    if mode == 'dyeing':
        opt.styling_mode = mode # dyeing

        opt.image_dir = './data/dyeing'
        opt.label_dir = './results/label/others'

    else: # styling_ref / styling_rand
        opt.styling_mode = 'styling'

        opt.image_dir = './results/img'
        opt.label_dir = './results/label/others'

    oth_dataloader = data.create_dataloader(opt)

    for i, data_i in enumerate(zip(cycle(src_dataloader),oth_dataloader)):
        src_data = data_i[0]
        oth_data = data_i[1]
        generated = model(src_data,oth_data, mode=opt.styling_mode)

        img_path = src_data['path']

        for b in range(generated.shape[0]):
            print('process image... %s' % img_path[b])
            visuals = OrderedDict([('input_label', src_data['label'][b]),
                               ('synthesized_image', generated[b])])

            visualizer.save_images(visuals, img_path[b:b + 1],opt.results_dir,f'results_{i}')
    
        








    