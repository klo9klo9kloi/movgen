# python test_encoder.py --name eta --dataroot 'data/dylan/eta/' --label_nc 3 --loadSize 640 --resize_or_crop none
import os
from collections import OrderedDict
from torch.autograd import Variable
from pix2pixHD.options.test_options import TestOptions
from dataset_creation import CreateDataLoader
from pose_to_image_model import MovGenModel
import pix2pixHD.util.util as util
from pix2pixHD.util.visualizer import Visualizer
from pix2pixHD.util import html
import torch

opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

# test
if not opt.engine and not opt.onnx:
    model = MovGenModel()
    model.initialize(opt)
    if opt.data_type == 16:
        model.half()
    elif opt.data_type == 8:
        model.type(torch.uint8)
            
    if opt.verbose:
        print(model)
else:
    from run_engine import run_trt_engine, run_onnx

prev_generation = None
for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    if opt.data_type == 16:
        data['label'] = data['label'].half()
    elif opt.data_type == 8:
        data['label'] = data['label'].uint8()

    minibatch = 1 
    pose_t = data['label'][:, :opt.label_nc, :, :]

    generated = model.inference(pose_t, prev_generation)
    prev_generation = generated
        
    visuals = OrderedDict([('input_label', pose_t[0].numpy().astype(np.int32)),
                           ('synthesized_image', util.tensor2im(generated.data[0]))])
    img_path = data['path']
    print('process image... %s' % img_path)
    visualizer.save_images(webpage, visuals, img_path)

webpage.save()