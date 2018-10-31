import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
from util import html
import torch
from util import util
from PIL import Image
import numpy as np

if __name__ == '__main__':
    with torch.no_grad():
        opt = TestOptions().parse()
        opt.nThreads = 1   # test code only supports nThreads = 1
        opt.batchSize = 1  # test code only supports batchSize = 1
        opt.serial_batches = True  # no shuffle
        opt.no_flip = True  # no flip

        data_loader = CreateDataLoader(opt)
        dataset = data_loader.load_data()
        model = create_model(opt)
        name_len = len(str(dataset.__len__()))

        for i, data in enumerate(dataset):

            model.set_input(data)
            result = model.test()
            pose = model.get_pose()
            img = model.get_img()
            img = util.tensor2im(img)
            result = util.tensor2im(result)
            pose = util.tensor2im(pose)
            source = np.concatenate([img,pose],axis=0)
            result = Image.fromarray(result)
            result = result.resize((opt.loadSize*2, opt.loadSize*2), Image.ANTIALIAS)
            result = np.concatenate([source, result], axis=1)
            result = Image.fromarray(result)
            tmp = str(i)
            name = []
            for j in range(name_len - len(tmp)):
                name.append('0')
            name.append(str(i))
            name = str().join(name)
            result.save('./results/'+name+'.png','PNG',quality=100)
            print(i/dataset.__len__())
        print("finished")