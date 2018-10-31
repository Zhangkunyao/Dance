#coding=utf-8
import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
import os
from Count_F_Measure import F_Measure

# --voc_dataroot /media/kun/Dataset/PASCAL-VOC/PASCAL-VOC2012/VOCdevkit/ --dataroot ./datasets/sal --name VOC_sal --model cycle_gan --no_dropout --gpu_ids 0,1
# --Imagenet True --imagenet_dataroot /media/kun/Dataset/Imagenet/VOC_Label_New/ --dataroot ./datasets/sal --name imagenet_sal --model cycle_gan --no_dropout --gpu_ids 1,0

if __name__ == '__main__':
    opt = TrainOptions().parse()
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    model = create_model(opt) #G输入是一张图片输出是一张图片（3通道）D输入一张图片，输出是一张ferture map。
    visualizer = Visualizer(opt)
    total_steps = 0
    count_print_loss = 0
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1,):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += 1
            epoch_iter += opt.batchSize
            model.set_input(data)

            # 100个batch 叠加一次标签转换
            if total_steps % 10 == 0:
                model.optimize_parameters(flag_shuffer=True,use = True)
            else:
                if epoch>=0:
                    model.optimize_parameters(flag_shuffer=False,use = True)
                else:
                    model.optimize_parameters(flag_shuffer=False, use=False)
            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_steps % opt.print_freq == 0:
                losses = model.get_current_losses()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)

                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses)
            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_networks('latest')

            iter_data_time = time.time()
        model.save_networks('latest')
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
    model.save_networks('latest')