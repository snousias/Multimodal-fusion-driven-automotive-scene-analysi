import os
import operator

from numpy import copy
import sys
import torch
import torch.utils.data
from torch.optim.lr_scheduler import StepLR
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.datasets as datasets
import copy
from engine.trainer import Trainer
from model.squeezedet import SqueezeDetWithLoss
from utils.config import Config
from utils.model import load_model, load_official_model, save_model
from utils.logger import Logger
from utils.misc import load_dataset
from eval import eval_dataset

from model.squeezedet import Fire
from weight_sharing import quant_convlayer_weights

torch.cuda.empty_cache()

torch.cuda.empty_cache()


optimizerSelection='sgd'

def quantize(input, method, acceleration):
    if method == 'dl':
        return quant_convlayer_weights(input, sub_dim=8, accel=acceleration, coeff=3, sparsity_level=2,
                                       clust_scheme=method)
    elif method == 'vq':
        return quant_convlayer_weights(input, sub_dim=8, accel=acceleration, coeff=None, sparsity_level=None,
                                       clust_scheme=method)
    else:
        return None


def train_weight_sharing(cfg):
    Dataset = load_dataset(cfg.dataset)
    train_dataset = Dataset('train', cfg)
    val_dataset = Dataset('val', cfg)
    cfg = Config().update_dataset_info(cfg, train_dataset)
    Config().print(cfg)
    logger = Logger(cfg)

    clust_scheme = cfg.sharing_method
    accel = cfg.sharing_acceleration_factor

    model = SqueezeDetWithLoss(cfg)
    if cfg.load_model != '':
        if cfg.load_model.endswith('f364aa15.pth') or cfg.load_model.endswith('a815701f.pth'):
            model = load_official_model(model, cfg.load_model)
        else:
            model = load_model(model, cfg.load_model)

    # original_stdout = sys.stdout  # Save a reference to the original standard output
    # with open('model.txt', 'w') as f:
    #     sys.stdout = f  # Change the standard output to the file we created.
    #     print(model)
    #     sys.stdout = original_stdout  # Reset the standard output to its original value
    # exit(3)
    #TODO

    learning_rate_base=copy.deepcopy(cfg.lr)

    layer_id=0
    for idx, ProcessedLayer in enumerate(model.base._modules['features']):
        cfg.lr=copy.deepcopy(learning_rate_base)

        if optimizerSelection=='adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
            trainer = Trainer(model, optimizer, None, cfg)
        if optimizerSelection=='sgd':
            optimizer = torch.optim.SGD(model.parameters(),
                                        lr=cfg.lr,
                                        momentum=cfg.momentum,
                                        weight_decay=cfg.weight_decay)
            lr_scheduler = StepLR(optimizer, 60, gamma=0.5)
            trainer = Trainer(model, optimizer, lr_scheduler, cfg)
        if isinstance(optimizer,torch.optim.SGD):
            print('SDG...!')
        if isinstance(optimizer,torch.optim.Adam):
            print('Adam...!')
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=cfg.batch_size,
                                                num_workers=cfg.num_workers,
                                                pin_memory=True,
                                                shuffle=True,
                                                drop_last=True)

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=cfg.batch_size,
                                                num_workers=cfg.num_workers,
                                                pin_memory=True)
        metrics = trainer.metrics if cfg.no_eval else trainer.metrics + ['mAP']
        best = 1E9 if cfg.no_eval else 0
        better_than = operator.lt if cfg.no_eval else operator.gt
        if isinstance(ProcessedLayer, torch.nn.Conv2d):
            print("Conv2d found")
            # if isinstance(ProcessedLayer, torch.nn.Conv2d):
            #     weights = ProcessedLayer.weight.cpu().detach().numpy()
            #     weights_with_sharing, stats = quantize(weights, clust_scheme, accel)
            #     layer_id=layer_id+1
            #     with torch.no_grad():
            #         model.base._modules['features'][idx].weight.data = torch.from_numpy(weights_with_sharing)
            #     for param in model.base._modules['features'][idx].parameters():
            #         param.requires_grad = False
            #     model.cuda()
            # pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            # print(60*"=")
            # print("Total training params:"+str(pytorch_total_params))
            # print(60 * "=")



        if isinstance(ProcessedLayer, Fire):
            print("Fire found")
            if len(ProcessedLayer._modules)>0:
                for _, ProcessedLayer_Name_Level_1 in enumerate(ProcessedLayer._modules):
                    ProcessedLayer_Level_1=ProcessedLayer._modules[ProcessedLayer_Name_Level_1]
                    if "expand" in ProcessedLayer_Name_Level_1:
                        print(ProcessedLayer_Name_Level_1)
                        if isinstance(ProcessedLayer_Level_1, torch.nn.Conv2d):
                            weights = ProcessedLayer_Level_1.weight.cpu().detach().numpy()
                            weights_with_sharing, stats = quantize(weights, clust_scheme, accel)
                            layer_id=layer_id+1
                            with torch.no_grad():
                                ProcessedLayer._modules[ProcessedLayer_Name_Level_1].weight.data = torch.from_numpy(weights_with_sharing)
                            for param in ProcessedLayer._modules[ProcessedLayer_Name_Level_1].parameters():
                                param.requires_grad = False
                            model.cuda()
            pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(60 * "=")
            print("Total training params:" + str(pytorch_total_params))
            print(60 * "=")



            # for name, param in model.named_parameters():
            #     if param.requires_grad:
            #         print(name, param.data)



            for epoch in range(1, cfg.num_epochs + 1):
                train_stats = trainer.train_epoch(epoch, train_loader)
                logger.update(train_stats, phase='train', epoch=epoch)

                save_path = os.path.join(cfg.save_dir, 'modelopt_layer_'+str(layer_id).zfill(2)+'_last.pth')
                save_model(model, save_path, epoch)

                if epoch % cfg.save_intervals == 0:
                    save_path = os.path.join(cfg.save_dir, 'modelopt_layer_'+str(layer_id).zfill(2)+'_{}.pth'.format(epoch))
                    save_model(model, save_path, epoch)

                if cfg.val_intervals > 0 and epoch % cfg.val_intervals == 0:
                    val_stats = trainer.val_epoch(epoch, val_loader)
                    logger.update(val_stats, phase='val', epoch=epoch)

                    if not cfg.no_eval:
                        aps = eval_dataset(val_dataset, save_path, cfg)
                        logger.update(aps, phase='val', epoch=epoch)

                    value = val_stats['loss'] if cfg.no_eval else aps['mAP']
                    if better_than(value, best):
                        best = value
                        save_path = os.path.join(cfg.save_dir, 'model_best.pth')
                        save_model(model, save_path, epoch)

                logger.plot(metrics)
                logger.print_bests(metrics)

    torch.cuda.empty_cache()
