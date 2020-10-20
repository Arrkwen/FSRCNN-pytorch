import argparse
import os
import copy

import PIL.Image as pil_image

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from datasets import TrainDataset, EvalDataset
from utils import AverageMeter, calc_psnr,tensor_to_PIL
from torch.utils.tensorboard import SummaryWriter

#from models import FSRCNN
from models262 import FSRCNN
from losses.losses import HuberLoss, CharbonnierLoss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, required=True)
    parser.add_argument('--eval-file', type=str, required=True)
    parser.add_argument('--vision-file', type=str, required=True)
    parser.add_argument('--outputs-dir', type=str, required=True)
    parser.add_argument('--outputs-file', type=str, required=True)
    parser.add_argument('--weights-file', type=str)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=20)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--residual', action='store_true')
    parser.add_argument('--loss-type',type=str,required=True)
    parser.add_argument('--loss-para',type=float,default=1.0)
    parser.add_argument('--optim-method',type=str,required=True)

    args = parser.parse_args()

    args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)
    # 对于网络结构固定，网络的输入形状（包括 batch size，图片大小，输入的通道）不变，可以为每个网络层寻找适合的卷积算法，实现网络加速
    cudnn.benchmark = True
    gpus = [6]
    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(gpus[0]))
        cudnn.benchmark = True
        # 为GPU中设置种子，生成随机数
        torch.cuda.manual_seed_all(args.seed)
    else:
        device = torch.device('cpu')
        # 为CPU中设置种子，生成随机数
        torch.manual_seed(args.seed)

    #model = FSRCNN(scale_factor=args.scale)
    model = FSRCNN(scale_factor=args.scale)
    # 多gpu计算
    if torch.cuda.device_count() > 1:
        print("Use", torch.cuda.device_count(), 'gpus')
        model = nn.DataParallel(model.to(device),device_ids=gpus,output_device=gpus[0])
    if isinstance(model,torch.nn.DataParallel):
        model = model.module

    if args.loss_type == 'MSELoss':
        criterion = nn.MSELoss()
    elif args.loss_type == 'HuberLoss':
        criterion = HuberLoss(delta=args.loss_para)
    else:
        criterion = CharbonnierLoss(eps=args.loss_para)

    print(args.loss_type)
    print(criterion)

    if args.optim_method == 'Adam':
        optimizer = optim.Adam([
            {'params': model.first_part.parameters()},
            {'params': model.mid_part.parameters()},
            {'params': model.last_part.parameters(), 'lr': args.lr * 0.1}
        ], lr=args.lr)
    elif args.optim_method == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr,momentum=0.9)

    train_dataset = TrainDataset(args.train_file)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True)
    eval_dataset = EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0

    #可视化psnr
    writer = SummaryWriter(args.vision_file)
    for epoch in range(args.num_epochs):
        model.train()
        epoch_losses = AverageMeter()
        
        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size), ncols=80) as t:
            t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))
            for data in train_dataloader:
                #lr,hr
                inputs, labels = data

                inputs = inputs.to(device)
                labels = labels.to(device)

                preds = model(inputs)

                loss = criterion(preds, labels)

                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(inputs))
        
        torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))
        
        model.eval()
        epoch_psnr = AverageMeter()

        for data in eval_dataloader:
            inputs, labels, lr_bicubic = data

            inputs = inputs.to(device)
            labels = labels.to(device)
            lr_bicubic = lr_bicubic.to(device)

            with torch.no_grad():
                preds = model(inputs).clamp(0.0, 1.0)

            if args.residual:
                preds = preds+lr_bicubic
                
            epoch_psnr.update(calc_psnr(preds, labels), len(inputs))

            

        print('eval psnr: {:.2f}'.format(epoch_psnr.avg))
        writer.add_scalar("psnr",epoch_psnr.avg,epoch)
        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())
    writer.close()
    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
    if args.residual:
        torch.save(best_weights, os.path.join(args.outputs_dir, args.outputs_file))
    else:
        torch.save(best_weights, os.path.join(args.outputs_dir, args.outputs_file))
