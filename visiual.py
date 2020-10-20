import argparse

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image
import torchvision

from models2104 import FSRCNN
from utils import convert_ycbcr_to_rgb, preprocess, calc_psnr
from torch.utils.tensorboard import SummaryWriter


def show_images(describe, images):
    writer = SummaryWriter('./vision_log/table5-3/')
    if images.dim()==3:
        images = images.unsqueeze(0)
        images = images.transpose(0,1) #1CHW->C1HW
    if images.size(0)==1 and images.size(1)>images.size(0):
        images = images.transpose(0,1) #1CHW->C1HW
    img_grid = torchvision.utils.make_grid(images)
    print(img_grid.size())
    writer.add_image(describe, img_grid)
    writer.close()

def get_feature_map(model,feature_map_list):
    def farward_hook(module, inp, outp):
        feature_map_list.append(outp)
    #修改指定层就可以了
    model.first_part.register_forward_hook(farward_hook)
    model.mid_part.register_forward_hook(farward_hook)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--image-file', type=str, required=True)
    parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--residual', action='store_true')

    args = parser.parse_args()

    

    cudnn.benchmark = True
    device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')

    # 加载模型参数
    model = FSRCNN(scale_factor=args.scale).to(device)
    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    # 打开图片
    image = pil_image.open(args.image_file).convert('RGB')

    image_width = (image.width // args.scale) * args.scale
    image_height = (image.height // args.scale) * args.scale

    #hr = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
    hr = image.crop((0,0,image_width,image_height))
    lr = hr.resize((hr.width // args.scale, hr.height //
                    args.scale), resample=pil_image.BICUBIC)
    bicubic = lr.resize((lr.width * args.scale, lr.height *
                         args.scale), resample=pil_image.BICUBIC)
    bicubic.save(args.image_file.replace(
        '.', '_bicubic_x{}.'.format(args.scale)))

    
    

    lr, _ = preprocess(lr, device)
    hr, _ = preprocess(hr, device)
    _, ycbcr = preprocess(bicubic, device)
    bicubic,_ = preprocess(bicubic,device)

    psnr = calc_psnr(hr, bicubic)
    print('PSNR: {:.2f}'.format(psnr))

    # 可视化原图lr和bicubic
    print(lr.size())
    print(bicubic.size())

    kernel = state_dict['first_part.0.weight']
    show_images('first kernel',kernel)
    show_images("lr image",lr)
    show_images("hr image",hr)
    show_images("bicubic",bicubic)

    model.eval()

    print(state_dict.keys())
    gpus=[6]
    if torch.cuda.device_count() > 1:
        print("Use", torch.cuda.device_count(), 'gpus')
        model = torch.nn.DataParallel(model.to(device),device_ids=gpus,output_device=gpus[0])
    
    if isinstance(model,torch.nn.DataParallel):
        model = model.module

    # 可视化第一层(self.)和最后一个卷积层特征图：
    feature_map_list = []  # 装feature map
    get_feature_map(model,feature_map_list)
    with torch.no_grad():
        preds = model(lr).clamp(0.0, 1.0)

    show_images("first_feature_map",feature_map_list[0])
    show_images("mid_feature_map",feature_map_list[1])

    show_images("before preds",preds)
    
    if args.residual:
        psnr = calc_psnr(hr, preds)
        print('preds and hr PSNR: {:.2f}'.format(psnr))
        preds = preds+bicubic

    psnr = calc_psnr(hr, preds)
    print('preds and hr PSNR: {:.2f}'.format(psnr))

    psnr = calc_psnr(hr, bicubic)
    print('bicubic and hr PSNR: {:.2f}'.format(psnr))
    

    show_images("after preds",preds)
    preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

    output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]
                      ).transpose([1, 2, 0])
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    output = pil_image.fromarray(output)
    output.save(args.image_file.replace(
        '.', '_fsrcnn_x{}.'.format(args.scale)))
