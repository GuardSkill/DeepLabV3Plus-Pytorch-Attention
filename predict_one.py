import torch
import torch.nn as nn
import network
import utils
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import cv2 as cv
from torch.nn import functional as F

model_map = {
    'deeplabv3_resnet50': network.deeplabv3_resnet50,
    'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
    'deeplabv3_resnet101': network.deeplabv3_resnet101,
    'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
    'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
    'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet,
    'deeplabv3plus_mobilenetSA': network.deeplabv3plus_mobilenetSA,
    'deeplabv3plus_mobilenetSAc': network.deeplabv3plus_mobilenetSAc,
    'deeplabv3plus_mobilenet_M': network.deeplabv3plus_mobilenetM,
    'deeplabv3plus_mobilenetSA_M': network.deeplabv3plus_mobilenetSAM,
    'deeplabv3plus_mobilenetSAc_M': network.deeplabv3plus_mobilenetSAcM
}


# 把label映射到不同的颜色 (voc数据集)
def voc_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap


if __name__ == "__main__":
    img_path = r"samples/114_image.png"              #要测试的图片
    result_path = r"samples/114_image_results.png"   #要保存的路劲

    # model_name ='deeplabv3plus_mobilenet'           #用什么模型什么方式
    # model_name ='deeplabv3plus_mobilenetSA'
    # model_name ='deeplabv3plus_mobilenetSAc'
    model_name = 'deeplabv3plus_mobilenet_M'
    # model_name ='deeplabv3plus_mobilenetSA_M'
    # model_name ='deeplabv3plus_mobilenetSAc_M'

    # 单模型读取
    ckpt_path = r'checkpoints/best_deeplabv3plus_mobilenet_voc_os16.pth'    #模型参数的位置
    # ckpt_path = r'checkpoints/best_deeplabv3plus_mobilenetSA_voc_os16.pth'
    # ckpt_path = r'checkpoints/best_deeplabv3plus_mobilenetSAc_voc_os16.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    val_transform = transforms.Compose([
        transforms.Resize(513),
        transforms.CenterCrop(513),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # 图像读取和预处理
    # img = Image.open(img_path).convert('RGB')
    # input = val_transform(img).unsqueeze(0)
    img = Image.open(img_path).convert('RGB')
    img = val_transform(img).unsqueeze(0)

    # 如果是多模型融合
    if 'M' in model_name:
        preds = []
        for class_num in range(1, 21):
            modelname = 'checkpoints/multiple_model/latest_%s_%s_class%d_os%d.pth' % (
                model_name, 'voc', class_num, 16,)
            checkpoint = torch.load(modelname, map_location=torch.device('cpu'))
            model = model_map[model_name](num_classes=21, output_stride=16)
            model.load_state_dict(checkpoint["model_state"])
            model = nn.DataParallel(model)
            model.to(device)
            model.eval()

            with torch.no_grad():
                outputs = nn.Sigmoid()(model(img))
                pred = outputs.detach().cpu().numpy()
                preds.append(pred)

        # 读取单个模型
        checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
        model = model_map[model_name.split('_')[0] + '_' + model_name.split('_')[1]](num_classes=21,
                                                                                     output_stride=16)
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        model.eval()
        # sin_outputs = model(images).cpu().numpy()
        sin_outputs = F.softmax(model(img), dim=1).detach().cpu().numpy()

        # 融合
        alpha = 0.5
        preds = np.concatenate(preds, 1)
        background_p = np.expand_dims(sin_outputs[:, 0, :, :], axis=1)  # 抽取单类模型预测的背景概率
        preds = np.concatenate((background_p, preds), 1)  # 单类模型与多类模型分数融合
        final_preds = alpha * preds + sin_outputs
        preds = np.argmax(final_preds, axis=1)
        pred = voc_cmap()[preds.squeeze(axis=0)].astype(np.uint8)
        Image.fromarray(pred).save(result_path)
        print("Prediction is saved in %s" % result_path)


    else:
        # 模型加载
        model = model_map[model_name](num_classes=21, output_stride=16)
        weights = torch.load(ckpt_path)["model_state"]
        model.load_state_dict(weights)
        model.to(device)
        model.eval()

        with torch.no_grad():
            print(img.shape)
            img = img.to(device)
            # preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            pred = model(img).max(dim=1)[1].cpu().numpy()[0, :, :]
            pred = voc_cmap()[pred].astype(np.uint8)

            Image.fromarray(pred).save(result_path)
            print("Prediction is saved in %s" % result_path)
