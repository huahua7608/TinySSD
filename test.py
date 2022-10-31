import torch
from model import *
from utils.loss import *
import torchvision
import glob
from torch.functional import F
from utils.display import *


net = TinySSD(num_classes=1)
net = net.to('cuda'if torch.cuda.is_available() else 'cpu')

net.load_state_dict(torch.load('weights/net_30.pkl'))


def predict(X):
    net.eval()
    anchors, cls_preds, bbox_preds = net(X.to('cuda'))
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    output = multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]


files = glob.glob('detection/test/*.jpg')
for name in files:
    X = torchvision.io.read_image(name).unsqueeze(0).float()
    img = X.squeeze(0).permute(1, 2, 0).long()

    output = predict(X)
    display(img, output.cpu(), threshold=0.6)
    # 保存结果img
    torchvision.io.write_jpeg(img, 'results/' + name.split('\\')[-1])
    # break
