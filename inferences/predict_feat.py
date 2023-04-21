import os
import torch
from PIL import Image
import torchvision.transforms as T
import baseline
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # 指定gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
normalize_transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])
transform = T.Compose([
            T.Resize([256,128]),
            T.ToTensor(),
            normalize_transform
        ])
def read_image(img_path):
    img = Image.open(img_path).convert('RGB')
    return img
model=baseline.Baseline()
model.load_param(r"C:\work\aic2023\resnet50_ibn_a_model_11.pth")
def inference_samples(model, transform, batch_size,gallery_list): # 传入模型，数据预处理方法，batch_size
    img_list=[]
    for g_img in tqdm(gallery_list):
        g_img = read_image(g_img)
        g_img = transform(g_img)
        img_list.append(g_img)
    img_data = torch.Tensor([t.numpy() for t in img_list])

    iter_n = len(img_list) // batch_size
    if len(img_list) % batch_size != 0:
        iter_n += 1
    all_feature = list()
    for i in tqdm(range(iter_n)):
        print("batch ----%d----" % (i))
        batch_data = img_data[i*batch_size:(i+1)*batch_size]
        with torch.no_grad():
            print(batch_data.shape)
            batch_feature = model(batch_data)
            all_feature.append(batch_feature)
    return all_feature
def eucliDist(A,B):
    return np.sqrt(sum(np.power((A - B), 2)))
def cos_sim(array1,array2):
    square1 = np.sum(np.square(array1), axis=1)[..., np.newaxis]
    # shape [1, m2]
    square2 = np.sum(np.square(array2), axis=1)[np.newaxis, ...]
    squared_dist = - 2 * np.matmul(array1, array2.T) + square1 + square2
    squared_dist[squared_dist < 0] = 0
    dist = np.sqrt(squared_dist)
    return dist
if __name__=="__main__":
    batch_size=2
    gallery_list=[r"C:\work\aic2023\data\sub\c047\images\c047_6_0_0.87255859375.jpg",r"C:\work\aic2023\data\sub\c047\images\c047_28_0_0.8271484375.jpg"]
    xs=inference_samples(model, transform, batch_size, gallery_list)
    xs=xs[0].detach().cpu().numpy()
    #x=np.mean(xs, axis=0)
    x=eucliDist(xs[0],xs[1])
    print(x)
