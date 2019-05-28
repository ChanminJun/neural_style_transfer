#%%
import copy
import gc
import time

import IPython
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# imsize = 128

# loader = transforms.Compose([
#     transforms.Resize(imsize),
#     transforms.ToTensor(),
# ])


def image_loader(image_name, imsize, content_size=None):
    loader = transforms.Compose([
        transforms.Resize(imsize),
        transforms.ToTensor(),
    ])
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    if content_size:
        image = Image.open(image_name)
        loader = transforms.Compose([
        transforms.Resize(content_size),
        transforms.ToTensor(),
    ])
        image = loader(image).unsqueeze(0)
    print("Image: {}, tensor size: {}".format(image_name, image.shape))
    return image.to(device, torch.float)

# unloader = transforms.ToPILImage()

# plt.ion()

def imshow(tensor, title=None):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

class ContentLoss(nn.Module):
    
    def __init__(self, target, ):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        
    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    a, b, c, d = input.size()
    
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
        
    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input
    

cnn = models.vgg19(pretrained=True).features.to(device).eval()

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

class Normalization(nn.Module):
    
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)
        
    def forward(self, img):
        return (img - self.mean) / self.std

content_layers_default = ["conv_4"]
style_layers_default = ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]

def get_style_model_and_losses(
    cnn, 
    normalization_mean, 
    normalization_std,
    style_img, content_img,
    content_layers=content_layers_default,
    style_layers=style_layers_default,
):
    cnn = copy.deepcopy(cnn)
    
    normalization = Normalization(normalization_mean, normalization_std).to(device)
    
    content_losses = []
    style_losses = []
    
    model = nn.Sequential(normalization)
    
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = "conv_{}".format(i)
        elif isinstance(layer, nn.ReLU):
            name = "relu_{}".format(i)
            
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = "pool_{}".format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = "bn_{}".format(i)
        else:
            raise RuntimeError("Unrecognized layer: {}".format(layer.__class__.__name__))
        
        model.add_module(name, layer)
        
        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)
            
        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)
            
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[:(i + 1)]
    
    return model, style_losses, content_losses
    

def get_input_optimizer(input_img, lr=1):
    optimizer = optim.LBFGS([input_img.requires_grad_()], lr=lr, history_size=5)
    return optimizer

def run_style_transfer(
        cnn, normalization_mean, normalization_std, content_path, style_path, 
        reference_path=None, imsize=128, num_steps=1000, style_weight=1e6,
        content_weight=1, lr=1, random_img=False, show_disp=True, art_to_art=False
    ):
    content_img = image_loader(content_path, imsize)
    style_img = image_loader(style_path, imsize)
    
    content_size = content_img.size()[-2:]
    style_size = style_img.size()[-2:]
    
    if content_size != style_size:
        print("WARNING: IMG SIZE DIFFERENT!!!")
        print(content_size, style_size)
        style_img = image_loader(style_path, imsize, content_size)
        
    if random_img:
        input_img = torch.rand(content_img.shape).to(device)
    else:
        input_img = content_img.clone()
    
    if reference_path and not art_to_art:
        reference_img = image_loader(reference_path, imsize, content_size)
        input_img = (9 * input_img + reference_img) / 10
        del reference_img
        torch.cuda.empty_cache()
    
    print("Building the style transfer model..")
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)
    optimizer = get_input_optimizer(input_img, lr=lr)
    
    print("Optimizing..")
    run = [0]
    best_img = None
    best_loss = np.inf
    while run[0] < num_steps:
        
        def closure():
            
            input_img.data.clamp_(0, 1)
            
            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0
            
            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss
                
            style_score *= style_weight
            content_score *= content_weight
                        
            loss = style_score + content_score
            loss.backward()
            
            run[0] += 1
            if run[0] % (num_steps/10) == 0:
                if show_disp:                
                    IPython.display.clear_output(wait=True)
                    plt.figure(figsize=(8,8))
                    imshow(input_img)
                print("run {}:".format(run))
                print("Style Loss : {:4f} Content Loss : {:4f}".format(
                    style_score, content_score))

            return style_score + content_score

        optimizer.step(closure)
    
        if closure() < best_loss:
            best_img = input_img.clone()
            best_loss = closure()
        

    best_img.data.clamp_(0, 1)
    
    return best_img

def transfer_procedure(
        content_path, style_path, result_path, no_pass=2, imsize=224, show=True,
        num_steps=10000, art_to_art=True
    ):
    imsize = imsize
    result_path = result_path
    for trial in range(1, no_pass + 1):
        resolution = imsize * trial
        if trial == 1:
            reference_path = None
        else:
            reference_path = result_path

        loader = transforms.Compose([
            transforms.Resize(resolution),
            transforms.ToTensor(),
        ])
    
        output = run_style_transfer(
            cnn, cnn_normalization_mean, cnn_normalization_std, content_path, 
            style_path, reference_path, imsize=resolution, lr=lr, style_weight=style_weight, art_to_art=art_to_art,
            num_steps=num_steps, content_weight=1, show_disp=show
        )
        if show:
            plt.figure()
            imshow(output, title="Output Image")
            plt.close("all")
            plt.ioff()
        
        save_image(output, result_path)
        num_steps /= 4

#%%
if __name__ == "__main__":

    content_url = "https://cdn.pixabay.com/photo/2013/01/05/21/02/mona-lisa-74050_960_720.jpg"
    style_url = "https://news.artnet.com/app/news-upload/2018/02/Lot-7-Pablo-Picasso-FEMME-AU-BE%CC%81RET-ET-A%CC%80-LA-ROBE-QUADRILLE%CC%81E-MARIE-THE%CC%81RE%CC%80SE-WALTER-est.-upon-request-860x1024.jpg"

    img_data = requests.get(content_url).content
    with open("./content_imgs/content_1.jpg", "wb") as handler:
        handler.write(img_data)

    img_data = requests.get(style_url).content
    with open("./style_imgs/style_1.jpg", "wb") as handler:
        handler.write(img_data)

    content_path = "./content_imgs/content_1.jpg"
    style_path = "./style_imgs/style_1.jpg"
    result_path = "./result_imgs/test_result.png"
    no_pass=2
    imsize=224
    show=False
    num_steps=10000
    lr = 1
    style_weight=1e6
    
    transfer_procedure(
        content_path, style_path, result_path, no_pass=no_pass, imsize=imsize,
        show=show, num_steps=num_steps, art_to_art=True
    )
#%%
