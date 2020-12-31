from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import copy
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Preprocessing

imsize = 512 if torch.cuda.is_available() else 128

img_transforms = transforms.Compose([
    transforms.ToTensor(),
])

img_removeTransforms = transforms.ToPILImage()

img_removeNormalization = transforms.Normalize((-2.12, -2.04, -1.80),(4.37, 4.46, 4.44))

# plt.ion()

def image_loader(img_name, maxSize=None, shape=None):
    image = Image.open(img_name).resize((imsize, imsize))
    image = img_transforms(image)
    return image

def imshow(tensor, title=None, isNormalized=False):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = img_removeTransforms(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
    plt.show()

# Normalization
class Normalization(nn.Module):
    def __init__(self):
        super(Normalization, self).__init__()

        self.mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        self.std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)

    def forward(self, img):
        return (img - self.mean.to(img.device)) / self.std.to(img.device)

# Content Loss is the mse between the content and the target images
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        # We dont need the gradients, hence we detach the target
        self.target = target.detach()
        self.loss = None

    def forward(self, input):
        self.loss = F.l1_loss(input, self.target)
        return input

# Style loss is the mse between the gram matrices between the style image and the target image

def gram_matrix(input):
    batch_size, numberOfFeatureMaps, height, width = input.size()

    features = input.view(batch_size * numberOfFeatureMaps, height * width)

    # Gram matrix will be the dot product of the features with its transpose. Check resources for better explanation
    G_matrix = torch.mm(features, features.t())

    # We normalize the values by dividing with the total number of elements in each feature map
    return G_matrix.div(batch_size * numberOfFeatureMaps * height * width)

class StyleLoss(nn.Module):

    def __init__(self, target_features):
        super(StyleLoss, self).__init__()
        self.target_Gram_Matrix = gram_matrix(target_features.detach()).detach()
        self.loss = None

    def forward(self, input):
        input_Gram_Matrix = gram_matrix(input)
        self.loss = F.l1_loss(input_Gram_Matrix, self.target_Gram_Matrix)
        return input

def rename_model_layers(model):

    block, number = 1, 1
    newModel = nn.Sequential()

    for layer in model.children():
        if isinstance(layer, nn.Conv2d):
            name = 'conv{}_{}'.format(block, number)
        elif isinstance(layer, nn.ReLU):
            name = 'relu{}_{}'.format(block, number)
            # inplace False, relu performs better
            layer = nn.ReLU(inplace=False)
            number += 1
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool{}_{}'.format(block, number)
            # The paper suggests that avg pool produces better results than max pool
            # So i add a new avg pool layer with same properties as maxpool
            layer = nn.AvgPool2d(layer.kernel_size, layer.stride)
            number = 1
            block += 1
        newModel.add_module(name, layer)
    
    return newModel

def modelAndLosses(content_img, style_img):
    vgg19 = models.vgg19(pretrained=True).features.eval()
    vgg19 = copy.deepcopy(vgg19)
    vgg19 = rename_model_layers(vgg19).to(device)
    
    content_layers = ['conv5_1']
    style_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']

    normalization = Normalization().to(device)

    # Initial content and style losses
    content_losses = []
    style_losses = []
    last_layer = 0

    model = nn.Sequential(normalization)

    
    for i, (name, layer) in enumerate(vgg19.named_children()):
        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img)
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)
            last_layer = i
        
        if name in style_layers:
            target = model(style_img)
            style_loss = StyleLoss(target)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)
            last_layer = i
        
    # for i in range(len(model) - 1, -1, -1):
    #     if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
    #         break;
    last_layer += 1 + len(content_losses) + len(style_losses)
    model = model[:(last_layer + 1)]
    return model, style_losses, content_losses

def get_input_optimizer(config, input_img):
    optimizer = optim.Adam([input_img.requires_grad_()], lr=config.lr)
    return optimizer

def transfer_style(config, content_img, style_img, input_img=None):
    
    print("Building the style transfer model...")
    model, style_losses, content_losses = modelAndLosses(content_img, style_img)

    model.to(device)
    content_img = content_img.unsqueeze(0).to(device)
    style_img = style_img.unsqueeze(0).to(device)

    batch_size, numberOfFeatureMaps, height, width = content_img.data.size()
    if input_img is None:
        input_img = torch.randn(content_img.data.size(), device=device)
        input_img = input_img * 0.01

    optimizer = get_input_optimizer(config, input_img)

    transform = nn.Sequential(
        transforms.RandomResizedCrop((width, height), scale=(0.97, 1.0), ratio=(0.97, 1.03)),
        transforms.RandomRotation(degrees=1.0)
    )

    print("Optimizing..")
    run = [0]
    while run[0] <= config.total_step:
        
        input_img.data.clamp_(0, 1)

        optimizer.zero_grad()
        input_img = model(transform(input_img))
        style_loss = 0
        content_loss = 0
    
        for sl in style_losses:
            style_loss += sl.loss
        for cl in content_losses:
            content_loss += cl.loss
        
        style_loss *= config.style_weight
        # style_loss *= 110
        content_loss *= 10

        loss = style_loss + content_loss
        loss.backward()
        optimizer.step()

        run[0] += 1
        if run[0]%config.log_step == 0:
            print("Run {}:".format(run[0]))
            print("Style loss: {:4f}, Content Loss: {:4f}".format(
                style_loss, content_loss
            ))
            print()
        
        if run[0]%config.sample_step == 0:
            img = input_img.clone().squeeze()
            # img = img_removeNormalization(img)
            torchvision.utils.save_image(img, 'output-{}.jpg'.format(run[0]));

    return input_img


def main(config):
    content_img = image_loader(config.content).to(device)
    style_img = image_loader(config.style).to(device)
    # imshow(content_img, title="Content Image", isNormalized=True)
    # imshow(style_img, title="Style Image", isNormalized=True)

    target_img = transfer_style(config, content_img, style_img)
    # imshow(target_img, title="Final Image", isNormalized=False)

    target_img = target_img.clone().squeeze()
    # target_img = img_removeNormalization(target_img)
    torchvision.utils.save_image(target_img, 'output-final.jpg');

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--content', type=str, default='samples/sample_content.jpeg')
    parser.add_argument('--style', type=str, default='samples/sample_style.jpg')
    parser.add_argument('--total_step', type=int,  default=2000)
    parser.add_argument('--log_step', type=int,  default=10)
    parser.add_argument('--sample_step', type=int,  default=500)
    parser.add_argument('--style_weight', type=float,  default=5000)
    parser.add_argument('--lr', type=float,  default=0.05)
    config = parser.parse_args()
    print(config)
    main(config)

#!python main.py --content dancing.jpg --style style.jpeg --total_step 2000 --sample_step 1000 --lr 0.05 --style_weight 5000 --log_step 100