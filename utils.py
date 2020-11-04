# Utilities Module
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
from torchvision import transforms as T
from PIL import Image
import os
from torchvision.models import vgg16
import hyperparmeters

vgg = vgg16(pretrained=True)
"""train_on_gpu = torch.cuda.is_available()
if train_on_gpu:
    vgg.cuda()"""

class Conv2DBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, k_size,
                 stride, padding, bias=True, batch_norm = True):
        """

        :param in_channels: input channels to the convolution layer
        :param out_channels:  output channels to the convolution layer
        :param k_size: kernel size
        :param stride: stride size
        :param padding: padding size
        :param bias: boolean value to include bias or not
        :param batch_norm: boolean value to include batch normalization or not
        """
        super(Conv2DBatchNorm, self).__init__()
        conv1 = nn.Conv2d(in_channels, out_channels, k_size, stride, padding, bias = bias)
        relu = nn.ReLU(inplace=True)
        if batch_norm:
            self.unit = nn.Sequential(conv1, nn.BatchNorm2d(out_channels),relu)
        else:
            self.unit = nn.Sequential(conv1,relu)

    def forward(self, x):
        x = self.unit(x)
        return x

class segnetencoder3(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        :param in_channels: input channels
        :param out_channels: out channels
        """
        super(segnetencoder3, self).__init__()
        self.conv1 = Conv2DBatchNorm(in_channels, out_channels, 3, 1, 1)
        self.conv2 = Conv2DBatchNorm(out_channels, out_channels, 3, 1,1)
        self.conv3 = Conv2DBatchNorm(out_channels, out_channels, 3, 1, 1)
        self.maxpool = nn.MaxPool2d(2,2,return_indices=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        unpooled_shape = x.size()
        x, indices = self.maxpool(x)
        return x, indices, unpooled_shape


class segnetencoder2(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        :param in_channels:
        :param out_channels:
        """
        super(segnetencoder2, self).__init__()
        self.conv1 = Conv2DBatchNorm(in_channels, out_channels, 3, 1, 1)
        self.conv2 = Conv2DBatchNorm(out_channels, out_channels, 3, 1,1)
        self.maxpool = nn.MaxPool2d(2,2,return_indices=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        unpooled_shape = x.size()
        x, indices = self.maxpool(x)
        return x, indices, unpooled_shape

class segnetdecoder2(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        :param in_channels: input channels to the decoder layer
        :param out_channels: output channels to the decoder layer
        """
        super(segnetdecoder2, self).__init__()
        self.conv1 = Conv2DBatchNorm(in_channels , out_channels, 3, 1, 1)
        self.conv2 = Conv2DBatchNorm(out_channels, out_channels, 3, 1, 1)
        self.unpool = nn.MaxUnpool2d(2,2)
    def forward(self, inputs, indices, output_shape):
        x = self.unpool(input=inputs, indices=indices,output_size=output_shape)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class segnetdecoder3(nn.Module):
    def __init__(self, in_channels, out_channels):
        """

        :param in_channels: input channels to the decoder layer
        :param out_channels: output channels to the decoder layer
        """
        super(segnetdecoder3, self).__init__()
        self.conv1 = Conv2DBatchNorm(in_channels , out_channels, 3, 1, 1)
        self.conv2 = Conv2DBatchNorm(out_channels, out_channels, 3, 1, 1)
        self.conv3 = Conv2DBatchNorm(out_channels, out_channels, 3,1,1)
        self.unpool = nn.MaxUnpool2d(2,2)

    def forward(self, inputs, indices, output_shape):
        x = self.unpool(input=inputs, indices=indices,output_size=output_shape)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

class Segnet(nn.Module):
    """Segnet Architecture Class for Semantic Segmentation"""
    def __init__(self,in_channels = 3,classes = 23):
        super(Segnet, self).__init__()

        self.encoder1 = segnetencoder2(in_channels, 64)
        self.encoder2 = segnetencoder2(64, 128)
        self.encoder3 = segnetencoder3(128, 256)
        self.encoder4 = segnetencoder3(256,512)
        self.encoder5 = segnetencoder3(512, 512)

        self.decoder5 = segnetdecoder3(512, 512)
        self.decoder4 = segnetdecoder3(512,256)
        self.decoder3 = segnetdecoder3(256, 128)
        self.decoder2 = segnetdecoder2(128, 64)
        self.decoder1 = segnetdecoder2(64, classes)

    def forward(self, x):
        # Encoder
        enc1, indices_1, unpool_shape1 = self.encoder1(x)
        enc2, indices_2, unpool_shape2 = self.encoder2(enc1)
        enc3, indices_3, unpool_shape3 = self.encoder3(enc2)
        enc4, indices_4, unpool_shape4 = self.encoder4(enc3)
        enc5, indices_5, unpool_shape5 = self.encoder5(enc4)
        # Decoder
        dc5 = self.decoder5(enc5, indices_5, unpool_shape5)
        dc4 = self.decoder4(dc5, indices_4, unpool_shape4)
        dc3 = self.decoder3(dc4, indices_3, unpool_shape3)
        dc2 = self.decoder2(dc3, indices_2, unpool_shape2)
        dc1 = self.decoder1(dc2, indices_1, unpool_shape1)

        return dc1

    def init_vgg16_params(self):
        blocks = [self.encoder1, self.encoder2, self.encoder3, self.encoder4, self.encoder5]

        features = list(vgg.features.children())

        vgg_layers = []
        for _layer in features:
            if isinstance(_layer, nn.Conv2d):
                vgg_layers.append(_layer)

        merged_layers = []
        for idx, conv_block in enumerate(blocks):
            if idx < 2:
                units  = [conv_block.conv1.unit, conv_block.conv2.unit]

            else:
                units = [
                    conv_block.conv1.unit,
                    conv_block.conv2.unit,
                    conv_block.conv3.unit
                ]
            for _unit in units:
                for _layer in _unit:
                    if isinstance(_layer, nn.Conv2d):
                        merged_layers.append(_layer)

        assert len(vgg_layers) == len(merged_layers)

        for l1,l2 in zip(vgg_layers, merged_layers):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data



class DroneDataset(Dataset):

    def __init__(self, img_path, mask_path,X, mean, std, transform=None):
        super(DroneDataset, self).__init__()
        self.image_path = img_path
        self.mask_path = mask_path
        self.X = X
        self.transform = transform
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """

        :param idx: idx of the mask and the image
        :return: get item per index
        """
        path_image = os.path.join(self.image_path, self.X[idx] + '.jpg')
        path_mask = os.path.join(self.mask_path, self.X[idx] + '.png')
        img = cv2.imread(path_image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(path_mask, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (224,224))
        mask = cv2.resize(mask, (224,224))
        if self.transform is not None:
            aug = self.transform(image = img, mask = mask)
            img = Image.fromarray(aug['image'])
            mask = aug['mask']

        else:
            img = Image.fromarray(img)

        t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])
        img = t(img)

        mask = torch.from_numpy(mask).long()
        return img, mask

        

def get_data_loader():
    """Function to return Data Loader"""
    X = list(os.listdir('./semantic_drone_dataset/original_images'))
    X = [i.strip('.jpg') for i in X]
    dataset = DroneDataset(img_path='./semantic_drone_dataset/original_images', mask_path='./semantic_drone_dataset/label_images_semantic',
                           X = X, mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])

    train_loader = DataLoader(dataset, batch_size=hyperparmeters.batch_size)
    return train_loader























