# Utilities Module
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
from torchvision import transforms as T
from PIL import Image

class Encoder(nn.Module):
    def __init__(self,input_features, output_features, blocks = 2,
                 drop_rate = 0.5):
        """
        :param input_features: number of Input_features
        :param output_features: number of Output Features
        :param blocks: number of Blocks
        :param drop_rate:dropout Rate
        """
        super(Encoder, self).__init__()
        layers = [nn.Conv2d(input_features,output_features,3,1,1),
                  nn.BatchNorm2d(output_features),
                  nn.ReLU(inplace=True)]

        if blocks > 1:
            layers += [nn.Conv2d(input_features, output_features, 3,1,1),
                       nn.BatchNorm2d(output_features),
                       nn.ReLU(inplace=True)]
            if blocks == 3:
                layers += [nn.Dropout(drop_rate)]

        self.features = nn.Sequential(*layers)

    def forward(self, x):
        output = self.features(x)
        return F.max_pool2d(output, 2, 2, return_indices = True), output.size()




class Decoder(nn.Module):
    def __init__(self, input_features, output_features, blocks = 2, drop_rate = 0.5):
        """

        :param input_features: number of Input Features
        :param output_features: number of Output Features
        :param blocks: number of Blocks
        :param drop_rate:Dropout Rate
        """
        super(Decoder, self).__init__()
        layers = [nn.Conv2d(input_features, output_features, 3, 1, 1),
                  nn.BatchNorm2d(output_features),
                  nn.ReLU(inplace=True)]

        if blocks > 1:
            layers += [nn.Conv2d(input_features, output_features, 3,1,1),
                       nn.BatchNorm2d(output_features),
                       nn.ReLU(inplace=True)]
            if blocks == 3:
                layers += [nn.Dropout(drop_rate)]

        self.features(*nn.Sequential())

    def forward(self, x, indices, size):
        unpooled = F.max_unpool2d(x, indices, 2,2,0,size)
        return self.features(unpooled)



class Segnet(nn.Module):
    def __init__(self, num_classes, input_features, drop_rate = 0.5):
        """

        :param num_classes: output classes
        :param input_features: input features
        :param drop_rate: dropout regularization
        """
        super(Segnet, self).__init__()
        # Encoder Layer
        self.encoder_1 = Encoder(input_features = input_features, output_features=64, blocks=2)
        self.encoder_2 = Encoder(input_features = 64, output_features=128, blocks = 2 )
        self.encoder_3 = Encoder(input_features = 128, output_features= 256, blocks = 3)
        self.encoder_4 = Encoder(input_features = 256, output_features = 512, blocks = 3)
        self.encoder_5 = Encoder(input_features = 512, output_features= 512, blocks = 3)


        # Decoder Layers
        self.decoder_1 = Decoder(input_features = 512, output_features = 512, blocks = 3)
        self.decoder_2 = Decoder(input_features = 512, output_features= 256, blocks = 3)
        self.decoder_3 = Decoder(input_features = 256, output_features = 128, blocks = 3)
        self.decoder_4 = Decoder(input_features = 128, output_features = 64, blocks = 2)
        self.decoder_5 = Decoder(input_features = 64, output_features = 64, blocks = 1)

        # final classifiers
        self.classifier = nn.Conv2d(64,num_classes,3,1,1)

    def forward(self, x):
        indices = []
        unpool_sizes = []

        #Encoder Layers
        (x, ind), size = self.encoder_1(x)
        indices.append(ind)
        unpool_sizes.append(size)

        (x, ind), size = self.encoder_2(x)
        indices.append(ind)
        unpool_sizes.append(size)

        (x, ind), size = self.encoder_3(x)
        indices.append(ind)
        unpool_sizes.append(size)

        (x, ind), size = self.encoder_4(x)
        indices.append(ind)
        unpool_sizes.append(size)

        (x,ind), size = self.encoder_5(x)
        indices.append(ind)
        unpool_sizes.append(size)

        #Decoder Layers
        indices = indices.reverse()
        unpool_sizes = unpool_sizes.reverse()

        x = self.decoder_1(x,indices[0], unpool_sizes[0])
        x = self.decoder_2(x, indices[1], unpool_sizes[1])
        x = self.decoder_3(x, indices[2], unpool_sizes[2])
        x = self.decoder_4(x, indices[3], unpool_sizes[3])
        x = self.decoder_5(x, indices[4], unpool_sizes[4])

        x = self.classifier(x)
        return x


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
        img = cv2.imread(self.image_path + self.X[idx] + '.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_path + self.X[idx] + '.png', cv2.IMREAD_GRAYSCALE)
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
























