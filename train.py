import torch
import torch.optim as optim
import torch.nn as nn
from utils import Segnet, DroneDataset
from torch.utils.data import DataLoader
import hyperparmeters
import os

train_on_gpu = torch.cuda.is_available()

def get_data_loader():
    X = list(os.listdir('./semantic_drone_dataset/original_images'))
    X = [i.strip('.jpg') for i in X]
    dataset = DroneDataset(img_path='./semantic_drone_dataset/original_images', mask_path='./semantic_drone_dataset/label_images_semantic',
                           X = X, mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])

    train_loader = DataLoader(dataset, batch_size=hyperparmeters.batch_size)
    return train_loader

def train():
    model = Segnet()
    if train_on_gpu:
        model.cuda()
    model.init_vgg16_params()
    optimizer = optim.Adam(model.parameters(), hyperparmeters.lr)
    criterion = nn.CrossEntropyLoss()
    train_loader = get_data_loader()
    epochs = hyperparmeters.epochs
    losses = []
    for i in range(epochs):
        for batch_i,(images,masks) in enumerate(train_loader):
            if train_on_gpu:
                images = images.cuda()
                masks = masks.cuda()

            optimizer.zero_grad()
            predicted_mask = model(images)
            loss = criterion(predicted_mask, masks)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        if i % 10 == 0:
            print("Epochs : {}/{} Loss: {.2f}".format(i,epochs, loss.item()))
if __name__ == "__main__":
    train()










