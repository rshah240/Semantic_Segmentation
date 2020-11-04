import torch
import torch.optim as optim
import torch.nn as nn
from utils import Segnet
import utils

import hyperparmeters
import os
from torch.utils.tensorboard import SummaryWriter

train_on_gpu = torch.cuda.is_available()
def train(retraining = False):
    """Training Function for training Segnet Architecture"""
    PATH = './Segnet_2.pth'
    writer = SummaryWriter("logs/Semantic_Segmentation_Experiment_1")
    model = Segnet()

    if retraining:
        # Training the model from the last checkpoint
        model_config = torch.load(PATH)
        model.load_state_dict(model_config['state_dict'])
    else:
        # Initialising Encoder with vgg16 parameters
        model.init_vgg16_params()
    train_loader = utils.get_data_loader()
    if train_on_gpu:
        model.cuda()
    Encoder_Layers = [model.encoder1, model.encoder2, model.encoder3, model.encoder4, model.encoder5]

    # Freezing Encoder Layers
    for i in Encoder_Layers:
        for param in i.parameters():
            param.requires_grad = False


    optimizer = optim.Adam(model.parameters(), hyperparmeters.lr)
    criterion = nn.CrossEntropyLoss()
    epochs = hyperparmeters.epochs
    losses = []
    images,masks = next(iter(train_loader))
    images = images.cuda()
    writer.add_graph(model,images)
    for i in range(epochs):
        model.train()
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
        writer.add_scalar('Training_loss',sum(losses)/((i+1)*(400/hyperparmeters.batch_size)),(i+1))
        if i % 10 == 0:
            print("Epochs : {}/{} Loss: {:.2f}".format(i,epochs,loss.item()))
    writer.close()
    print("Saving Model")


    model_config = {'state_dict': model.state_dict(),'epochs': epochs,'losses': losses}
    torch.save(model_config, PATH)
    print("Finished Training")


if __name__ == "__main__":
    train(False)










