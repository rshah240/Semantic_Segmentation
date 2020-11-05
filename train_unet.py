import torch
import torch.optim as optim
import torch.nn as nn
from utils import UNet
import utils
import argparse
import hyperparmeters
from torch.utils.tensorboard import SummaryWriter


train_on_gpu = torch.cuda.is_available()
def train(retraining = False):
    """Training Function for training Segnet Architecture"""
    PATH = './Unet_1.pth'
    writer = SummaryWriter("logs/Semantic_Segmentation_Experiment_2")
    model = UNet(num_classes=23)

    if retraining:
        # Training the model from the last checkpoint
        model_config = torch.load(PATH)
        model.load_state_dict(model_config['state_dict'])
    else:
        # Initialising Weights of the models
        model.initialize_weights()
    train_loader = utils.get_data_loader()
    if train_on_gpu:
        model.cuda()

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--retraining',type=bool, required=False, default=False, help='Bool Value to retrain the model '
                                                                                      'or not')
    args = parser.parse_args()
    retraining = args.retraining
    train(retraining)
