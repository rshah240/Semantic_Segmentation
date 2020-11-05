# Semantic Segmentation
Semantic Segmentation on images captured by drone. 
Implementation is done using pytorch.

## Usage
To train the model Unet or Segnet Model.

```bash
python train_unet.py --retraining=False
python train_segnet.py --retraining=False
```

## Dataset Download
The Dataset has been taken from a Kaggle Competition

[Click here to download the data](https://www.kaggle.com/bulentsiyah/semantic-drone-dataset)


## Tensorboard Instance
To run the tensorboard instance
```bash
tensorboard --logdir=logs
```

## Sample Outputs
![Output Segnet](https://github.com/rshah240/Semantic_Segmentation/blob/master/output_images/Segnet_1.png)







![Output Unet](https://github.com/rshah240/Semantic_Segmentation/blob/master/output_images/Segnet_1.png)

