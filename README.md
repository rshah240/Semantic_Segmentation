# Semantic Segmentation
Semantic Segmentation on images captured by drone. 
Implementation of Segnet and Unet Architecture using pytorch.


## Usage
To train the model Unet or Segnet Model.

```bash
python train_unet.py --retraining=False
python train_segnet.py --retraining=False
```
## Requirements
pytorch

matplotlib

cv2

tensorboard



## Dataset Download
The Dataset has been taken from a Kaggle Competition

[Click here to download the data](https://www.kaggle.com/bulentsiyah/semantic-drone-dataset)


## Tensorboard Instance
To run the tensorboard instance
```bash
tensorboard --logdir=logs
```

## Sample Outputs
Segnet output


![Output Segnet](https://github.com/rshah240/Semantic_Segmentation/blob/master/output_images/Segnet_1.png)






Unet Output 



![Output Unet](https://github.com/rshah240/Semantic_Segmentation/blob/master/output_images/Unet_1.png)




## Download the Trained Models

[Link to download the Segnet Model](https://drive.google.com/file/d/129u46DgX7Gf5HzOD-udT-g_gkFsB_Fcz/view?usp=sharing)


[Link to download the Unet Model](https://drive.google.com/file/d/1cVkN6GkgQogjwJS2IVX7pmfoFBKco31K/view?usp=sharing)


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
