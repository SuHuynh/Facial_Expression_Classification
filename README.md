# Facial_Expression_Classification
This repo is for classify emotion of human face

## Data preparation
- Using the CK48+ dataset. It contains 1908 facial images with 7 facial emotions: anger, contempt, disgust, fear, happy, sad and surprise.
- Link [download](https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch/tree/master/CK%2B48) dataset
- Extract and copy to the folder ./dataset follow the below folder tree:
<pre>
Facial_Expression_Classification
└── dataset
    ├── anger
    ├── contempt
    ├── disgust
    ├── fear
    ├── happy
    ├── sadness
    └── surprise
</pre>
- run ```python create_file_csv.py``` to create 2 files ```data_train.csv``` and ```data_test.csv```. These files contain all paths to training images and testing images.
## Model
- We use the VGG16 to train this task. The model is defined in ```vgg16.py```
## Training
- To train the model, run ```python train.py```
- The checkpoints will saved in folder ```./saved_models``` after every epoch
## Testing
- Modify the path of testing image you want in ```test.py```. Then, run ```python test.py``` and see the results
