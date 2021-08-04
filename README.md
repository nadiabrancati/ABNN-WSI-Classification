# ABNN-WSI-Classification
This repository contains the code to reproduce results of the [Gigapixel Histopathological Image Analysis Using
Attention-Based Neural Networks](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9447746) paper.
The structure of CNN consists in a compressing path and a learning path. In the compressing path, the gigapixel image is packed into a grid-based feature map by using a residual network devoted to the feature extraction of each patch into which the image has been divided. In the learning path, attention modules (Maxpooling and Minpooling) are applied to the grid-based feature map, taking into account spatial correlations of neighboring patch features to find regions of interest, which are then used for the final whole slide classification.
|![Step 1](https://github.com/nadiabrancati/ABNN-WSI-Classification/blob/main/img/method1.png)|
|:--:| 
|*Step 1: Compressing path*|

|![Step 2](https://github.com/nadiabrancati/ABNN-WSI-Classification/blob/main/img/method2.png)|
|:--:| 
|*Step 2: Learning path*|

The experiments are based on Camelyon16 and TUPAC 16 datasets. However, new esperiments have been made by using WSI of [BRACS dataset](https://www.bracs.icar.cnr.it/).
# Installation
The ```requirements.txt``` file should list all Python libraries that the present code depend on, and they will be installed using:

```pip install -r requirements.txt```

# Running the code
The options to run the script "ABNN_WSI.py" containing the main, are:
```Training a model

optional arguments:
  -h, --help            show this help message and exit
  --model_type {RESNET18,RESNET34}
                        Models used to create the Tensor_U [RESNET18,RESNET34]
  --model_pretrained    if original pretrained model this parameter should be
                        set to True
  --model_path MODEL_PATH
                        path of the model saved for each epoch
  --model_path_fin MODEL_PATH_FIN
                        path of the final saved model
  --data_dir DATA_DIR   path of the train dataset
  --val_dir VAL_DIR     path of the validation dataset
  --test_dir TEST_DIR   path of the test dataset
  --aug_dir AUG_DIR     path of the first dataset for the augmentation
  --aug_dir2 AUG_DIR2   path of the second dataset for the augmentation
  --save_dir SAVE_DIR   path of the directory where tensors will be saved
  --mode {TRAIN,TENSOR}
                        possible options: TRAIN and TENSOR
  --seed SEED           Seed value
  --gpu_list GPU_LIST   number of the GPU that will be used
  --debug               for debug mode
  --ext EXT             extension of the structure to load: png for images
                        (mode=TENSORS) and pth for tensors (mode=TRAIN)
  --patch_size PATCH_SIZE
                        Patch Size
  --patch_scale PATCH_SCALE
                        Patch Scale
  --num_epoch NUM_EPOCH
                        max epoch
  --batch_size BATCH_SIZE
                        batch size
  --learning_rate LEARNING_RATE
                        learning rate
  --dropout DROPOUT     dropout rate
  --filters_out FILTERS_OUT
                        number of Attention Map Filters
  --filters_in FILTERS_IN
                        number of Input Map Filters
```
# Step 1: Compressing path
Compressing path to create tensors for the Step 2 can be set by using the parameters ```--mode``` of the script equal to ```TENSORS```. 

An example of usage for compressing path is:

```python ABNN_WSI.py --mode TENSOR --model_type RESNET34 --data_dir path-for-loading-images  --gpu_list 0 --seed 0 --save_dir path-to-save-the-tensors --ext svs```

# Step 2: Learning path
Learning path to train and test the model can be set by using the parameters ```--mode``` of the script equal to ```TRAIN```. 

An example of usage for learning path is:

```python ABNN_WSI.py --mode TRAIN --data_dir path-for-loading-training-tensors --val_dir path-for-loading-validation-tensors --test_dir path-for-loading-test-tensors --aug_dir path-for-loading-some-augmentation-tensors --aug_dir2 path-for-loading-other-augmentation-tensors --gpu_list 0 --seed 0 --model_path path-to-save-model-for-each-epoch --model_path_fin path-to-save-the-final-model --batch_size 16 --learning_rate 0.0001 --ext pth```
