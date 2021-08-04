# WSI-Classification
This repository contains the code to reproduce results of the [Gigapixel Histopathological Image Analysis Using
Attention-Based Neural Networks](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9447746) paper.

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
  --filters_out FILTERS_OUT
                        number of Attention Map Filters
  --filters_in FILTERS_IN
                        number of Input Map Filters
  --dropout DROPOUT     dropout rate
  --gpu_list GPU_LIST   number of the GPU that will be used
  --debug               for debug mode
  --ext EXT             extension of the structure to load: png for images
                        (mode=TENSORS) and pth for tensors (mode=TRAIN)```
