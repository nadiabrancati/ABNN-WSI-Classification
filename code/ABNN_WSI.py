"""
Created on July 2021

@author: Nadia Brancati

"""
import argparse
import numpy as np
from TensorDataset import *
from torch.utils.data.sampler import SubsetRandomSampler
from modelsMinMax import *
import torch.optim as optim
import random
import torchvision
from sklearn.metrics import *

#Loaders for datasets
def loaders(data_dir, val_dir, test_dir,batch_size=1,random_seed=0,shuffle=True, extension='svs'):

    dataset = TensorDataset(data_dir,extension) #train dataset
    dataset_val= TensorDataset(val_dir, extension) #validation dataset
    dataset_test = TensorDataset(test_dir, extension) #test dataset


    num_sample = len(dataset) #number of samples in train set
    num_classes = len(dataset.classes) #number of classes
    indices = list(range(num_sample)) #indices of samples in train set

    num_sample_val = len(dataset_val) #number of samples in validation set
    indices_val=list(range(num_sample_val)) #indices of samples in validation set

    num_sample_test = len(dataset_test) #number of samples in test set
    indices_test = list(range(num_sample_test)) #indices of samples in test set

    #shuffle of the samples
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        np.random.shuffle(indices_val)
        np.random.shuffle(indices_test)

    train_sampler = SubsetRandomSampler(indices) #Samples elements randomly in train set
    valid_sampler = SubsetRandomSampler(indices_val) #Samples elements randomly in validation set
    test_sampler = SubsetRandomSampler(indices_test) #Samples elements randomly in test set

    #train dataset loader
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler
    )

    # validation dataset loader
    valid_loader = torch.utils.data.DataLoader(
        dataset_val, batch_size=1, sampler=valid_sampler
    )

    # test dataset loader
    test_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler
    )


    return (num_classes,train_loader,valid_loader, test_loader)

#Creation of 3D tensors starting from images
def tensors_creation(model_image, args, device):
    #load Image Dataset
    dset = ImageDataset(root_dir=args.data_dir, patch=args.patch_size, scale=args.patch_scale, overlap=0,
                        device=torch.device("cpu"), extension=args.ext)
    #for each image
    for i in range(len(dset)):
        try:
            #return image, label and path of the image
            input, label, filename = dset.__getitem__(i)
            #number of map filters for the output tensor
            num_filters=args.filters_in
            #height and width of the tensor (height image/patch_size and width image/patch_size)
            H = input.shape[0]
            W = input.shape[1]
            #path of the destination tensor file
            file_dst = os.path.join(args.save_dir, filename) + ".pth"
            if not os.path.exists(file_dst):
                control_param=3 #parameter to control the final dimension of the sensor, which has to be at least a dimension of 3x3xnum_filters
                start_H=0
                start_W=0
                #control of tensor size
                if (H < control_param and W < control_param):
                    tensor_U = torch.zeros([control_param, control_param, num_filters], device=device)
                    start_H = 1
                    start_W = 1
                else:
                    if (H<control_param or W<control_param):
                        if H<control_param:
                            tensor_U = torch.zeros([control_param, W, num_filters], device=device)
                            start_H=1
                        if W<control_param:
                            tensor_U = torch.zeros([H, control_param, num_filters], device=device)
                            start_W=1
                    else:
                        tensor_U = torch.zeros([H, W, num_filters], device=device)
                #batch size for the model equal to width of the tensor
                bs = W
                for h in range(0, H):
                    for w in range(0, W, bs):
                        dim = bs
                        if w > W - bs:
                            dim = W - bs
                        batch = input[h][w:w + dim][:][:]
                        #pass the batch of patches to the model
                        ris = model_image(batch.cuda(device=device))
                        ris = torch.squeeze(ris)
                        #insertion of the results of the batch in the final tensor
                        tensor_U[h+start_H, w+start_W:w+start_W + dim] = ris.detach()
                #addition of two new dimension to the tensor for train and test phases
                tensor_U = tensor_U.unsqueeze(0)
                tensor_U = tensor_U.unsqueeze(0)
                if not os.path.exists(file_dst):
                    #saving of the tensor in the destination path
                    torch.save(tensor_U, file_dst)
        except Exception as ex:
            print(ex)
            continue


#Test of the model
def test(model,test_loader,device, dset):
    model.eval()
    true_labels = []
    predicted_labels = []
    #for each test/validation tensor
    for i in range(len(test_loader)):
        #index of the tensor
        k = test_loader.sampler.indices[i]
        try:
            with torch.no_grad():
                #recover tensor information
                ris_model, label, file_name = dset.tensor_and_info(k)
                #permutation of the tensor in order to fit in the model
                ris_model = ris_model.permute(0, 1, 4, 2, 3)
                #output label prediction
                output = model(ris_model.cuda(device=device))
                output = output.unsqueeze(0)
                #comparison between true label and predicted label
                true_labels.append(label.cpu().numpy()[0])
                predicted_labels.append(np.argmax(output.detach().cpu().numpy()[0]))


        except Exception as ex:
            print(ex)
        continue
    #measure of the performance
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision, recall, f1, support = precision_recall_fscore_support(true_labels, predicted_labels)
    conf_mat = confusion_matrix(true_labels, predicted_labels)
    return accuracy, precision, recall, f1, conf_mat

#Training of the model
def train(model, args, device, optimizer, num_epochs, train_loader, valid_loader, test_loader, model_path, model_path_fin, bs):
    #recover of the all datasets
    dset = ImageDataset(root_dir=args.data_dir, patch=args.patch_size, scale=args.patch_scale, overlap=0,
                        device=torch.device("cpu"), extension=args.ext)
    dset_test = ImageDataset(root_dir=args.test_dir, patch=args.patch_size, scale=args.patch_scale,
                             overlap=0,
                             device=torch.device("cpu"), extension=args.ext)
    dset_val = ImageDataset(root_dir=args.val_dir, patch=args.patch_size, scale=args.patch_scale, overlap=0,
                            device=torch.device("cpu"), extension=args.ext)
    #augmentation datasets
    dset_aug = ImageDataset(root_dir=args.aug_dir, patch=args.patch_size, scale=args.patch_scale, overlap=0,
                            device=torch.device("cpu"), extension=args.ext)
    dset_aug2 = ImageDataset(root_dir=args.aug_dir2, patch=args.patch_size, scale=args.patch_scale, overlap=0,
                            device=torch.device("cpu"), extension=args.ext)

    best_epoch = 0
    loss = torch.nn.CrossEntropyLoss()
    #parameter of control for the saving of the final model
    mean_f1=0

    for epoch in range(num_epochs):
        running_samples = 0
        running_losses = 0
        model= (model.train(True))
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        accuracy = 0
        #number of augmentation
        num_op = 12
        global array_aug
        array_aug = []
        #array_aug is the array to store the augmentation already used
        for j in range(len(dset)):
            column = []
            for i in range(num_op):
                column.append(i)
            array_aug.append(column)

        for i in range(num_op):
            #train for a single operation of augmentation
            true_labels, predicted_labels, running_loss, running_sample = train_single_op(model,
                                                                                      train_loader,
                                                                                      bs, optimizer, loss,
                                                                                      dset, device,dset_aug,dset_aug2)
            #calculation of performance for single operation
            running_samples += running_sample
            running_losses += running_loss
            accuracy += accuracy_score(true_labels, predicted_labels)
        print("Actual model obtained at epoch {}, Accuracy={},Loss={}".format(
            str(epoch), accuracy / num_op, running_loss / running_samples))

        #save the current trained model
        model_string = model_path_fin + ".pt"
        torch.save(model.state_dict(), model_string)
        #validate the model
        accuracy, precision, recall, F1, conf_mat = test(model, valid_loader, device, dset_val)
        print(
            "Actual model obtained at epoch {}, Validation/Accuracy={}, Mean(F1)={}, CONF={}".format(
                str(epoch), accuracy, np.mean(F1), conf_mat))

        #if mean of F1 (for validation set) is higher than F1 previously computated, save the new model
        if (np.mean(F1)) >=mean_f1:
            model_string = model_path + ".pt"
            torch.save(model.state_dict(), model_string)
            mean_f1 = np.mean(F1)
            best_epoch = epoch
            print("MODEL SAVED!!! Actual best model obtained at epoch {}, Accuracy={}, Mean(F1)={}".format(
                str(best_epoch), accuracy, np.mean(F1)))
        #test the model
        accuracy, precision, recall, F1, conf_mat = test(model, test_loader, device,
                                                                     dset_test)

        print("Actual model obtained at epoch {}, Test/Accuracy={} Mean(F1)={},CONF={}".format(
            str(epoch), accuracy, np.mean(F1), conf_mat))

    return best_epoch, np.mean(F1)
#Train a single augmentation operation
def train_single_op(model, train_loader, bs, optimizer, loss, dset, device,dset_aug,dset_aug2):
    true_labels = []
    predicted_labels = []
    running_loss = 0.0
    running_samples = 0
    j = 0
    outputs = None
    labels = None
    optimizer.zero_grad()
    shift = 3
    for i in range(len(train_loader)):
        k = train_loader.sampler.indices[i]
        j += 1
        try:
            ris_model, label, file_name = dset.tensor_and_info(k)
            flag = 0
            num_op = 12
            operation = np.random.choice(num_op)
            while flag == 0:
                if array_aug[k][operation] < num_op + 1:
                    array_aug[k][operation] = num_op + 1
                    flag = 1
                else:
                    operation = np.random.choice(num_op)
            # rotation of 90° and flip along axis 1
            if operation == 1:
                ris_model = torch.squeeze(ris_model, 0)
                ris_model = torch.squeeze(ris_model, 0)
                ris_model = ris_model.transpose(0, 1).flip(1)
                ris_model = ris_model.unsqueeze(0)
                ris_model = ris_model.unsqueeze(0)
            #rotation of 90°
            if operation == 2:
                ris_model = torch.squeeze(ris_model, 0)
                ris_model = torch.squeeze(ris_model, 0)
                ris_model = ris_model.transpose(0, 1)
                ris_model = ris_model.unsqueeze(0)
                ris_model = ris_model.unsqueeze(0)
            #rotation of 270°
            if operation == 3:
                ris_model = torch.squeeze(ris_model, 0)
                ris_model = torch.squeeze(ris_model, 0)
                ris_model = ris_model.transpose(0, 1).flip(0)
                ris_model = ris_model.unsqueeze(0)
                ris_model = ris_model.unsqueeze(0)
            # flip along the axis 0
            if operation == 4:
                ris_model = torch.squeeze(ris_model, 0)
                ris_model = torch.squeeze(ris_model, 0)
                ris_model = ris_model.flip(0)
                ris_model = ris_model.unsqueeze(0)
                ris_model = ris_model.unsqueeze(0)
            # flip along the axis 1
            if operation == 5:
                ris_model = torch.squeeze(ris_model, 0)
                ris_model = torch.squeeze(ris_model, 0)
                ris_model = ris_model.flip(1)
                ris_model = ris_model.unsqueeze(0)
                ris_model = ris_model.unsqueeze(0)
            #translation to the right of 'shift' pixels
            if operation == 6:
                image = torch.zeros(ris_model.shape)
                image[:, :, shift:ris_model.shape[2], :, :] = ris_model[:, :, 0:ris_model.shape[2] - shift,
                                                              :, :]
                ris_model = image
            # downward translation of 'shift' pixels
            if operation == 7:
                image = torch.zeros(ris_model.shape)
                image[:, :, :, shift:ris_model.shape[3], :] = ris_model[:, :, :,
                                                              0:ris_model.shape[3] - shift, :]
                ris_model = image
            # translation to the left of 'shift' pixels
            if operation == 8:
                image = torch.zeros(ris_model.shape)
                image[:, :, 0:ris_model.shape[2] - shift, :, :] = ris_model[:, :, shift:ris_model.shape[2],
                                                                  :, :]
                ris_model = image
            # upward translation of 'shift' pixels
            if operation == 9:
                image = torch.zeros(ris_model.shape)
                image[:, :, :, 0:ris_model.shape[3] - shift, :] = ris_model[:, :, :,
                                                                  shift:ris_model.shape[3], :]
                ris_model = image
            #augmented dataset number one (zoom out 1)
            if operation == 10:
                ris_model, label, file_name = dset_aug.tensor_and_info(k)
            # augmented dataset number two (zoom out 2)
            if operation == 11:
                ris_model, label, file_name = dset_aug2.tensor_and_info(k)
            #permutation of the tensor in order to fit in the model
            ris_model = ris_model.permute(0, 1, 4, 2, 3)
            #calculation of the predicted label
            output = model(ris_model.cuda(device=device))
            output = output.unsqueeze(0)
            if outputs is None:
                outputs = output
            else:
                outputs = torch.cat((outputs, output))
            if labels is None:
                labels = label.cuda(device=device)
            else:
                labels = torch.cat((labels, label.cuda(device=device)))
            #for loss calculation
            if j >= bs:
                j = 0
                ris = loss(outputs, labels)
                true_labels.extend(labels.cpu().numpy())
                predicted_labels.extend(np.argmax(outputs.detach().cpu().numpy(), axis=1))
                running_samples += len(labels)
                running_loss += ris.cpu().detach().numpy()
                optimizer.zero_grad()
                ris.backward()
                optimizer.step()
                outputs = None
                labels = None

        except Exception as ex:
            print(ex)
        continue
    return true_labels, predicted_labels, running_loss, running_samples

def main(args):
    #inizializaiont of the environment and set of seed
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_list
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    shuffle=True
    #if mode is TRAIN
    if args.mode == "TRAIN":
        (num_classes, train_loader, valid_loader, test_loader) = loaders(data_dir=args.data_dir,
                                                                         val_dir=args.val_dir,
                                                                         test_dir=args.test_dir,
                                                                         batch_size=1,
                                                                         random_seed=seed,
                                                                         shuffle=shuffle,
                                                                         extension=args.ext)


        #load of the model
        model = AttentionModel(num_classes=num_classes, filters_out=args.filters_out, filters_in=args.filters_in, dropout=args.dropout,
                                device=device).to(device=device)
        #train all parameters of the model
        for param in model.parameters():
            param.requires_grad = True
        #Adam optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate,  weight_decay=1e-3)
        #train and test the model
        epoch, F1 = train(model, args, device, optimizer,
                               args.num_epoch, train_loader,
                               valid_loader, test_loader, args.model_path,
                              args.model_path_fin, args.batch_size)
        print("Best model obtained at: {} with F1 = {}".format(str(epoch), F1))


    #if mode is TENSOR
    if args.mode == "TENSOR":

        model=None
        # if the model chosen for the creation of the tensor is RESNET18
        if args.model_type == "RESNET18":
            #if the original pretained model is chosen
            if args.model_pretrained:
                model = torchvision.models.resnet18(pretrained=True)
            # if a new pretained model is chosen
            else:
                model = torchvision.models.resnet18()
                model.fc = nn.Linear(512, 3)
                model.load_state_dict(torch.load(args.model_path))
            #parameters are not trained
            for param in model.parameters():
                param.requires_grad = False
            #last layer for the classification is deleted: only features are extracted to create the tensor
            model = torch.nn.Sequential(*(list(model.children())[:-1]))
            model.cuda(device=device)
        # if the model chosen for the creation of the tensor is RESNET34
        if args.model_type == "RESNET34":
            # if the original pretained model is chosen
            if args.model_pretrained:
                model = torchvision.models.resnet34(pretrained=True)
            # if a new pretrained model is chosen
            else:
                model = torchvision.models.resnet34()
                model.fc = nn.Linear(512, 3)
                model.load_state_dict(torch.load(args.model_path))
            # parameters are not trained
            for param in model.parameters():
                param.requires_grad = False
            # last layer for the classification is deleted: only features are extracted to create the tensor
            model = torch.nn.Sequential(*(list(model.children())[:-1]))
            model.cuda(device=device)
        #creation of the tensors
        tensors_creation(model,args,device=device)
    print("END")



if __name__ == '__main__':
    MODEL_PATH = "path-to-save/load-the-model"
    DATA_DIR = "path-for-loading-images/tensors"
    SAVE_DIR = "path-to-save-tensors"
    parser = argparse.ArgumentParser(description='Training a model')

    # General parameters
    parser.add_argument('--model_type', choices=['RESNET18','RESNET34'],default="RESNET34",help="Models used to create the Tensor_U [RESNET18,RESNET34] ")
    parser.add_argument('--model_pretrained', help='if original pretrained model this parameter should be set to True')
    parser.add_argument('--model_path', default=MODEL_PATH, help='path of the model saved for each epoch')
    parser.add_argument('--model_path_fin', default=MODEL_PATH, help='path of the final saved model')
    parser.add_argument('--data_dir',default=DATA_DIR, help='path of the train dataset')
    parser.add_argument('--val_dir', default=DATA_DIR, help='path of the validation dataset')
    parser.add_argument('--test_dir', default=DATA_DIR, help='path of the test dataset')
    parser.add_argument('--aug_dir', default=DATA_DIR, help='path of the first dataset for the augmentation')
    parser.add_argument('--aug_dir2', default=DATA_DIR, help='path of the second dataset for the augmentation')
    parser.add_argument('--save_dir', default=SAVE_DIR, help='path of the directory where tensors will be saved')
    parser.add_argument('--mode', choices=['TRAIN','TENSOR'], default="TRAIN", help="possible options: TRAIN and TENSOR")
    parser.add_argument('--seed', type=int, default=1, help='Seed value')
    parser.add_argument('--gpu_list', default="0", help='number of the GPU that will be used')
    parser.add_argument('--debug', action='store_true', help='for debug mode')
    parser.add_argument('--ext', default='pth', help='extension of the structure to load: svs/png for images (mode=TENSORS) and pth for tensors (mode=TRAIN)')

    # Training parameters
    parser.add_argument('--patch_size', type=int, default=224, help='Patch Size')
    parser.add_argument('--patch_scale', type=int, default=224, help='Patch Scale')
    parser.add_argument('--num_epoch', type=int, default=100, help='max epoch')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    
    # Model parameters
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--filters_out', type=int, default=64, help='number of Attention Map Filters')
    parser.add_argument('--filters_in', type=int, default=512, help='number of Input Map Filters')

    args = parser.parse_args()
    main(args)
