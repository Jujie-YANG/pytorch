import copy
import json
import os
# https://pytorch.org/docs/stable/torchvision/index.html
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from torch import nn
from torchvision import transforms, models, datasets

# Read data and preprocess data
data_dir = './flower_data/'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'

data_transforms = {
    'train': transforms.Compose([
        # Data Augmentation
        transforms.RandomRotation(45),  # randomly rotate (margin: -45 ~ 45 degree)
        transforms.CenterCrop(224),  # crop starting from the centre, CenterCrop(size=(224, 224))
        transforms.RandomHorizontalFlip(p=0.5),  # randomly horizontal flip with a possibility 50
        transforms.RandomVerticalFlip(p=0.5),
        # transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),  # 亮度,对比度,饱和度,色相
        # transforms.RandomGrayscale(p=0.025),  # 0.025概率转 换成灰度率，3通道就是R=G=B
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # mean, std standard deviation
        # (normalize [R,G,B]), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

batch_size = 8

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in
               ['train', 'valid']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
class_names = image_datasets['train'].classes  # ['1', '10', '100', '101', '102', ...]

# Read real category names of corresponding numbers
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)  # {'1': 'pink primrose', '10': 'globe thistle', '100': 'blanket flower',...}

model_name = 'resnet'  # optional models: ['resnet', 'alexnet', 'vgg', 'squeezenet', 'densenet', 'inception']
# whether use pretrained features of the specific model
feature_extract = True

# Train on GPU or CPU
train_on_gpu = torch.cuda.is_available()
print(train_on_gpu)
if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Use all parameters of the model and do not update that is to eliminate gradient when backtracking
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # choose appropriate models, different models have different initialize methods
    model_ft = None
    input_size = 0

    if model_name == "resnet":

        """ Resnet152
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes),
                                    nn.LogSoftmax(dim=1))
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg16(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


model_ft, input_size = initialize_model("resnet", 102, feature_extract, use_pretrained=True)

# Use GPU to calculate
model_ft = model_ft.to(device)

# save in BNSP model
filename = 'checkpoint.pth'

# whether train all layers
params_to_update = model_ft.parameters()
print("Params to learn:")
if not feature_extract:
    for name, param in model_ft.named_parameters():
        if param.requires_grad:
            print("\t", name)
else:
    params_to_update = []  # only parameters in the final full connected layer need to be updated
    for name, param in model_ft.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
            print("\t", name)  # fc.0.weight fc.0.bias

# Set optimizer
# only update parameters in FC layer, if all parameters, model.params()
optimizer_ft = optim.Adam(params_to_update, lr=1e-2)  # lr=0.01
# learning rate decay policy
scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)  # lr reduce to 1/10 every 7 epoch
# 最后一层已经LogSoftmax()了，所以不能nn.CrossEntropyLoss()来计算了，nn.CrossEntropyLoss()相当于logSoftmax()和nn.NLLLoss()整合
criterion = nn.NLLLoss()


# Define training model
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False, filename=filename):
    since = time.time()
    best_acc = 0
    """
    checkpoint = torch.load(filename)
    best_acc = checkpoint['best_acc']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.class_to_idx = checkpoint['mapping']
    """
    # model need to be put in your CPU or GPU
    model.to(device)

    val_acc_history = []
    train_acc_history = []
    train_losses = []
    valid_losses = []
    LRs = [optimizer.param_groups[0]['lr']]

    # initialize current params, store best weights when training
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # train and valid
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0  # the loss of current epoch 
            running_corrects = 0  # the number of corrects of current epoch

            # Take all data
            for inputs, labels in dataloaders[phase]:  # dataloaders = {'train':... ,'valid':...}
                inputs = inputs.to(device)  # put all data to GPU or CPU
                labels = labels.to(device)

                # clear gradient to zero and then update
                optimizer.zero_grad()
                # only calculate and update gradients when phase=='train'
                with torch.set_grad_enabled(phase == 'train'):
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:  # model.resnet runs here
                        outputs = model(inputs)  # outputs=102
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # update params when phase == 'train '
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()  # finish only one iteration

                # Calculate loss of current iteration (1  epoch = 200 iterations)
                running_loss += loss.item() * inputs.size(0)  # refer to the 'batch' dimension, inputs=batch*h*w
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)  # mean loss
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            time_elapsed = time.time() - since  # how many time a epoch waste
            print('Time elapsed {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # Save best model after testing on validation dataset
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                state = {
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(state, filename)
            if phase == 'valid':
                val_acc_history.append(epoch_acc)
                valid_losses.append(epoch_loss)
                scheduler.step(epoch_loss)
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_losses.append(epoch_loss)

        # after iterating each epoch (for epoch in range(num_epochs):)
        print('Optimizer learning rate : {:.7f}'.format(optimizer.param_groups[0]['lr']))
        LRs.append(optimizer.param_groups[0]['lr'])
        print()
        scheduler.step()  # lr decay

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # after training, use the best saved model as the result
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history, valid_losses, train_losses, LRs


# Start training!
model_ft = models.resnet50()
model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs = train_model(model_ft, dataloaders,
                                                                                            criterion, optimizer_ft,
                                                                                            num_epochs=20,
                                                                                            is_inception=(
                                                                                                    model_name == "inception"))

# Then train all params by setting all gradients to be True
for param in model_ft.parameters():
    param.requires_grad = True

# train all params instead and reduce a little learning rate
optimizer = optim.Adam(params_to_update, lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# Loss function
criterion = nn.NLLLoss()

# Load the checkpoint
checkpoint = torch.load(filename)
best_acc = checkpoint['best_acc']
model_ft.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])
# model_ft.class_to_idx = checkpoint['mapping']

model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs = train_model(model_ft, dataloaders,
                                                                                            criterion, optimizer,
                                                                                            num_epochs=10,
                                                                                            is_inception=(
                                                                                                    model_name == "inception"))

# Load already trained model
model_ft, input_size = initialize_model(model_name, 102, feature_extract, use_pretrained=True)

# GPU mode
model_ft = model_ft.to(device)

#   save the file
filename = 'seriouscheckpoint.pth'

# Load trained model
checkpoint = torch.load(filename)
best_acc = checkpoint['best_acc']
model_ft.load_state_dict(checkpoint['state_dict'])


# Preprocess test dataset
def process_image(image_path):
    # Read test dataset
    img = Image.open(image_path)
    # Do condition statements because Resize & thumbnail methods can only reduce size
    if img.size[0] > img.size[1]:
        img.thumbnail((10000, 256))
    else:
        img.thumbnail((256, 10000))
    # Crop to make sure the size of inputs are the same
    left_margin = (img.width - 224) / 2
    bottom_margin = (img.height - 224) / 2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    img = img.crop((left_margin, bottom_margin, right_margin,
                    top_margin))
    # The same preprocessing methods
    # Use the same mean and std methods as the training dataset, normalize in 0-1
    img = np.array(img) / 255
    mean = np.array([0.485, 0.456, 0.406])  # provided mean
    std = np.array([0.229, 0.224, 0.225])  # provided std
    img = (img - mean) / std

    # the first position is color channel in Pytorch, we need to transpose (0,1,2) -> (2,0,1) = (color,h,w)
    img = img.transpose((2, 0, 1))

    return img


def imshow(image, ax=None, title=None):
    """show data"""
    if ax is None:
        fig, ax = plt.subplots()

    # restore the position of the color channel: (color,h,w) = (0,1,2) -> (1,2,0) =
    image = np.array(image).transpose((1, 2, 0))

    # restore the preprocessing operation
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.set_title(title)

    return ax


image_path = 'image_06621.jpg'
img = process_image(image_path)
imshow(img)

# get a batch size of test data
dataiter = iter(dataloaders['valid'])
images, labels = dataiter.next()

model_ft.eval()
# output.shape = torch.Size([8, 102]) batch=8
# output shows possibilities of each data in 102 classes
if train_on_gpu:
    output = model_ft(images.cuda())
else:
    output = model_ft(images)

# Get the class with the largest possibilities
_, preds_tensor = torch.max(output, 1)

preds = np.squeeze(preds_tensor.numpy()) if not train_on_gpu else np.squeeze(preds_tensor.cpu().numpy())


# preds = array([77, 22, 46, 46, 64, 93, 28, 48], dtype=int64)


# Show prediction results in images
def im_convert(tensor):
    """ show data"""

    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    return image


fig = plt.figure(figsize=(20, 20))
columns = 4
rows = 2

for idx in range(columns * rows):
    ax = fig.add_subplot(rows, columns, idx + 1, xticks=[], yticks=[])
    plt.imshow(im_convert(images[idx]))
    ax.set_title("{} ({})".format(cat_to_name[str(preds[idx])], cat_to_name[str(labels[idx].item())]),
                 color=("green" if cat_to_name[str(preds[idx])] == cat_to_name[str(labels[idx].item())] else "red"))
plt.show()
