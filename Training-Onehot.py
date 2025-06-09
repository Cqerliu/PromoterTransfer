'''

PromoterTransfer: This program is used for transfer learning of promoter biological gene sequences in fasta format.

'''
from shutil import copy, rmtree
import random
import numpy as np
from PIL import Image
from Bio import SeqIO
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import json
import matplotlib.pyplot as plt
import os
import torch.optim as optim
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import pandas as pd
import datetime
import shutil

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    # The block parameters here represent different residual structures
    def __init__(self, block, blocks_num, num_classes=2, include_top=True):
        super(ResNet, self).__init__()
        # This parameter is for building a more complex network based on ResNet in the future
        self.include_top = include_top
        # in_channel indicates the depth of the input feature matrix
        self.in_channel = 64

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def resnet18(num_classes=2, include_top=True):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top)


# This function is used to calculate the number of four original indicators such as the true positive rate
def count(y_true, y_pre):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    if len(y_true) != len(y_pre):
        return print("Error!")
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pre[i] == 1:
            TP = TP + 1
        elif y_true[i] == 0 and y_pre[i] == 0:
            TN = TN + 1
        elif y_true[i] == 1 and y_pre[i] == 0:
            FN = FN + 1
        elif y_true[i] == 0 and y_pre[i] == 1:
            FP = FP + 1
        else:
            return print('error')
    return TP, TN, FP, FN

# Convert DNA to one-hot encoded matrix
def DNAmarix(seq):
    li = []
    for i in str(seq.seq):
        if i == 'A':
            li.append([0, 0, 0, 255])
        elif i == 'T':
            li.append([0, 0, 255, 0])
        elif i == 'G':
            li.append([0, 255, 0, 0])
        elif i == 'C':
            li.append([255, 0, 0, 0])
        else:
            li.append([0, 0, 0, 0])
    return li


def mk_file(file_path: str):
    if os.path.exists(file_path):
        rmtree(file_path)
    os.makedirs(file_path)


def main():
    folder = 'E.coli'
    negative_file = 'E.coli_negative.txt'
    promoter_file = 'E.coli_promoter.txt'
    batch_size = 16                                  # Set each batch size
    epochs = 100                                     # Set the number of epochs

    root_path = './' + folder + '/'
    input_file_negative = root_path + negative_file  # Fasta format sequence non-promoter file
    input_file_positive = root_path + promoter_file  # Fasta format sequence promoter file

    result_csv = root_path + 'result.csv'
    mean_result = open(result_csv, 'a')
    mean_result.write('AUC,ACC,Precision,MCC,F1,Sensitivity,Specificity,\n')

    '''
     Different ratios of migration predictions, for example, when the ratio value is 1 in the 
     first for loop, training set: validation set = 1:9. 
    '''
    starttime = datetime.datetime.now()
    for ratio in range(9, 10):
        split_rate = (10 - ratio) / 10
        image_root_path = root_path + str(ratio) + '：' + str(10 - ratio)

        image_save_path_positive = os.path.join(image_root_path, 'positive')
        image_save_path_negative = os.path.join(image_root_path, 'negative')
        mk_file(image_save_path_positive)
        mk_file(image_save_path_negative)

        print('Generate one-hot encoded matrix!')
        index = 1
        for myseq in SeqIO.parse(input_file_positive, 'fasta'):
            sequence = DNAmarix(myseq)
            channel = []
            channel.append(sequence)
            channel.append(sequence)
            channel.append(sequence)

            image_matrix = np.array(channel)
            image_matrix = np.transpose(image_matrix)
            image_matrix = np.uint8(image_matrix)
            image = Image.fromarray(image_matrix)
            image.save(image_save_path_positive + '/' + str(index) + '.tif')
            index += 1

        index = 1
        for myseq in SeqIO.parse(input_file_negative, 'fasta'):
            sequence = DNAmarix(myseq)
            channel = []
            channel.append(sequence)
            channel.append(sequence)
            channel.append(sequence)

            image_matrix = np.array(channel)
            image_matrix = np.transpose(image_matrix)
            image_matrix = np.uint8(image_matrix)
            image = Image.fromarray(image_matrix)
            image.save(image_save_path_negative + '/' + str(index) + '.tif')
            index += 1

        print("The one-hot encoding matrix is saved!")

        # Set random seed
        random.seed(0)

        # Divide the dataset
        data_root = image_root_path
        assert os.path.exists(image_root_path)
        flower_class = [cla for cla in os.listdir(image_root_path)
                        if os.path.isdir(os.path.join(image_root_path, cla))]

        # Create a folder to save the training set
        train_root = os.path.join(data_root, "train")
        mk_file(train_root)
        for cla in flower_class:
            # Create folders for each category
            mk_file(os.path.join(train_root, cla))

        # Create a folder to save the validation set
        val_root = os.path.join(data_root, "val")
        mk_file(val_root)
        for cla in flower_class:
            # Create folders for each category
            mk_file(os.path.join(val_root, cla))

        for cla in flower_class:
            cla_path = os.path.join(image_root_path, cla)
            images = os.listdir(cla_path)
            num = len(images)
            # Index of the randomly sampled validation set
            eval_index = random.sample(images, k=int(num * split_rate))
            for index, image in enumerate(images):
                if image in eval_index:
                    # Copy the files assigned to the validation set to the appropriate directory
                    image_path = os.path.join(cla_path, image)
                    new_path = os.path.join(val_root, cla)
                    copy(image_path, new_path)
                else:
                    # Copy the files assigned to the training set to the appropriate directory
                    image_path = os.path.join(cla_path, image)
                    new_path = os.path.join(train_root, cla)
                    copy(image_path, new_path)
                print("\r[{}] processing [{}/{}]".format(cla, index + 1, num), end="")  # processing bar
            print()

        deldir = image_root_path + '/negative'
        dellist = os.listdir(deldir)
        for f in dellist:
            filepath = os.path.join(deldir, f)
            if os.path.isfile(filepath):
                os.remove(filepath)
            elif os.path.isdir(filepath):
                shutil.rmtree(filepath, True)
        shutil.rmtree(deldir, True)
        deldir = image_root_path + '/positive'
        dellist = os.listdir(deldir)
        for f in dellist:
            filepath = os.path.join(deldir, f)
            if os.path.isfile(filepath):
                os.remove(filepath)
            elif os.path.isdir(filepath):
                shutil.rmtree(filepath, True)
        shutil.rmtree(deldir, True)

        print("The dataset is divided!")

        # Model training starts
        print("Training begin!")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Using {} device.".format(device))

        data_transform = {
            "train": transforms.Compose([
                transforms.Resize(4),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
            "val": transforms.Compose([transforms.Resize(4),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
        image_path = image_root_path

        assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
        train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                             transform=data_transform["train"])
        train_num = len(train_dataset)

        # {'negative':0, 'positive':1}
        flower_list = train_dataset.class_to_idx
        cla_dict = dict((val, key) for key, val in flower_list.items())
        # write dict into json file
        json_str = json.dumps(cla_dict, indent=4)
        with open(image_root_path + '/class_indices.json', 'w') as json_file:
            json_file.write(json_str)

        nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
        print('Using {} dataloader workers every process'.format(nw))

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size, shuffle=True,
                                                   num_workers=nw)

        validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                                transform=data_transform["val"])
        val_num = len(validate_dataset)
        validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                      batch_size=batch_size, shuffle=False,
                                                      num_workers=nw)

        print("Using {} images for training, {} images for validation.".format(train_num,
                                                                               val_num))

        # Model loading
        net = resnet18()

        '''
        In the code commented below, the pretraining process annotates the three lines below. 
        When fine-tuning in the transfer learning process, open the following three lines of 
        comments and modify the path of the pre-trained model. 
        '''
        model_weight_path = "./promoter.pth"
        assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
        net.load_state_dict(torch.load(model_weight_path), False)

        # Define the optimizer and loss function
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.0001)

        best_auc = 0.0
        save_path = image_root_path + '/promoter.pth'

        acc_list = []  # This acc list is used for drawing
        auc_list = []  # This auc list is used for drawing

        # Save data for each epochs
        results_csv = image_root_path + '/' + 'data.csv'
        results = open(results_csv, 'a')
        results.write('AUC,ACC,Precision,MCC,F1,Sensitivity,Specificity\n')

        # Save data for each epochs
        for epoch in range(epochs):
            # train
            net.train()
            running_loss = 0.0
            for step, data in enumerate(train_loader, start=0):
                images, labels = data
                optimizer.zero_grad()
                logits = net(images.to(device))
                loss = loss_function(logits, labels.to(device))
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                # print train process
                rate = (step + 1) / len(train_loader)
                a = "*" * int(rate * 50)
                b = "." * int((1 - rate) * 50)
                print("\rtrain loss: {:^3.0f}%[{}->{}]{:.4f}".format(int(rate * 100), a, b, loss), end="")
            print()

            # validate
            net.eval()
            acc = 0.0
            labels = []
            predicts = []
            y_score = []
            with torch.no_grad():
                for val_data in validate_loader:
                    val_images, val_labels = val_data
                    labels.extend(val_labels.numpy().tolist())
                    outputs = net(val_images.to(device))  # eval model only have last output layer
                    outputs = F.softmax(outputs, dim=1)
                    predict_y = torch.max(outputs, dim=1)[1]
                    y_score.extend(outputs[:, 1].numpy().tolist())
                    predicts.extend(
                        predict_y.numpy().tolist())
                    acc += (predict_y == val_labels.to(device)).sum().item()

                # Calculation of each indicator value
                val_accurate = (acc / val_num)
                val_AUC = roc_auc_score(np.array(labels), np.array(y_score))
                val_precision = precision_score(np.array(labels), np.array(predicts))
                val_f1 = f1_score(np.array(labels), np.array(predicts))

                '''
                Save the optimal pre-training model, open the annotation during pre-training, 
                and annotate the following if statement during fine-tuning
                '''
                if ratio > 8:
                    if val_AUC > best_auc:
                        best_auc = val_AUC
                        torch.save(net.state_dict(), save_path)

                TP, TN, FP, FN = count(labels, predicts)
                MCC = float(TP * TN - FP * FN) / ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))**0.5
                TPR = TP / (TP + FN)
                TNR = TN / (TN + FP)

                print('[epoch %d]  '
                      'train_loss: %.3f  '
                      'AUC: %.3f  '
                      'ACC: %.3f  '
                      'Precision: %.3f  '
                      'MCC: %.3f  '
                      'F1: %.3f  '
                      'Sensitivity: %.3f  '
                      'Specificity: %.3f' %
                      (epoch + 1,
                       running_loss / step,
                       val_AUC,
                       val_accurate,
                       val_precision,
                       MCC,
                       val_f1,
                       TPR,
                       TNR))

                results.write(str(val_AUC) + ',' +
                              str(val_accurate) + ',' +
                              str(val_precision) + ',' +
                              str(MCC) + ',' +
                              str(val_f1) + ',' +
                              str(TPR) + ',' +
                              str(TNR) + '\n')

            acc_list.append(val_accurate)
            auc_list.append(val_AUC)
        results.close()

        df = pd.read_csv(results_csv)
        data = df.mean()
        for i in range(len(data)):
            mean_result.write(str(data.iloc[i]) + ',')
        mean_result.write('\n')

        # Plot AUC/Acc curve
        x = np.linspace(1, len(acc_list), len(acc_list))
        plt.plot(x, acc_list, label='Acc', ls='-', lw=1, marker='o')
        plt.plot(x, auc_list, label='AUC', ls='-', lw=1, marker='^')
        plt.title('Ratio=' + str(ratio) + ':' + str(10 - ratio))
        plt.xlabel('Epochs')
        plt.ylabel('Acc/AUC')
        plt.savefig(image_root_path + '/plot.png')
        plt.legend()
        plt.show()
        print('Training over！')

        # Delete the training set and validation set data generated in the middle
        deldir = image_root_path + '/train'
        dellist = os.listdir(deldir)
        for f in dellist:
            filepath = os.path.join(deldir, f)
            if os.path.isfile(filepath):
                os.remove(filepath)
            elif os.path.isdir(filepath):
                shutil.rmtree(filepath, True)
        shutil.rmtree(deldir, True)
        deldir = image_root_path + '/val'
        dellist = os.listdir(deldir)
        for f in dellist:
            filepath = os.path.join(deldir, f)
            if os.path.isfile(filepath):
                os.remove(filepath)
            elif os.path.isdir(filepath):
                shutil.rmtree(filepath, True)
        shutil.rmtree(deldir, True)

    endtime = datetime.datetime.now()
    lengthtime = (endtime - starttime).seconds
    mean_result.write("The time is:" + str(endtime - starttime) + "," + str(lengthtime) + "s")
    mean_result.close()
    print("The length of time is (in days) ：", endtime - starttime)
    print('All of over!')

if __name__ == '__main__':
    main()