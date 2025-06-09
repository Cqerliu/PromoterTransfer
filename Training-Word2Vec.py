from torchtext.legacy import data
import datetime
import os
import random
from shutil import rmtree
import copy
from Bio import SeqIO
import csv
import numpy as np
import pandas as pd
from gensim.models import word2vec
from torchtext.vocab import Vectors, GloVe
from torch import nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import time
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score




#################################################################################################################
# ResNet network
# Basic network block, where BasicBlock is a subclass inherited from nn.Module
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1,
                 downsample=None):  # in_channel: depth of input feature matrix; out_channel: depth of output feature matrix (number of 3x3 convolution kernels, e.g., 64 in 3x3,64); downsample is None by default, only used for dashed connections
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        # First convolution layer: kernel size 3, stride 1, padding 1
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        # Second convolution layer: kernel size 3, stride 1, padding 1
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:  # downsample is used for dashed connections requiring size reduction
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
    # block: type of residual block (e.g., BasicBlock, BottleNeck); blocks_num: number of blocks in each stage (e.g., [2,2,2,2] for 18-layer); num_classes: number of classification categories
    def __init__(self, vocab_size, pad_idx, block, blocks_num, num_classes=2, include_top=True):
        super(ResNet, self).__init__()

        self.include_top = include_top  # Used for building more complex networks based on ResNet
        self.in_channel = 64  # Depth of input feature matrix, i.e., after 3x3 max pooling, all convolution layers have 64 channels

        # Text embedding layer
        self.embedding = nn.Embedding(vocab_size, 100, padding_idx=pad_idx)
        self.conv = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.in_channel, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # layer1 represents conv2_x, implemented by _make_layer(); similarly for layer2-3
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:  # include_top is True by default
            self.avgpool = nn.AdaptiveAvgPool2d(
                (1, 1))  # Output size = (1, 1): average pooling reduces input to 1x1 regardless of original size
            self.fc = nn.Linear(512 * block.expansion,
                                1)  # Input nodes: flattened result after average pooling; 512 is for 18/34-layer conv5_x (50/101/152-layer use 512*4); output: num_classes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num,
                    stride=1):  # block: residual block type; channel: number of channels in first layer of residual block (e.g., 64 for conv2_x); block_num: number of residual blocks in this layer (e.g., 2 for 18-layer conv2_x)
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

        x = self.embedding(x)
        # Input shape: (32, 200, 100) -> after unsqueeze: (32, 1, 200, 100)
        x = x.unsqueeze(1)
        # After self.conv: (32, 3, 200, 100)
        x = self.conv(x)

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


def resnet18(vocab_size, pad_idx, num_classes=2, include_top=True):
    return ResNet(vocab_size, pad_idx, BasicBlock, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top)


def mk_file(file_path: str):
    if os.path.exists(file_path):
        # Delete existing folder and recreate if it exists
        rmtree(file_path)
    os.makedirs(file_path)

def seqword(seq):
    li = []
    seq_str = str(seq.seq)
    length = len(seq_str)
    # Sliding window
    for index, value in enumerate(seq_str):
        if index + 5 < length:
            li.append(seq_str[index:index+6])

    # # Non-sliding window
    # for index in range(0, length, 6):
    #     li.append(seq_str[index:index+6])
    return li

# Calculate four original metrics (TP, TN, FP, FN)
def count(y_true, y_pre):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    if len(y_true) != len(y_pre):
        return print("Error!")
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pre[i] == 1:
            TP += 1
        elif y_true[i] == 0 and y_pre[i] == 0:
            TN += 1
        elif y_true[i] == 1 and y_pre[i] == 0:
            FN += 1
        elif y_true[i] == 0 and y_pre[i] == 1:
            FP += 1
        else:
            return print('error')
    return TP, TN, FP, FN

def main():

    ################################################### generate
    # words_species_name = "Dro_Genomic"
    #
    # root_path = "./dataset/words/"+words_species_name+"/"
    # bio = root_path + words_species_name +".txt"
    # seq_words = root_path + "seq_words.txt"
    # seq_vectors = root_path + "seq_vectors.txt"
    # with open(seq_words, 'a+', encoding='utf-8') as f:
    #     for myseq in SeqIO.parse(bio, 'fasta'):
    #         sequence = seqword(myseq)
    #         f.write(" ".join(sequence) + '\n')
    #     f.close()
    # sentences = word2vec.Text8Corpus(seq_words)
    # model = word2vec.Word2Vec(sentences=sentences, vector_size=100)
    # model.wv.save_word2vec_format(seq_vectors, binary=False)

    BATCH_SIZE = 16
    EPOCHS = 5
    START = 2
    END = 3

    species_name = "S_cerevisiae"
    corpus_list = ['SD_to_S']
    words_species_name_list = ['Dro_Genomic']

    for index, words_species_name in enumerate(words_species_name_list):
        corpus = corpus_list[index]
        embedding_root_path = './dataset/embedding/' + species_name + '/'
        negative_name = species_name + "_negative"
        positive_name = species_name + "_promoter"

        input_negative = embedding_root_path + negative_name + ".txt"  # FASTA format non-promoter sequence file
        train_data = embedding_root_path + species_name + ".csv"
        with open(train_data, 'a+', encoding='utf-8') as f:
            f.write('text,label\n')
            for myseq in SeqIO.parse(input_negative, 'fasta'):
                sequence = seqword(myseq)
                f.write(" ".join(sequence) + ',' + '0' + '\n')
            f.close()

        input_positive = embedding_root_path + positive_name + ".txt"  # FASTA format promoter sequence file
        train_data = embedding_root_path + species_name + ".csv"
        with open(train_data, 'a+', encoding='utf-8') as f:
            for myseq in SeqIO.parse(input_positive, 'fasta'):
                sequence = seqword(myseq)
                f.write(" ".join(sequence) + ',' + '1' + '\n')
            f.close()

        embedding_path = './dataset/embedding/' + species_name + '/' + corpus
        vectors_root_path = "./dataset/words/" + words_species_name

        avg_results_csv = embedding_path + '/' + 'avg_data.csv'
        avg_results = open(avg_results_csv, 'a')
        avg_results.write('Ratio,AUC,ACC,Precision,MCC,F1,Sensitivity,Specificity\n')

        for rate in range(START, END):
            val_rate = rate / 10
            result_save_path = embedding_path + "/" + str(rate) + '：' + str(10 - rate)  # Image save path
            mk_file(result_save_path)

            results_csv = result_save_path + '/' + 'process_data.csv'
            results = open(results_csv, 'a')
            results.write('AUC,ACC,Precision,MCC,F1,Sensitivity,Specificity\n')

            mytokenize = lambda x: x.split(" ")
            TEXT = data.Field(sequential=True, tokenize=mytokenize,
                              include_lengths=True, use_vocab=True,
                              batch_first=True, fix_length=300)
            LABEL = data.Field(sequential=False, use_vocab=False,
                               pad_token=None, unk_token=None)
            # Process columns of the dataset to be read
            train_test_fields = [
                ("text", TEXT),
                ("label", LABEL)

            ]
            # Read data
            traindata, testdata = data.TabularDataset.splits(
                path=embedding_root_path,
                format="csv",
                train=species_name + ".csv",
                test=species_name + ".csv",
                fields=train_test_fields,
                skip_header=True
            )
            print("Training dataset size:", len(traindata))

            train_data, val_data = traindata.split(split_ratio=val_rate, random_state=random.seed(0), stratified=True, strata_field='label')
            print("Training subset size:", len(train_data), "Validation subset size:", len(val_data))

            vec = Vectors(vectors_root_path + "/seq_vectors.txt", vectors_root_path)   # First argument: path to input vocabulary; second: path to save vocabulary model (.pt file)
            TEXT.build_vocab(train_data, max_size=20000, vectors=vec)
            # print(train_data[0].text)
            LABEL.build_vocab(train_data)

            # Class label distribution
            print("Class label frequencies in training set:", LABEL.vocab.freqs)

            train_iter = data.BucketIterator(train_data, batch_size=BATCH_SIZE)
            val_iter = data.BucketIterator(val_data, batch_size=BATCH_SIZE)


            # Get a batch of data for demonstration
            for step, batch in enumerate(train_iter):
                if step > 0:
                    break

            # Instantiate model with ResNet parameters
            INPUT_DIM = len(TEXT.vocab)  # Vocabulary size
            EMBEDDING_DIM = 100  # Embedding dimension
            PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]  # Index of padding token
            model = resnet18(INPUT_DIM, PAD_IDX)



            ################################################################################################################
            # pretrained_embeddings = TEXT.vocab.vectors
            # model.embedding.weight.data.copy_(pretrained_embeddings)


            # Initialize vectors for unknown ('<unk>') and padding ('<pad>') tokens to zero
            UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
            model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
            model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

            # Adam optimizer and binary cross-entropy with logits loss
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            optimizer = optim.Adam(model.parameters(), lr=0.0001)
            criterion = nn.BCEWithLogitsLoss()
            print("Using {} device.".format(device))

            def train_epoch(model, iterator, optimizer, criterion):
                epoch_loss = 0
                epoch_acc = 0
                train_corrects = 0; train_num = 0
                model.train()
                for batch in iterator:
                    optimizer.zero_grad()
                    pre = model(batch.text[0]).squeeze(1)
                    loss = criterion(pre, batch.label.type(torch.FloatTensor))
                    pre_lab = torch.round(torch.sigmoid(pre))
                    train_corrects += torch.sum(pre_lab.long() == batch.label)
                    train_num += len(batch.label)                                # Number of samples
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                epoch_loss = epoch_loss / train_num
                epoch_acc = train_corrects.double().item() / train_num
                return epoch_loss, epoch_acc

            def evaluate(model, iterator, criterion):
                epoch_loss = 0; epoch_acc = 0
                train_corrects = 0; train_num = 0
                model.eval()
                predicts = []
                labels = []
                scores = []
                with torch.no_grad():
                    for batch in iterator:
                        pre = model(batch.text[0]).squeeze(1)
                        loss = criterion(pre, batch.label.type(torch.FloatTensor))
                        pre_lab = torch.round(torch.sigmoid(pre))
                        scores.extend(F.softmax(pre, dim=0).numpy().tolist())
                        predicts.extend(pre_lab.long().numpy().tolist())
                        train_corrects += torch.sum(pre_lab.long() == batch.label)
                        labels.extend(batch.label.numpy().tolist())
                        train_num += len(batch.label)
                        epoch_loss += loss.item()
                    epoch_loss = epoch_loss / train_num
                    epoch_acc = train_corrects.double().item() / train_num
                return epoch_loss, epoch_acc, labels, predicts, scores

            flag = False
            for epoch in range(EPOCHS):
                start_time = time.time()
                train_loss, train_acc = train_epoch(model, train_iter, optimizer, criterion)
                val_loss, val_acc, val_labels, val_predicts, val_scores = evaluate(model, val_iter, criterion)
                end_time = time.time()

                val_AUC = roc_auc_score(np.array(val_labels), np.array(val_scores))
                val_precision = precision_score(np.array(val_labels), np.array(val_predicts))
                val_f1 = f1_score(np.array(val_labels), np.array(val_predicts))
                TP, TN, FP, FN = count(val_labels, val_predicts)
                # 马修斯相关系数
                if TP + FP == 0 or TP + FN == 0 or TN + FP == 0 or TN + FN == 0:
                    flag = True
                else:
                    flag = False
                if flag:
                    MCC = "null"
                else:
                    MCC = float(TP * TN - FP * FN) / ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5
                TPR = TP / (TP + FN)  # sensitivity
                TNR = TN / (TN + FP)  # specificity


                print("Epochs:", epoch + 1, "|", "Epoch Time:", end_time - start_time, "s")
                print("Val Loss:", val_loss, "|",
                      "Val Acc:", val_acc, "|",
                      "Val AUC:", val_AUC, "|",
                      "Val MCC:", MCC, "|",
                      "Val Sn:", TPR, "|",
                      "Val Sp:", TNR, "|",
                      "Val F1:", val_f1, "|",
                      "Val Precision:", val_precision
                      )
                results.write(str(val_AUC) + ',' +
                              str(val_acc) + ',' +
                              str(val_precision) + ',' +
                              str(MCC) + ',' +
                              str(val_f1) + ',' +
                              str(TPR) + ',' +
                              str(TNR) + '\n')
            results.close()
            df = pd.read_csv(results_csv)
            avg_data = list(map(str, df.mean()))
            avg_results.write(str(rate) + "_:_" + str(10-rate) + "," + ",".join(avg_data) + '\n')
        avg_results.close()



if __name__=="__main__":
    main()