from torchtext import data
import os
import random
from shutil import rmtree
from Bio import SeqIO
import numpy as np
import pandas as pd
from torchtext.vocab import Vectors
from gensim.models import word2vec
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
    seq_str = str(seq.seq).upper().strip()
    # Filter out non-standard DNA characters (only keep A, T, G, C)
    # Remove all non-ATGC characters (including N, -, spaces, etc.)
    seq_str = ''.join([c for c in seq_str if c in 'ATGC'])
    length = len(seq_str)
    
    # If sequence is too short or empty, return empty list
    if length < 6:
        return []
    
    # Sliding window
    for index, value in enumerate(seq_str):
        if index + 5 < length:
            kmer = seq_str[index:index+6]
            # Ensure k-mer only contains standard DNA characters
            if all(c in 'ATGC' for c in kmer):
                li.append(kmer)

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

# Multi-threading related locks have been removed

def run_single_experiment(exp_idx, species_name, corpus, words_species_name, experiments, 
                          BATCH_SIZE, EPOCHS, START, END, REPEAT_TIMES, 
                          USE_EARLY_STOPPING, EARLY_STOPPING_PATIENCE, 
                          EARLY_STOPPING_MIN_DELTA, EARLY_STOPPING_MONITOR, total_experiments):
    """Function to run a single experiment"""
    experiment_num = exp_idx + 1
    
    print(f"\n{'='*80}")
    print(f"Experiment [{experiment_num}/{total_experiments}] | Target Species: {species_name} | Corpus: {corpus}")
    print(f"  Vector Source Directory: {words_species_name} (dataset/words/{words_species_name}/)")
    print(f"{'='*80}\n")
    
    embedding_root_path = './dataset/embedding/' + species_name + '/'
    
    # Automatically create directory if it doesn't exist
    os.makedirs(embedding_root_path, exist_ok=True)
    
    negative_name = species_name + "_negative"
    positive_name = species_name + "_positive"

    input_negative = embedding_root_path + negative_name + ".txt"  # FASTA format non-promoter sequence file
    train_data = embedding_root_path + species_name + ".csv"
    
    print(f"[Experiment {experiment_num}] Preprocessing data...")
    
    # If file exists, delete it first to avoid duplicate data
    if os.path.exists(train_data):
        os.remove(train_data)
        print(f"[Experiment {experiment_num}]   Deleted old file: {train_data}")
    
    # Write header and negative data
    print(f"[Experiment {experiment_num}]   Processing negative data: {input_negative}")
    negative_count = 0
    skipped_negative = 0
    with open(train_data, 'w', encoding='utf-8') as f:
        f.write('text,label\n')
        for myseq in SeqIO.parse(input_negative, 'fasta'):
            sequence = seqword(myseq)
            # Skip empty sequences or sequences that are too short
            if len(sequence) == 0:
                skipped_negative += 1
                continue
            # Ensure sequence text doesn't contain commas, use quotes to wrap just in case
            seq_text = " ".join(sequence)
            f.write(f'"{seq_text}",0\n')
            negative_count += 1
            if negative_count % 100 == 0:
                print(f"[Experiment {experiment_num}]     Processed {negative_count} negative sequences", end='\r')
    print(f"[Experiment {experiment_num}]   Negative data completed: {negative_count} sequences")
    if skipped_negative > 0:
        print(f"[Experiment {experiment_num}]   Skipped invalid negative sequences: {skipped_negative}")

    input_positive = embedding_root_path + positive_name + ".txt"  # FASTA format promoter sequence file
    # Append positive data (without header)
    print(f"[Experiment {experiment_num}]   Processing positive data: {input_positive}")
    positive_count = 0
    skipped_positive = 0
    with open(train_data, 'a', encoding='utf-8') as f:
        for myseq in SeqIO.parse(input_positive, 'fasta'):
            sequence = seqword(myseq)
            # Skip empty sequences or sequences that are too short
            if len(sequence) == 0:
                skipped_positive += 1
                continue
            # Ensure sequence text doesn't contain commas, use quotes to wrap just in case
            seq_text = " ".join(sequence)
            f.write(f'"{seq_text}",1\n')
            positive_count += 1
            if positive_count % 100 == 0:
                print(f"[Experiment {experiment_num}]     Processed {positive_count} positive sequences", end='\r')
    print(f"[Experiment {experiment_num}]   Positive data completed: {positive_count} sequences")
    if skipped_positive > 0:
        print(f"[Experiment {experiment_num}]   Skipped invalid positive sequences: {skipped_positive}")
    print(f"[Experiment {experiment_num}]   Total: {negative_count + positive_count} sequences\n")

    embedding_path = './dataset/embedding/' + species_name + '/' + corpus
    
    # Automatically create directory if it doesn't exist
    os.makedirs(embedding_path, exist_ok=True)

    avg_results_csv = embedding_path + '/' + 'avg_data.csv'
    # If file doesn't exist, create it and write header; if exists, open in append mode
    if not os.path.exists(avg_results_csv):
        with open(avg_results_csv, 'w', encoding='utf-8') as f:
            f.write('Ratio,AUC,ACC,Precision,MCC,F1,Sensitivity,Specificity\n')
    
    # Open file outside lock to avoid holding lock for too long
    avg_results = open(avg_results_csv, 'a')

    total_rates = END - START
    current_rate_idx = 0
    
    for rate in range(START, END):
        current_rate_idx += 1
        val_rate = rate / 10
        result_save_path = embedding_path + "/" + str(rate) + '：' + str(10 - rate)  # Result save path
        
        # Check if result files exist
        results_csv = result_save_path + '/' + 'process_data.csv'
        stats_csv = result_save_path + '/' + 'statistics.csv'
        
        # Check if all repeated experiment results already exist
        skip_experiment = False
        if os.path.exists(results_csv) and os.path.exists(stats_csv):
            # Check the number of repeated experiments in process_data.csv
            try:
                df = pd.read_csv(results_csv)
                if len(df) >= REPEAT_TIMES:
                    print(f"[Experiment {experiment_num}] {'#'*80}")
                    print(f"[Experiment {experiment_num}] Skipping experiment: Results for ratio {rate}:{10-rate} already exist")
                    print(f"[Experiment {experiment_num}]   Found {len(df)} repeated experiment results (need {REPEAT_TIMES})")
                    print(f"[Experiment {experiment_num}]   Result file: {results_csv}")
                    print(f"[Experiment {experiment_num}] {'#'*80}\n")
                    skip_experiment = True
            except:
                pass
        
        if skip_experiment:
            continue
        
        print(f"[Experiment {experiment_num}] {'#'*80}")
        print(f"[Experiment {experiment_num}] Experiment Progress: [{current_rate_idx}/{total_rates}] | Current Ratio: {rate}:{10-rate}")
        print(f"[Experiment {experiment_num}] Species: {species_name} | Corpus: {corpus} | Vector Source: {words_species_name}")
        print(f"[Experiment {experiment_num}] {'#'*80}\n")
        
        # Create result save directory
        mk_file(result_save_path)

        # Store best metrics for all repeated experiments
        all_best_metrics = []
        
        # Repeated experiment loop
        for repeat_idx in range(REPEAT_TIMES):
            print(f"[Experiment {experiment_num}] {'='*80}")
            print(f"[Experiment {experiment_num}] Repeated Experiment [{repeat_idx+1}/{REPEAT_TIMES}] | Ratio: {rate}:{10-rate}")
            print(f"[Experiment {experiment_num}] {'='*80}\n")
            
            # Use different random seeds for each experiment to re-split the dataset
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
            
            # Re-split dataset using different random seeds
            train_data, val_data = traindata.split(
                split_ratio=val_rate, 
                random_state=random.seed(repeat_idx),  # Use different random seed for each experiment
                stratified=True, 
                strata_field='label'
            )
            
            if repeat_idx == 0:
                print(f"[Experiment {experiment_num}] Dataset Information:")
                print(f"[Experiment {experiment_num}]   - Total dataset size: {len(traindata)}")
                print(f"[Experiment {experiment_num}]   - Training set size: {len(train_data)}")
                print(f"[Experiment {experiment_num}]   - Validation set size: {len(val_data)}")
                print(f"[Experiment {experiment_num}]   - Training/Validation ratio: {len(train_data)}:{len(val_data)}")

            print(f"[Experiment {experiment_num}] Loading Word2Vec vectors...")
            print(f"[Experiment {experiment_num}]   Vector directory: dataset/words/{words_species_name}/")
            
            vectors_root_path = "./dataset/words/" + words_species_name
            vec = Vectors(vectors_root_path + "/seq_vectors.txt", vectors_root_path)
            
            print(f"[Experiment {experiment_num}] Building vocabulary...")
            TEXT.build_vocab(train_data, max_size=20000, vectors=vec)
            LABEL.build_vocab(train_data)

            # Ensure device is defined (before model creation)
            # Use default CUDA device, PyTorch will automatically manage multi-threaded GPU access
            if not torch.cuda.is_available():
                raise RuntimeError("GPU is not available! All training must run on GPU.")
            # Use default CUDA device instead of hardcoding cuda:0, so PyTorch can better manage multi-threaded GPU access
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            train_iter = data.BucketIterator(train_data, batch_size=BATCH_SIZE, device=device)
            val_iter = data.BucketIterator(val_data, batch_size=BATCH_SIZE, device=device)

            # Get a batch of data for demonstration
            for step, batch in enumerate(train_iter):
                if step > 0:
                    break

            # Instantiate model with ResNet parameters
            INPUT_DIM = len(TEXT.vocab)
            EMBEDDING_DIM = 100
            PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
            model = resnet18(INPUT_DIM, PAD_IDX)

            ###########################################Transfer learning#########################################################
            # Automatically determine whether to enable transfer learning:
            # - Experiments in experiments (exp_idx < len(experiments)): Cross-species transfer learning, enable pretrained embeddings
            # - Experiments in experiments_2 (exp_idx >= len(experiments)): Same-species training, do not use transfer learning
            is_transfer_learning_experiment = exp_idx < len(experiments)
            
            if is_transfer_learning_experiment:
                print(f"[Experiment {experiment_num}]   Transfer learning enabled: Target species ({species_name}) ≠ Vector source ({words_species_name})")
                pretrained_embeddings = TEXT.vocab.vectors
                model.embedding.weight.data.copy_(pretrained_embeddings)
            else:
                print(f"[Experiment {experiment_num}]   Transfer learning disabled: Same-species training (Target: {species_name}, Vector source: {words_species_name})")

            # Initialize vectors for unknown ('<unk>') and padding ('<pad>') tokens to zero
            UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
            model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
            model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

            # Adam optimizer and binary cross-entropy with logits loss
            # device is already defined when creating BucketIterator
            print(f"[Experiment {experiment_num}] Using device: {device}")
            
            # Move model to GPU
            model = model.to(device)
            
            optimizer = optim.Adam(model.parameters(), lr=0.0001)
            criterion = nn.BCEWithLogitsLoss()

            def train_epoch(model, iterator, optimizer, criterion, epoch, total_epochs):
                epoch_loss = 0
                epoch_acc = 0
                train_corrects = 0; train_num = 0
                model.train()
                total_batches = len(iterator)
                for batch_idx, batch in enumerate(iterator):
                    optimizer.zero_grad()
                    # BucketIterator has specified device, data is already on GPU, but ensure on GPU for safety
                    text_data = batch.text[0].to(device)
                    labels = batch.label.to(device)
                    
                    pre = model(text_data).squeeze(1)
                    loss = criterion(pre, labels.type(torch.FloatTensor).to(device))
                    pre_lab = torch.round(torch.sigmoid(pre))
                    train_corrects += torch.sum(pre_lab.long() == labels)
                    train_num += len(labels)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    
                    # Print batch progress
                    if (batch_idx + 1) % max(1, total_batches // 10) == 0 or (batch_idx + 1) == total_batches:
                        current_loss = epoch_loss / train_num if train_num > 0 else 0
                        current_acc = train_corrects.double().item() / train_num if train_num > 0 else 0
                        print(f"[Experiment {experiment_num}]   Epoch [{epoch+1}/{total_epochs}] | Batch [{batch_idx+1}/{total_batches}] | "
                              f"Loss: {current_loss:.4f} | Acc: {current_acc:.4f}", end='\r')
                
                print()
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
                total_batches = len(iterator)
                with torch.no_grad():
                    for batch_idx, batch in enumerate(iterator):
                        # BucketIterator has specified device, data is already on GPU, but ensure on GPU for safety
                        text_data = batch.text[0].to(device)
                        batch_labels = batch.label.to(device)
                        
                        pre = model(text_data).squeeze(1)
                        loss = criterion(pre, batch_labels.type(torch.FloatTensor).to(device))
                        pre_lab = torch.round(torch.sigmoid(pre))
                        scores.extend(F.softmax(pre, dim=0).cpu().numpy().tolist())
                        predicts.extend(pre_lab.long().cpu().numpy().tolist())
                        train_corrects += torch.sum(pre_lab.long() == batch_labels)
                        labels.extend(batch_labels.cpu().numpy().tolist())
                        train_num += len(batch_labels)
                        epoch_loss += loss.item()
                        
                        # Print validation progress
                        if (batch_idx + 1) % max(1, total_batches // 5) == 0 or (batch_idx + 1) == total_batches:
                            print(f"[Experiment {experiment_num}]   Validation progress: [{batch_idx+1}/{total_batches}]", end='\r')
                    
                    print()
                    epoch_loss = epoch_loss / train_num
                    epoch_acc = train_corrects.double().item() / train_num
                return epoch_loss, epoch_acc, labels, predicts, scores

            # Store metrics for all epochs of current experiment
            epoch_metrics = []
            
            # Early stopping mechanism variables
            best_monitor_value = -float('inf')  # Best monitored metric value
            patience_counter = 0  # Counter for epochs without improvement
            best_epoch_idx = 0  # Best epoch index
            
            print(f"[Experiment {experiment_num}] Starting training | Training/Validation ratio: {rate}:{10-rate} | Total epochs: {EPOCHS}")
            if USE_EARLY_STOPPING:
                print(f"[Experiment {experiment_num}] Early stopping: Enabled | Monitor metric: {EARLY_STOPPING_MONITOR} | Patience: {EARLY_STOPPING_PATIENCE} | Min delta: {EARLY_STOPPING_MIN_DELTA}")
            print()
            
            for epoch in range(EPOCHS):
                start_time = time.time()
                print(f"[Experiment {experiment_num}] [Experiment {repeat_idx+1}/{REPEAT_TIMES}] [Epoch {epoch+1}/{EPOCHS}] Training...")
                train_loss, train_acc = train_epoch(model, train_iter, optimizer, criterion, epoch, EPOCHS)
                print(f"[Experiment {experiment_num}] [Experiment {repeat_idx+1}/{REPEAT_TIMES}] [Epoch {epoch+1}/{EPOCHS}] Validating...")
                val_loss, val_acc, val_labels, val_predicts, val_scores = evaluate(model, val_iter, criterion)
                end_time = time.time()

                val_AUC = roc_auc_score(np.array(val_labels), np.array(val_scores))
                val_precision = precision_score(np.array(val_labels), np.array(val_predicts), zero_division=0)
                val_f1 = f1_score(np.array(val_labels), np.array(val_predicts), zero_division=0)
                TP, TN, FP, FN = count(val_labels, val_predicts)
                
                # Matthews correlation coefficient
                if TP + FP == 0 or TP + FN == 0 or TN + FP == 0 or TN + FN == 0:
                    MCC = None
                else:
                    MCC = float(TP * TN - FP * FN) / ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5
                TPR = TP / (TP + FN)  # sensitivity
                TNR = TN / (TN + FP)  # specificity

                epoch_time = end_time - start_time
                
                # Save current epoch metrics
                epoch_metrics.append({
                    'AUC': val_AUC,
                    'ACC': val_acc,
                    'Precision': val_precision,
                    'MCC': MCC if MCC is not None else 0,
                    'F1': val_f1,
                    'Sensitivity': TPR,
                    'Specificity': TNR
                })
                
                # Early stopping mechanism check
                if USE_EARLY_STOPPING:
                    current_monitor_value = epoch_metrics[-1][EARLY_STOPPING_MONITOR]
                    
                    # Check if there is improvement
                    if current_monitor_value > best_monitor_value + EARLY_STOPPING_MIN_DELTA:
                        best_monitor_value = current_monitor_value
                        best_epoch_idx = epoch
                        patience_counter = 0
                        improved = True
                    else:
                        patience_counter += 1
                        improved = False
                    
                    # Print progress (every 10 epochs or when improved)
                    if (epoch + 1) % 10 == 0 or improved or (epoch + 1) == EPOCHS:
                        print(f"[Experiment {experiment_num}] [Experiment {repeat_idx+1}/{REPEAT_TIMES}] Epoch [{epoch+1}/{EPOCHS}] | Time: {epoch_time:.2f}s")
                        print(f"[Experiment {experiment_num}]   AUC: {val_AUC:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f} | MCC: {MCC if MCC is not None else 'null'}")
                        if improved:
                            print(f"[Experiment {experiment_num}]   ✓ {EARLY_STOPPING_MONITOR} improved to {best_monitor_value:.4f} (Best Epoch: {best_epoch_idx+1})")
                        else:
                            print(f"[Experiment {experiment_num}]   - {EARLY_STOPPING_MONITOR} not improved (Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE})")
                    
                    # Early stopping check
                    if patience_counter >= EARLY_STOPPING_PATIENCE:
                        print(f"[Experiment {experiment_num}] {'='*80}")
                        print(f"[Experiment {experiment_num}] Early stopping triggered! Validation {EARLY_STOPPING_MONITOR} did not improve within {EARLY_STOPPING_PATIENCE} epochs")
                        print(f"[Experiment {experiment_num}] Best {EARLY_STOPPING_MONITOR}: {best_monitor_value:.4f} (Epoch {best_epoch_idx+1})")
                        print(f"[Experiment {experiment_num}] Current {EARLY_STOPPING_MONITOR}: {current_monitor_value:.4f} (Epoch {epoch+1})")
                        print(f"[Experiment {experiment_num}] Training ended early, trained for {epoch+1}/{EPOCHS} epochs")
                        print(f"[Experiment {experiment_num}] {'='*80}\n")
                        break
                else:
                    # When not using early stopping, print every 10 epochs
                    if (epoch + 1) % 10 == 0 or (epoch + 1) == EPOCHS:
                        print(f"[Experiment {experiment_num}] [Experiment {repeat_idx+1}/{REPEAT_TIMES}] Epoch [{epoch+1}/{EPOCHS}] | Time: {epoch_time:.2f}s")
                        print(f"[Experiment {experiment_num}]   AUC: {val_AUC:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f} | MCC: {MCC if MCC is not None else 'null'}")
            
            # Find best metrics for current experiment (based on monitor metric)
            if not USE_EARLY_STOPPING:
                best_epoch_idx = max(range(len(epoch_metrics)), key=lambda i: epoch_metrics[i][EARLY_STOPPING_MONITOR])
            best_metrics = epoch_metrics[best_epoch_idx].copy()
            best_metrics['best_epoch'] = best_epoch_idx + 1
            all_best_metrics.append(best_metrics)
            
            print(f"[Experiment {experiment_num}] [Experiment {repeat_idx+1}/{REPEAT_TIMES}] Completed | Best Epoch: {best_epoch_idx+1}")
            print(f"[Experiment {experiment_num}]   Best Metrics - AUC: {best_metrics['AUC']:.4f} | ACC: {best_metrics['ACC']:.4f} | "
                  f"F1: {best_metrics['F1']:.4f} | MCC: {best_metrics['MCC']:.4f}")
        
        # Calculate mean and standard deviation of best metrics for all repeated experiments
        metrics_names = ['AUC', 'ACC', 'Precision', 'MCC', 'F1', 'Sensitivity', 'Specificity']
        stats_results = {}
        
        for metric_name in metrics_names:
            values = [m[metric_name] for m in all_best_metrics]
            mean_val = np.mean(values)
            std_val = np.std(values)
            stats_results[metric_name] = {'mean': mean_val, 'std': std_val}
        
        # Save file write operations
        results_csv = result_save_path + '/' + 'process_data.csv'
        if not os.path.exists(results_csv):
            with open(results_csv, 'w', encoding='utf-8') as f:
                f.write('Repeat,AUC,ACC,Precision,MCC,F1,Sensitivity,Specificity,Best_Epoch\n')
        
        with open(results_csv, 'a', encoding='utf-8') as f:
            for idx, best_metrics in enumerate(all_best_metrics):
                f.write(f"{idx+1}," +
                       f"{best_metrics['AUC']:.6f}," +
                       f"{best_metrics['ACC']:.6f}," +
                       f"{best_metrics['Precision']:.6f}," +
                       f"{best_metrics['MCC']:.6f}," +
                       f"{best_metrics['F1']:.6f}," +
                       f"{best_metrics['Sensitivity']:.6f}," +
                       f"{best_metrics['Specificity']:.6f}," +
                       f"{best_metrics['best_epoch']}\n")
        
        # Save statistical results (mean ± std)
        stats_csv = result_save_path + '/' + 'statistics.csv'
        if not os.path.exists(stats_csv):
            with open(stats_csv, 'w', encoding='utf-8') as f:
                f.write('Metric,Mean,Std\n')
        
        with open(stats_csv, 'a', encoding='utf-8') as f:
            f.write(f"Ratio_{rate}_{10-rate}\n")
            for metric_name in metrics_names:
                mean_val = stats_results[metric_name]['mean']
                std_val = stats_results[metric_name]['std']
                f.write(f"{metric_name},{mean_val:.6f},{std_val:.6f}\n")
            f.write("\n")
        
        # Save to avg_data.csv (format: mean±std)
        avg_data_str = str(rate) + "_:_" + str(10-rate)
        for metric_name in metrics_names:
            mean_val = stats_results[metric_name]['mean']
            std_val = stats_results[metric_name]['std']
            avg_data_str += f",{mean_val:.6f}±{std_val:.6f}"
        avg_results.write(avg_data_str + '\n')
        avg_results.flush()  # Ensure data is written immediately
        
        # Print statistical results
        print(f"[Experiment {experiment_num}] {'#'*80}")
        print(f"[Experiment {experiment_num}] Ratio {rate}:{10-rate} All repeated experiments completed!")
        print(f"[Experiment {experiment_num}] {'='*80}")
        print(f"[Experiment {experiment_num}] Statistical Results (mean±std, based on best metrics from {REPEAT_TIMES} repeated experiments):")
        for metric_name in metrics_names:
            mean_val = stats_results[metric_name]['mean']
            std_val = stats_results[metric_name]['std']
            print(f"[Experiment {experiment_num}]   {metric_name}: {mean_val:.4f} ± {std_val:.4f}")
        print(f"[Experiment {experiment_num}] {'#'*80}\n")
    
    # Close file after all rate loops end
    avg_results.close()
    print(f"\n[Experiment {experiment_num}] {'='*80}")
    print(f"[Experiment {experiment_num}] Experiment completed! Results saved")
    print(f"[Experiment {experiment_num}] {'='*80}\n")


def main():
    ################################################### generate
    # words_species_name = "Dro_Genomic"
    
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
    EPOCHS = 50
    START = 1
    END = 10
    
    # Number of repeated experiments: Each experiment will re-randomize train/validation split
    REPEAT_TIMES = 5
    
    # Early stopping mechanism parameters
    USE_EARLY_STOPPING = True  # Whether to enable early stopping
    EARLY_STOPPING_PATIENCE = 10  # Stop if validation metric doesn't improve within this many epochs
    EARLY_STOPPING_MIN_DELTA = 0.0001  # Minimum improvement magnitude (AUC)
    EARLY_STOPPING_MONITOR = 'AUC'  # Monitored metric: 'AUC', 'ACC', 'F1'

    # Define 8 experiment configurations
    # Format: (target_species, corpus_name, words_species_name)
    # words_species_name: Used to specify the vector file directory name under dataset/words/
    experiments = [
        # 1. E. coli (source) -> B. subtilis (target)
        ("B.subtilis", "E_to_B", "E"),
        # 2. B. subtilis (source) -> E. coli (target)
        ("E.coli", "B_to_E", "B"),
        # 3. D. melanogaster (source) -> S. cerevisiae (target)
        ("S.cerevisiae", "D_to_S", "D"),
        # 4. S. cerevisiae (source) -> D. melanogaster (target)
        ("D.melanogaster", "S_to_D", "S"),
        # 5. D. melanogaster + S. cerevisiae (source) -> B. subtilis (target)
        ("B.subtilis", "SD_to_B", "SD"),
        # 6. D. melanogaster + S. cerevisiae (source) -> E. coli (target)
        ("E.coli", "SD_to_E", "SD"),
        # 7. B. subtilis + E. coli (source) -> S. cerevisiae (target)
        ("S.cerevisiae", "BE_to_S", "BE"),
        # 8. B. subtilis + E. coli (source) -> D. melanogaster (target)
        ("D.melanogaster", "BE_to_D", "BE"),
    ]

    # Same-species training experiments (without transfer learning)
    # Format: (target_species, corpus_name, words_species_name)
    # Note: target_species and words_species_name should correspond to the same species
    experiments_2 = [
        # 1. B. subtilis -> B. subtilis (same-species training, without transfer learning)
        ("B.subtilis", "B_to_B", "B"),
        # 2. E. coli -> E. coli (same-species training, without transfer learning)
        ("E.coli", "E_to_E", "E"),
        # 3. S. cerevisiae -> S. cerevisiae (same-species training, without transfer learning)
        ("S.cerevisiae", "S_to_S", "S"),
        # 4. D. melanogaster -> D. melanogaster (same-species training, without transfer learning)
        ("D.melanogaster", "D_to_D", "D"),
    ]

    # Select experiments to run
    # experiments: Transfer learning experiments (cross-species)
    # experiments_2: Same-species training experiments (without transfer learning)
    experiments_to_run = experiments + experiments_2
    
    total_experiments = len(experiments_to_run)
    print(f"\n{'='*80}")
    print(f"Starting processing | Total experiments: {total_experiments}")
    print(f"  - Transfer learning experiments: {len(experiments)}")
    print(f"  - Same-species training experiments: {len(experiments_2)}")
    print(f"  - Using sequential execution (single-threaded)")
    print(f"{'='*80}\n")
    
    # Sequentially execute all experiments
    print(f"Starting sequential execution of all {total_experiments} experiments...\n")
    
    # Sequentially execute each experiment
    for exp_idx, (species_name, corpus, words_species_name) in enumerate(experiments_to_run):
        try:
            run_single_experiment(
                exp_idx, species_name, corpus, words_species_name, experiments,
                BATCH_SIZE, EPOCHS, START, END, REPEAT_TIMES,
                USE_EARLY_STOPPING, EARLY_STOPPING_PATIENCE,
                EARLY_STOPPING_MIN_DELTA, EARLY_STOPPING_MONITOR, total_experiments
            )
            print(f"\n✓ Experiment {exp_idx+1}/{total_experiments} completed\n")
        except Exception as e:
            print(f"\n✗ Experiment {exp_idx+1}/{total_experiments} failed: {e}\n")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print(f"All experiments completed!")
    print(f"{'='*80}\n")


if __name__=="__main__":
    main()