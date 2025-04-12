# Word2Vec-ResNet for Cross-Domain Promoter Prediction

## Project Description
This project provides a transfer learning framework combining **Word2Vec embeddings** with **ResNet architectures** to address dimension explosion in promoter sequence analysis. It also includes complete workflows for source domain pre-training and target domain model transfer, supporting both CPU/GPU environments. The transfer learning strategy enables using bacterial (B.subtilis) pre-trained models to predict promoter sequences in eukaryotes (D.melanogaster, S.cerevisiae).

## Dataset Information
### Data Sources
| Organism               | Source                                  |
|------------------------|-----------------------------------------|
| Bacillus subtilis      | [DBTBS Database](http://dbtbs.hgc.jp/)  |
| Escherichia coli       | [RegulonDB](http://regulondb.ccg.unam.mx/) |
| Saccharomyces cerevisiae | [EPD Database](https://epd.epfl.ch/)  |
| Drosophila melanogaster | [EPD Database](https://epd.epfl.ch/)  |

### Key Files
- `test.txt`: Promoter sequences for prediction (FASTA format)
- `test/`: Directory containing one-hot encoded matrices from test.txt
- `class_indices.json`: Label-to-class mapping file
- `B.subtilis/`: Negative example data of Bacillus subtilis
- `D.melanogaster/`: Negative example data of Drosophila melanogaster
- `E.coli/`: Negative example data of Escherichia coli
- `S.cerevisiae/`: Negative example data of Saccharomyces cerevisiae

## Code Structure
```bash
├── Training.py          # Main training script (CPU/GPU compatible)
├── Prediction.py        # Prediction script
├── promoter.pth         # B.subtilis pre-trained model (CPU version)
└── class_indices.json   # Label mapping file
```

## Usage Instructions
### Data Preparation
1. **Download the dataset** from the provided links and place them in the `data` directory.
2. **Prepare the data** by converting it into FASTA format.

### Model Training
**Run the training script**: Training.py is used to train and generate the pre-trained model in the source domain and the target model in the target domain. There are instructions on how to generate the two models in the source code (can run in both CPU and GPU environments).

### Promoter Prediction
**Run the prediction script**: Prediction.py is used to predict the promoter sequences in the target domain using the pre-trained model. The script reads the test data, and uses the pre-trained model to make predictions.

## Requirements
- Python 3.8+
- Core dependencies:`torch`, `torchvision`, `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `biopython`, `pillow`
