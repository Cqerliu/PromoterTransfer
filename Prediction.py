import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from Training import resnet18, DNAmarix, mk_file
import numpy as np
from Bio import SeqIO
import glob



def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    mk_file('/test')
    index = 1
    for myseq in SeqIO.parse('test.txt', 'fasta'):
        sequence = DNAmarix(myseq)
        channel = []
        channel.append(sequence)
        channel.append(sequence)
        channel.append(sequence)
        image_matrix = np.array(channel)
        image_matrix = np.transpose(image_matrix)
        image_matrix = np.uint8(image_matrix)
        image = Image.fromarray(image_matrix)
        image.save('./test/test' + str(index) + '.tif')
        index = index + 1

    # 存放图片的文件夹路径
    WSI_MASK_PATH = './test/'
    paths = glob.glob(os.path.join(WSI_MASK_PATH, '*.tif'))
    paths.sort()

    i = 1
    for img_path in paths:
        img = Image.open(img_path)
        plt.imshow(img)
        # [N, C, H, W]
        img = data_transform(img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        # read class_indict
        json_path = './class_indices.json'
        assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

        json_file = open(json_path, "r")
        class_indict = json.load(json_file)

        # create model
        model = resnet18(num_classes=2).to(device)

        # load model weights
        weights_path = "./promoter.pth"
        assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
        model.load_state_dict(torch.load(weights_path, map_location=device))

        # prediction
        model.eval()
        with torch.no_grad():
            # predict class
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
        print('Article ' + str(i) + ' Sequence:')

        print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                     predict[predict_cla].numpy())
        plt.title(print_res)
        for i in range(len(predict)):
            print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                      predict[i].numpy()))
        plt.show()
        i = i + 1


if __name__ == '__main__':
    main()