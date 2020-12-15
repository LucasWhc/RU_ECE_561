import argparse
import time
from tensorboardX import SummaryWriter
from dataset import dataset_loader
from torchvision import transforms
import torch
from torch import nn
from PIL import Image
from torch.utils.data import DataLoader
import re

data_path = './test_img/1.jpg'
est_width = 30
est_height = 30

def main():
    print('Welcome to the gesture recognition program!')

    # load model
    print('Model loading...')
    try:
        LeNet_num = torch.load('./model/LeNet5_kinect_leap_grayscale.pkl', map_location='cpu')
        LeNet_alph = torch.load('./model/LeNet5_sign_mnist_grayscale.pkl', map_location='cpu')
        ResNet34_num = torch.load('./model/ResNet34_kinect_leap_grayscale.pkl', map_location='cpu')
        # print(ResNet34_num)
        ResNet34_alph = torch.load('./model/ResNet34_sign_mnist_grayscale.pkl', map_location='cpu')
        print('Loading successful')
    except OSError as result:
        print('Loading failed...')
        print('Message: %s'%result)
        print('Exiting...')
        print()
        return

    while True:
        type = input('choose the type of prediction(1 means number, 2 means alphabet, q means quit the program):')
        print()
        # choosing the best model to predict
        if type == '1' or type == '2':
            # load test picture
            data_path = input('Please input the global path of the test image:')
            # print(data_path)
            print('Now loading the image...')
            try:
                img = Image.open(data_path)
                print('Loading successful')
                img.show()
            except OSError as result:
                print('Loading failed...')
                print("Message: %s"%result)
                print()
                continue
            if type == '1':
                if img.width > est_width and img.height > est_height:
                    pred_model = ResNet34_num
                    text = 'ResNet34_num'
                    img = img.resize((224, 224))
                else:
                    pred_model = LeNet_num
                    text = 'LeNet_num'
                    img = img.resize((28, 28))
                print('The model we recommend is %s' % text)
            elif type == '2':
                if img.width > est_width and img.height > est_height:
                    pred_model = ResNet34_alph
                    text = 'ResNet34_alph'
                    img = img.resize((224, 224))
                else:
                    pred_model = LeNet_alph
                    text = 'LeNet_alph'
                    img = img.resize((28, 28))
                print('The model we recommend is %s' % text)
        elif type == 'q':
            break
        else:
            print('Input error. Please follow the instruction...')
            continue

        # preprocess image
        transform = transforms.Compose([transforms.ToTensor()])
        img = img.convert('L')
        img_as_tensor = transform(img)
        img_as_tensor = img_as_tensor.unsqueeze(0)

        # prediction part
        output = pred_model(img_as_tensor)
        predicted = output.argmax(dim=1)
        # print(output)
        print('The result of your image is %d, the ground truth is' % predicted.item() + ' ' +
              re.findall(r"\d+", data_path)[0])
        print()
    print('Thanks for using!')
    return


if __name__ == "__main__":
    main()
