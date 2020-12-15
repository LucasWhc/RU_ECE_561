from PIL import Image
import pandas as pd
import numpy as np

loc = './csv/sign_mnist_test.csv'

def datareader(index):
    data = pd.read_csv('./csv/sign_mnist_test.csv')
    label = np.asarray(data.iloc[:, 0])
    img_as_np = np.asarray(data.iloc[index][1:]).reshape(28, 28).astype('uint8')
    img_as_img = Image.fromarray(img_as_np)
    img_as_img.show()
    img_as_img.save('./test_img/'+str(label[index])+'_sign.jpg')

def main():
    datareader(8)
    return

if __name__ == "__main__":
    main()


