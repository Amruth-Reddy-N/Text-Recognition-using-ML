import joblib
import numpy as np
from PIL import Image, ImageOps

def preprocess_image(path):
    im = Image.open(path).convert('L')
    im = ImageOps.invert(im)
    im = im.resize((28, 28))
    arr = np.array(im).astype(np.float32) / 255.0
    arr = np.rot90(arr, k=3)
    arr = np.fliplr(arr)
    return arr.flatten()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='./models/svm_emnist.joblib')
    parser.add_argument('image', help='Path to input image')
    args = parser.parse_args()

    model = joblib.load(args.model)
    x = preprocess_image(args.image)
    pred = model.predict([x])[0]
    print('Predicted label (EMNIST index):', pred)
