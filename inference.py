import torch
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, Resize
import warnings
import argparse
import cv2
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights

warnings.filterwarnings('ignore')

def get_args():
    parser = argparse.ArgumentParser("Test arguments")
    parser.add_argument("--image_path", "-s", type=str)
    parser.add_argument("--image_size", "-i", type=int, default=224)
    parser.add_argument("--checkpoint-path", "-t", type=str, default="trained_models/animals/best.pt")
    args = parser.parse_args()
    return args

def inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    categories = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider",
                  "squirrel"]
    ori_image = cv2.imread(args.image_path)
    image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (args.image_size, args.image_size))
    image = np.transpose(image, (2, 0, 1))/255.
    image = np.expand_dims(image, axis=0)
    image = torch.from_numpy(image).float()
    image = image.to(device)

    # model = AdvancedCNN(num_classes=10).to(device)   # Our own model
    model = resnet18(weights=None)
    model.fc = nn.Linear(512, 10)
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    softmax = nn.Softmax()
    with torch.no_grad():
        prediction = model(image)
        print(prediction)
        prob = softmax(prediction)

    max_value, max_index = torch.max(prob, dim=1)
    cv2.imshow("{} with confident score of {}".format(categories[max_index[0]], max_value[0]), ori_image)
    cv2.waitKey(0)





if __name__ == '__main__':
    args = get_args()
    inference(args)
