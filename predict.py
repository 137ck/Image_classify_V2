import numpy as np
import torch
import argparse
import json
from PIL import Image

def main():
    args = get_arguments()
    cuda = args.cuda
    model = load_checkpoint(args.checkpoint, cuda)
    model.idx_to_class = dict([[m,n] for m, n in model.class_to_idx.items()])
    
    with open(args.categories, 'r') as f:
        cat_to_name = json.load(f)
      
    prob, classes = class_prediction_fuc(args.input, model, args.cuda, topk=int(args.top_k))
    print([cat_to_name[x] for x in classes])

def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("input", action="store")
    parser.add_argument("checkpoint", action="store")


    parser.add_argument("--top_k", action="store", dest="top_k", default=5, help="Quantities of top results you like")
    parser.add_argument("--category_names", action="store", dest="categories", default="cat_to_name.json", 
                        help="Quantities of top results you like")
    parser.add_argument("--cuda", action="store_true", dest="cuda", default=False, help=" Cuda True means using the      GPU")

    return parser.parse_args()

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = checkpoint['optimizer']
    epochs = checkpoint['epoch_nums']
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model

def process_image(image):

    image_rate = image.size[1] / image.size[0]
    image = image.resize((256, int(image_rate*256)))
    h_half = image.size[1]*0.5
    w_half = image.size[0]*0.5
    image = image.crop((w_half - 112,
                        h_half - 112,
                        w_half + 112,
                        h_half + 112))
    image = np.array(image)
    image = image / 255
    
    standard_deviation = np.array([0.229, 0.224, 0.225])
    average = np.array([0.485, 0.456, 0.406])
    image = (image - average) / standard_deviation
    image = image.transpose((2, 0, 1))
    
    return torch.from_numpy(image)

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    

    image = image.numpy().transpose((1, 2, 0))
    
  
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    

    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def class_prediction_fuc(image_path, model,cuda,topk=5):
    if cuda:
        model.cuda()
    else:
        model.cpu()
    image=none
    model.eval()
    with Image.open(image_path) as img:
        image = process_image(img)
    if cuda:
        image = image.cuda()
    else:
        image = image.cpu()
    logp = model.forward(image.unsqueeze(0).float())
    posibilities = torch.exp(logp)
    probs, indexes = posibilities.topk(topk, dim=1)
    probs=[y.item() for y in probs.data[0]]
    indexes=[x.item() for x in indexes.data[0]]  
    idx_to_class = {m: n for n,m in model.class_to_idx.items()}
    class_types = [idx_to_class[x] for x in indexes]
    return probs,class_types

main()


