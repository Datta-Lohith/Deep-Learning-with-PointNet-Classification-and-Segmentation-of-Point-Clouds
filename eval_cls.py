import numpy as np
import argparse

import torch
from models import cls_model
from utils import create_dir, viz_cls_seg
from tqdm import tqdm

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_cls_class', type=int, default=3, help='The number of classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='best_model')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='./data/cls/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/cls/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output')

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    # ------ TODO: Initialize Model for Classification Task ------
    model = cls_model().to(args.device)
    
    # Load Model Checkpoint
    model_path = './checkpoints/cls/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))


    # Sample Points per Object
    ind = np.random.choice(10000,args.num_points, replace=False)
    test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:]).to(args.device)
    test_label = torch.from_numpy(np.load(args.test_label)).to(args.device)

    # ------ TODO: Make Prediction ------
    batch_size = 32
    num_batch = (test_data.shape[0] + batch_size - 1) // batch_size 
    pred_label = []

    for i in tqdm(range(num_batch)):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, test_data.shape[0])
        batch_data = test_data[start_idx:end_idx].to(args.device)
        pred = model(batch_data)
        pred_label_ = torch.argmax(pred, dim=-1, keepdim=False).to(args.device)
        pred_label.append(pred_label_)
    pred_label = torch.cat(pred_label, dim=0)

    # Compute Accuracy
    test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.size()[0])
    print ("test accuracy: {}".format(test_accuracy))    #test accuracy: 0.9758656873032528

    # Visualize Segmentation Result (Pred VS Ground Truth)
    for idx in range(0,len(test_data),25):
        if test_label[idx].unsqueeze(0) == 0:
            label_t = "chair"
        elif test_label[idx].unsqueeze(0) == 1: 
            label_t = "vase"       
        elif test_label[idx].unsqueeze(0) == 2:
            label_t = "lamp"
        viz_cls_seg(test_data[idx].cpu(), test_label[idx].cpu(), "{}/cls/gt/{}_{}.gif".format(args.output_dir, idx, label_t), args.device, args.num_points, "cls_gt", str(idx)+"_"+label_t)
        if pred_label[idx].unsqueeze(0) == 0:
            label_p = "chair"
        elif pred_label[idx].unsqueeze(0) == 1: 
            label_p = "vase"       
        elif pred_label[idx].unsqueeze(0) == 2:
            label_p = "lamp"
        viz_cls_seg(test_data[idx].cpu(), pred_label[idx].cpu(), "{}/cls/pred/{}_{}.gif".format(args.output_dir, idx, label_p), args.device, args.num_points, "cls_pred", str(idx)+"_"+label_p)
    

    #test accuracy: 0.25078698845750264     --num_points    10
    #test accuracy: 0.9108079748163693      --num_points    100
    #test accuracy: 0.9695697796432319      --num_points    500
    #test accuracy: 0.9716684155299056      --num_points    1000
