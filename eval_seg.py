import numpy as np
import argparse

import torch
from models import seg_model
from utils import create_dir, viz_cls_seg
from tqdm import tqdm


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_seg_class', type=int, default=6, help='The number of segmentation classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='best_model')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='./data/seg/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/seg/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output')

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    # ------ TODO: Initialize Model for Segmentation Task  ------
    model = seg_model().to(args.device)
    
    # Load Model Checkpoint
    model_path = './checkpoints/seg/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))


    # Sample Points per Object
    ind = np.random.choice(10000,args.num_points, replace=False)
    test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:])
    test_label = torch.from_numpy((np.load(args.test_label))[:,ind])

    # ------ TODO: Make Prediction ------
    batch_size = 16
    num_batches = (test_data.shape[0] // batch_size)

    pred_label = torch.zeros_like(test_label)

    for batch_idx in tqdm(range(num_batches)):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, test_data.shape[0])

        batch_data = test_data[start_idx:end_idx].to(args.device)
        predictions = model(batch_data)

        batch_predictions = torch.argmax(predictions, dim=-1, keepdim=False).cpu()
        pred_label[start_idx:end_idx, :] = batch_predictions


    test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.reshape((-1,1)).size()[0])
    print ("test accuracy: {}".format(test_accuracy))   #test accuracy: 0.8852329011345219

    # Visualize Segmentation Result (Pred VS Ground Truth)
    for idx in range(0,len(test_data),50):
        viz_cls_seg(test_data[idx], test_label[idx], "{}/seg_points/gt/{}_chair.gif".format(args.output_dir, idx), args.device, args.num_points, "seg", "gt_"+str(idx)+"_chair")
        viz_cls_seg(test_data[idx], pred_label[idx], "{}/seg_points/pred/{}_chair.gif".format(args.output_dir,idx), args.device, args.num_points, "seg", "pred_"+str(idx)+"_chair")


    #test accuracy: 0.6784440842787682      --num_points    10
    #test accuracy: 0.7993192868719611      --num_points    100
    #test accuracy: 0.8486807131280389      --num_points    500
    #test accuracy: 0.8548427876823339      --num_points    1000 