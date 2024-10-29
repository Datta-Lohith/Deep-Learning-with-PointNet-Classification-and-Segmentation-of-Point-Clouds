import numpy as np
import argparse

import torch
from data_loader import get_data_loader
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


def rotate_test_data(point_cloud, angles):

    # Convert angles to radians 
    alpha, beta, gamma = np.deg2rad(angles)

    # Define rotation matrices
    Rx = torch.Tensor([[1,         0,           0       ],
                       [0, np.cos(alpha), -np.sin(alpha)],
                       [0, np.sin(alpha),  np.cos(alpha)]]).to(args.device) 
    
    Ry = torch.Tensor([[ np.cos(beta), 0, np.sin(beta)], 
                       [     0,        1,       0     ], 
                       [-np.sin(beta), 0, np.cos(beta)]]).to(args.device)
    
    Rz = torch.Tensor([[np.cos(gamma), -np.sin(gamma), 0],
                       [np.sin(gamma),  np.cos(gamma), 0],
                       [      0,             0,        1]]).to(args.device)

    # Combine rotations
    R_xyz = torch.matmul(Rz, torch.matmul(Ry, Rx))

    rotated_point_cloud = torch.matmul(point_cloud, R_xyz)

    return rotated_point_cloud


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

    #input point cloud rotation for Robustness Analysis
    angle_x, angle_y, angle_z = 90,0,0
    rotation_angles = (angle_x, angle_y, angle_z) 
    test_data = rotate_test_data(test_data, rotation_angles)


    # ------ TODO: Make Prediction ------
    batch_size = 4
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
    print ("test accuracy: {}".format(test_accuracy))   

    # Visualize Segmentation Result (Pred VS Ground Truth)
    for idx in range(0,len(test_data),10):
        if test_label[idx].unsqueeze(0) == 0:
            label_t = "chair"
        elif test_label[idx].unsqueeze(0) == 1: 
            label_t = "vase"       
        elif test_label[idx].unsqueeze(0) == 2:
            label_t = "lamp"
        viz_cls_seg(test_data[idx].cpu(), test_label[idx].cpu(), "{}/cls_rot/gt/{}_{}.gif".format(args.output_dir, idx, label_t), args.device, args.num_points, "cls_gt", str(idx)+"_"+label_t)
        if pred_label[idx].unsqueeze(0) == 0:
            label_p = "chair"
        elif pred_label[idx].unsqueeze(0) == 1: 
            label_p = "vase"       
        elif pred_label[idx].unsqueeze(0) == 2:
            label_p = "lamp"
        viz_cls_seg(test_data[idx].cpu(), pred_label[idx].cpu(), "{}/cls_rot/pred/{}_{}.gif".format(args.output_dir, idx, label_p), args.device, args.num_points, "cls_pred", str(idx)+"_"+label_p)
    

    #test accuracy: 0.938090241343127       for all 10 deg
    #test accuracy: 0.3578174186778594      for all 30 deg
    #test accuracy: 0.2465897166841553      for all 45 deg
    #test accuracy: 0.4994753410283316      for all 90 deg

    #test accuracy: 0.968520461699895       for x 10 deg
    #test accuracy: 0.727177334732424       for x 30 deg
    #test accuracy: 0.3588667366211962      for x 45 deg
    #test accuracy: 0.4491080797481637      for x 90 deg

    #test accuracy: 0.9727177334732424      for y 10 deg
    #test accuracy: 0.9034627492130115      for y 30 deg
    #test accuracy: 0.7481636935991606      for y 45 deg
    #test accuracy: 0.4994753410283316      for y 90 deg

    #test accuracy: 0.9559286463798531      for z 10 deg
    #test accuracy: 0.7460650577124869      for z 30 deg
    #test accuracy: 0.40713536201469047     for z 45 deg
    #test accuracy: 0.2203567681007345      for z 90 deg