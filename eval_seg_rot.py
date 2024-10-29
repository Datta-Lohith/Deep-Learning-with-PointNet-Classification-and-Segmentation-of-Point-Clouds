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


def rotate_test_data(point_cloud, angles):

    # Convert angles to radians 
    alpha, beta, gamma = np.deg2rad(angles)

    # Define rotation matrices
    Rx = torch.Tensor([[1,         0,           0       ],
                       [0, np.cos(alpha), -np.sin(alpha)],
                       [0, np.sin(alpha),  np.cos(alpha)]])
    
    Ry = torch.Tensor([[ np.cos(beta), 0, np.sin(beta)], 
                       [     0,        1,       0     ], 
                       [-np.sin(beta), 0, np.cos(beta)]])
    
    Rz = torch.Tensor([[np.cos(gamma), -np.sin(gamma), 0],
                       [np.sin(gamma),  np.cos(gamma), 0],
                       [      0,             0,        1]])

    # Combine rotations
    R_xyz = torch.matmul(Rz, torch.matmul(Ry, Rx))

    rotated_point_cloud = torch.matmul(point_cloud, R_xyz)

    return rotated_point_cloud


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

    #input point cloud rotation for Robustness Analysis
    angle_x, angle_y, angle_z = 0,0,90
    rotation_angles = (angle_x, angle_y, angle_z) 
    test_data = rotate_test_data(test_data, rotation_angles)


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
    print ("test accuracy: {}".format(test_accuracy))


    # Visualize Segmentation Result (Pred VS Ground Truth)
    for idx in range(0,len(test_data),50):
        viz_cls_seg(test_data[idx], test_label[idx], "{}/seg_rot/gt/{}_chair.gif".format(args.output_dir, idx), args.device, args.num_points, "seg", "gt_"+str(idx)+"_chair")
        viz_cls_seg(test_data[idx], pred_label[idx], "{}/seg_rot/pred/{}_chair.gif".format(args.output_dir,idx), args.device, args.num_points, "seg", "pred_"+str(idx)+"_chair")


    #test accuracy: 0.8072061588330632      for all 10 deg
    #test accuracy: 0.6612316045380875      for all 30 deg
    #test accuracy: 0.594016531604538       for all 45 deg
    #test accuracy: 0.6160053484602918      for all 90 deg

    #test accuracy: 0.8496586709886548      for x 10 deg
    #test accuracy: 0.7525256077795786      for x 30 deg
    #test accuracy: 0.3588667366211962      for x 45 deg
    #test accuracy: 0.1625353322528363      for x 90 deg

    #test accuracy: 0.8391905996758509      for y 10 deg
    #test accuracy: 0.7756982171799027      for y 30 deg
    #test accuracy: 0.7306922204213938      for y 45 deg
    #test accuracy: 0.6160053484602918      for y 90 deg

    #test accuracy: 0.8407273905996758      for z 10 deg
    #test accuracy: 0.6974222042139384      for z 30 deg
    #test accuracy: 0.5888212317666126      for z 45 deg
    #test accuracy: 0.3890283630470016      for z 90 deg     