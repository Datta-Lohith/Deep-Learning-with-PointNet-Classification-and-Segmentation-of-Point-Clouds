# Deep Learning with PointNet: Classification and Segmentation of Point Clouds

## Overview
Implementation of PointNet-based architecture for classification and segmentation with point clouds.

`models.py` is where the model structures are defined. `train.py` loads data, trains models, logs trajectories, and saves checkpoints. `eval_cls.py` and `eval_seg.py` contain scripts to evaluate model accuracy and visualize segmentation results.

## Dependencies
To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

Ensure you have `pip` installed. If not, you can install it by following the instructions [here](https://pip.pypa.io/en/stable/installation/).

## Data Preparation
Download the zip file (~2GB) from [here](https://drive.google.com/file/d/1wXOgwM_rrEYJfelzuuCkRfMmR0J7vLq_/view?usp=sharing). Put the unzipped `data` folder under the root directory. There are two folders (`cls` and `seg`) corresponding to two tasks, each of which contains `.npy` files for training and testing.

## Classification Model
- Input: points clouds from across 3 classes (chairs, vases, and lamps objects)

- Output: probability distribution indicating predicted classification (Dimension: Batch * Number of Classes)

Run `python train.py --task cls` to train the model, and `python eva_cls.py` for evaluation.

#### Test Accuracy : __97.58%__

#### Results

- Correct Classifications/Predictions
  
| Class | Ground Truth | Prediction | Ground Truth | Prediction | Ground Truth | Prediction |
|-------|--------------|------------|--------------|------------|--------------|------------|
| Chair | <img src="output/cls/gt/75_chair.gif" alt="Input RGB" width="200"/> | <img src="output/cls/pred/75_chair.gif" alt="Input RGB" width="200"/> | <img src="output/cls/gt/125_chair.gif" alt="Input RGB" width="200"/> | <img src="output/cls/pred/125_chair.gif" alt="Input RGB" width="200"/> | <img src="output/cls/gt/435_chair.gif" alt="Input RGB" width="200"/> | <img src="output/cls/pred/435_chair.gif" alt="Input RGB" width="200"/> |
| Vase  | <img src="output/cls/gt/640_vase.gif" alt="Input RGB" width="200"/>  | <img src="output/cls/pred/640_vase.gif" alt="Input RGB" width="200"/>  | <img src="output/cls/gt/655_vase.gif" alt="Input RGB" width="200"/>  | <img src="output/cls/pred/655_vase.gif" alt="Input RGB" width="200"/>  | <img src="output/cls/gt/705_vase.gif" alt="Input RGB" width="200"/>  | <img src="output/cls/pred/705_vase.gif" alt="Input RGB" width="200"/>  |
| Lamp  |  <img src="output/cls/gt/790_lamp.gif" alt="Input RGB" width="200"/>  | <img src="output/cls/pred/790_lamp.gif" alt="Input RGB" width="200"/>  |  <img src="output/cls/gt/880_lamp.gif" alt="Input RGB" width="200"/>  |  <img src="output/cls/pred/880_lamp.gif" alt="Input RGB" width="200"/>  |  <img src="output/cls/gt/910_lamp.gif" alt="Input RGB" width="200"/>  |  <img src="output/cls/pred/910_lamp.gif" alt="Input RGB" width="200"/>  |

- Incorrect Classifications/Predictions

| Class | Ground Truth | Prediction |
|-------|--------------|------------|
| Chair | <img src="output/cls/gt/595_chair.gif" alt="Input RGB" width="200"/> | <img src="output/cls/pred/595_lamp.gif" alt="Input RGB" width="200"/> |
| Vase  | <img src="output/cls/gt/620_vase.gif" alt="Input RGB" width="200"/>  | <img src="output/cls/pred/620_lamp.gif" alt="Input RGB" width="200"/>  | 
| Lamp  |  <img src="output/cls/gt/750_lamp.gif" alt="Input RGB" width="200"/>  | <img src="output/cls/pred/750_vase.gif" alt="Input RGB" width="200"/>  |  


#### Interpretation
The overall performance of the classification model is commendable, especially in accurately identifying chairs where it exhibits a high level of precision. 
Nevertheless, there are instances where the model tends to misclassify certain samples, particularly confusing between lamps and vases. 
This suggests the model encounters challenges distinguishing between classes with similar structural characteristics.


## Segmentation Model
- Input: points of chair objects (6 semantic segmentation classes)
- Output: segmentation of points (Dimension: Batch * Number of Points per Object * Number of Segmentation Classes)

Run `python train.py --task seg` to train the model. Running `python eval_seg.py` will save two gifs, one for ground truth and the other for model prediction.

#### Test Accuracy : __88.52%__


#### Results

- Good Segmentations/Predictions

| Ground Truth | Prediction | Ground Truth | Prediction |
|--------------|------------|--------------|------------|
| <img src="output/seg/gt/75_chair.gif" alt="Input RGB" width="200"/> | <img src="output/seg/pred/75_chair.gif" alt="Input RGB" width="200"/> | <img src="output/seg/gt/125_chair.gif" alt="Input RGB" width="200"/> | <img src="output/seg/pred/125_chair.gif" alt="Input RGB" width="200"/> |
| <img src="output/seg/gt/435_chair.gif" alt="Input RGB" width="200"/> | <img src="output/seg/pred/435_chair.gif" alt="Input RGB" width="200"/> | <img src="output/seg/gt/55_chair.gif" alt="Input RGB" width="200"/> | <img src="output/seg/pred/55_chair.gif" alt="Input RGB" width="200"/> |
| <img src="output/seg/gt/260_chair.gif" alt="Input RGB" width="200"/> | <img src="output/seg/pred/260_chair.gif" alt="Input RGB" width="200"/> | <img src="output/seg/gt/10_chair.gif" alt="Input RGB" width="200"/> | <img src="output/seg/pred/10_chair.gif" alt="Input RGB" width="200"/> |


- Bad Segmentations/Predictions

| Ground Truth | Prediction |
|--------------|------------|
| <img src="output/seg/gt/605_chair.gif" alt="Input RGB" width="200"/> | <img src="output/seg/pred/605_chair.gif" alt="Input RGB" width="200"/> |
| <img src="output/seg/gt/140_chair.gif" alt="Input RGB" width="200"/> | <img src="output/seg/pred/140_chair.gif" alt="Input RGB" width="200"/> |
| <img src="output/seg/gt/335_chair.gif" alt="Input RGB" width="200"/> | <img src="output/seg/pred/335_chair.gif" alt="Input RGB" width="200"/> |

#### Interpretation

The model excels in accurately segmenting the various components of a well-defined conventional chair, especially where the different segments like chair legs, backrests, armrests, etc are distinguishably defined. 
However, its performance experiences a decline when faced with unconventional chair designs, particularly those that integrate the legs and base. 
In such instances, the model encounters challenges, often misidentifying a substantial part of the legs as the seat.


## Robustness Analysis

###  Rotating Pointclouds

Evaluating the classification and segmentation models on rotated point-clouds. Point-clouds are either rotated along all x,y, and z axes combined or individually along a single axis.

#### Procedure
   ```
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
```

Before starting the prediction the input test_data (point clouds) is rotated with the desired angles along x,y, and z.

```
    #input point cloud rotation for Robustness Analysis
    angle_x, angle_y, angle_z = 90,0,0
    rotation_angles = (angle_x, angle_y, angle_z) 
    test_data = rotate_test_data(test_data, rotation_angles)
```
#### Visualizations - Classification

Rotation along x, y, and z:

| Class | Ground Truth | 0&deg; | 45&deg; | 90&deg; | 
|-------|--------------|------------|--------------|------------|
| Chair | <img src="output/cls_rotation/all_xyz/cls_rot_all_10/gt/460_chair.gif" alt="Input RGB" width="200"/> | <img src="output/cls/pred/460_chair.gif" alt="Input RGB" width="200"/> | <img src="output/cls_rotation/all_xyz/cls_rot_all_45/pred/460_vase.gif" alt="Input RGB" width="200"/> | <img src="output/cls_rotation/all_xyz/cls_rot_all_90/pred/460_chair.gif" alt="Input RGB" width="200"/> | 
| Vase  | <img src="output/cls_rotation/all_xyz/cls_rot_all_10/gt/650_vase.gif" alt="Input RGB" width="200"/>  | <img src="output/cls/pred/650_vase.gif" alt="Input RGB" width="200"/>  | <img src="output/cls_rotation/all_xyz/cls_rot_all_45/pred/650_vase.gif" alt="Input RGB" width="200"/>  | <img src="output/cls_rotation/all_xyz/cls_rot_all_90/pred/650_vase.gif" alt="Input RGB" width="200"/>  |
| Lamp  |  <img src="output/cls_rotation/all_xyz/cls_rot_all_10/gt/950_lamp.gif" alt="Input RGB" width="200"/>  | <img src="output/cls/pred/950_lamp.gif" alt="Input RGB" width="200"/>  |  <img src="output/cls_rotation/all_xyz/cls_rot_all_45/pred/950_vase.gif" alt="Input RGB" width="200"/>  |  <img src="output/cls_rotation/all_xyz/cls_rot_all_90/pred/950_lamp.gif" alt="Input RGB" width="200"/>  | 
| Test Accuracy  | -  | __97.58%__ | __24.65%__  |  __49.94%__  | 

Rotation along x only:

| Class | Ground Truth | 0&deg; | 45&deg; | 90&deg; | 
|-------|--------------|------------|--------------|------------|
| Chair | <img src="output/cls_rotation/along_x/cls_rot_x_10/gt/520_chair.gif" alt="Input RGB" width="200"/> | <img src="output/cls/pred/520_chair.gif" alt="Input RGB" width="200"/> | <img src="output/cls_rotation/along_x/cls_rot_x_45/pred/520_chair.gif" alt="Input RGB" width="200"/> | <img src="output/cls_rotation/along_x/cls_rot_x_90/pred/520_chair.gif" alt="Input RGB" width="200"/> | 
| Vase  | <img src="output/cls_rotation/along_x/cls_rot_x_10/gt/640_vase.gif" alt="Input RGB" width="200"/>  | <img src="output/cls/pred/640_vase.gif" alt="Input RGB" width="200"/>  | <img src="output/cls_rotation/along_x/cls_rot_x_45/pred/640_vase.gif" alt="Input RGB" width="200"/>  | <img src="output/cls_rotation/along_x/cls_rot_x_90/pred/640_vase.gif" alt="Input RGB" width="200"/>  |
| Lamp  |  <img src="output/cls_rotation/along_x/cls_rot_x_10/gt/790_lamp.gif" alt="Input RGB" width="200"/>  | <img src="output/cls/pred/790_lamp.gif" alt="Input RGB" width="200"/>  |  <img src="output/cls_rotation/along_x/cls_rot_x_45/pred/790_lamp.gif" alt="Input RGB" width="200"/>  |  <img src="output/cls_rotation/along_x/cls_rot_x_90/pred/790_lamp.gif" alt="Input RGB" width="200"/>  | 
| Test Accuracy  | -  | __97.58%__ | __35.88%__  |  __44.91%__  | 

Test Accuracy with varying rotations (Classification) - 

![Test Accuracy with varying rotations (Classification)](output/cls_rotation/cls_rot_chart.png)


#### Visualizations - Segmentation

Rotation along x, y, and z:

| Class | Ground Truth | 0&deg; | 45&deg; | 90&deg; | 
|-------|--------------|------------|--------------|------------|
| Example 1 | <img src="output/seg_rotation/all_xyz/seg_rot_all_10/gt/0_chair.gif" alt="Input RGB" width="200"/> | <img src="output/seg/pred/0_chair.gif" alt="Input RGB" width="200"/> | <img src="output/seg_rotation/all_xyz/seg_rot_all_45/pred/0_chair.gif" alt="Input RGB" width="200"/> | <img src="output/seg_rotation/all_xyz/seg_rot_all_90/pred/0_chair.gif" alt="Input RGB" width="200"/> | 
| Example 2  | <img src="output/seg_rotation/all_xyz/seg_rot_all_10/gt/400_chair.gif" alt="Input RGB" width="200"/>  | <img src="output/seg/pred/400_chair.gif" alt="Input RGB" width="200"/>  | <img src="output/seg_rotation/all_xyz/seg_rot_all_45/pred/400_chair.gif" alt="Input RGB" width="200"/>  | <img src="output/seg_rotation/all_xyz/seg_rot_all_90/pred/400_chair.gif" alt="Input RGB" width="200"/>  |
| Example 3  |  <img src="output/seg_rotation/all_xyz/seg_rot_all_10/gt/600_chair.gif" alt="Input RGB" width="200"/>  | <img src="output/seg/pred/600_chair.gif" alt="Input RGB" width="200"/>  |  <img src="output/seg_rotation/all_xyz/seg_rot_all_45/pred/600_chair.gif" alt="Input RGB" width="200"/>  |  <img src="output/seg_rotation/all_xyz/seg_rot_all_90/pred/600_chair.gif" alt="Input RGB" width="200"/>  | 
| Test Accuracy  | -  | __88.52%__ | __59.40%__  |  __61.60%__  | 

Rotation along x only:

| Class | Ground Truth | 0&deg; | 45&deg; | 90&deg; | 
|-------|--------------|------------|--------------|------------|
| Example 1 | <img src="output/seg_rotation/along_x/seg_rot_x_10/gt/200_chair.gif" alt="Input RGB" width="200"/> | <img src="output/seg/pred/200_chair.gif" alt="Input RGB" width="200"/> | <img src="output/seg_rotation/along_x/seg_rot_x_45/pred/200_chair.gif" alt="Input RGB" width="200"/> | <img src="output/seg_rotation/along_x/seg_rot_x_90/pred/200_chair.gif" alt="Input RGB" width="200"/> | 
| Example 2  | <img src="output/seg_rotation/along_x/seg_rot_x_10/gt/150_chair.gif" alt="Input RGB" width="200"/>  | <img src="output/seg/pred/150_chair.gif" alt="Input RGB" width="200"/>  | <img src="output/seg_rotation/along_x/seg_rot_x_45/pred/150_chair.gif" alt="Input RGB" width="200"/>  | <img src="output/seg_rotation/along_x/seg_rot_x_90/pred/150_chair.gif" alt="Input RGB" width="200"/>  |
| Example 3  |  <img src="output/seg_rotation/along_x/seg_rot_x_10/gt/500_chair.gif" alt="Input RGB" width="200"/>  | <img src="output/seg/pred/500_chair.gif" alt="Input RGB" width="200"/>  |  <img src="output/seg_rotation/along_x/seg_rot_x_45/pred/500_chair.gif" alt="Input RGB" width="200"/>  |  <img src="output/seg_rotation/along_x/seg_rot_x_90/pred/500_chair.gif" alt="Input RGB" width="200"/>  | 
| Test Accuracy  | -  | __88.52%__ | __35.88%__  |  __16.25%__  | 

Test Accuracy with varying rotations (Segmentation) - 

![Test Accuracy with varying rotations (Classification)](output/seg_rotation/seg_rot_chart.png)

#### Interpretation
-  In the classification task, an intriguing pattern emerges when point clouds are rotated at angles to 90° along all the axes, even though the final orientation is very similar to the one without rotations, the accuracy of the model falls to half.  Interestingly, when the point clouds are completely inverted, the model's performance improves slightly compared to the 90° rotations.  Another observation was that the accuracy falls drastically when point clouds are only rotated along x compared to other directions, even the combined one. 
  
- In the segmentation task, the model heavily relies on spatial features. Notably, it consistently segments the lower portion of the point cloud as chair legs, even in scenarios involving conventional chair structures.


### Varying Number of Points

Evaluating the classification and segmentation models with varying the number of points present in the point clouds.

#### Procedure

Feed the --num_points argument while running eval_cls.py and eval_seg.py to vary the number of points.

```
  $  python3 eval_cls.py --num_points <n>
  $  python3 eval_seg.py --num_points <n>
     #where n = 10, 100, 500, 1000
```

#### Visualizations - Classification

| Class | Ground Truth | 10 points | 100 points | 1000 points | 
|-------|--------------|------------|--------------|------------|
| Chair | <img src="output/cls/gt/425_chair.gif" alt="Input RGB" width="200"/> | <img src="output/cls_num_points/cls_points_10/pred/425_lamp.gif" alt="Input RGB" width="200"/> | <img src="output/cls_num_points/cls_points_100/pred/425_chair.gif" alt="Input RGB" width="200"/> | <img src="output/cls_num_points/cls_points_1000/pred/425_chair.gif" alt="Input RGB" width="200"/> | 
| Vase  | <img src="output/cls/gt/700_vase.gif" alt="Input RGB" width="200"/>  | <img src="output/cls_num_points/cls_points_10/pred/700_lamp.gif" alt="Input RGB" width="200"/>  | <img src="output/cls_num_points/cls_points_100/pred/700_vase.gif" alt="Input RGB" width="200"/>  | <img src="output/cls_num_points/cls_points_1000/pred/700_vase.gif" alt="Input RGB" width="200"/>  |
| Lamp  |  <img src="output/cls/gt/850_lamp.gif" alt="Input RGB" width="200"/>  | <img src="output/cls_num_points/cls_points_10/pred/850_lamp.gif" alt="Input RGB" width="200"/>  |  <img src="output/cls_num_points/cls_points_100/pred/850_lamp.gif" alt="Input RGB" width="200"/>  |  <img src="output/cls_num_points/cls_points_1000/pred/850_lamp.gif" alt="Input RGB" width="200"/>  | 
| Test Accuracy  | -  | __25.05%__ | __91.08%__  |  __96.95%__  | 

Test Accuracy with varying num_points (Classification)

![Test Accuracy with varying num_points (Classification)](output/cls_num_points/cls_num_points_chart.png)  

#### Visualizations - Segmentation

| Class | Ground Truth | 10 points | 100 points | 1000 points | 
|-------|--------------|------------|--------------|------------|
| Example 1 | <img src="output/seg/gt/0_chair.gif" alt="Input RGB" width="200"/> | <img src="output/seg_num_points/seg_points_10/pred/0_chair.gif" alt="Input RGB" width="200"/> | <img src="output/seg_num_points/seg_points_100/pred/0_chair.gif" alt="Input RGB" width="200"/> | <img src="output/seg_num_points/seg_points_1000/pred/0_chair.gif" alt="Input RGB" width="200"/> | 
| Example 2  | <img src="output/seg/gt/400_chair.gif" alt="Input RGB" width="200"/>  | <img src="output/seg_num_points/seg_points_10/pred/400_chair.gif" alt="Input RGB" width="200"/>  | <img src="output/seg_num_points/seg_points_100/pred/400_chair.gif" alt="Input RGB" width="200"/>  | <img src="output/seg_num_points/seg_points_1000/pred/400_chair.gif" alt="Input RGB" width="200"/>  |
| Example 3  |  <img src="output/seg/gt/600_chair.gif" alt="Input RGB" width="200"/>  | <img src="output/seg_num_points/seg_points_10/pred/600_chair.gif" alt="Input RGB" width="200"/>  |  <img src="output/seg_num_points/seg_points_100/pred/600_chair.gif" alt="Input RGB" width="200"/>  |  <img src="output/seg_num_points/seg_points_1000/pred/600_chair.gif" alt="Input RGB" width="200"/>  | 
| Test Accuracy  | -  | __67.84%__ | __79.93%__  |  __84.86%__  | 

Test Accuracy with varying num_points (Segmentation)
![Test Accuracy with varying num_points (Segmentation)](output/seg_num_points/seg_num_points_chart.png)

#### Interpretation
- In the classification task, the model performs terribly when the number of points is too low, for --num_points = 10, the model classifies the majority of the inputs as lamps. However, the accuracy of the model rises exponentially by just increasing the number of points to 100, and the rise in accuracy becomes stagnant after a point.  

- Unlike the classification task, the segmentation task does not perform too badly for a smaller number of points, and the increase in accuracy is almost linear throughout until it reaches its optimum level, unlike the exponential rise seen in the classification task.

