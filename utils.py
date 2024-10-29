import os
import torch
import pytorch3d
from pytorch3d.renderer import (
    AlphaCompositor,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
)
import imageio
import numpy as np
from PIL import Image, ImageDraw


def save_checkpoint(epoch, model, args, best=False):
    if best:
        path = os.path.join(args.checkpoint_dir, 'best_model.pt')
    else:
        path = os.path.join(args.checkpoint_dir, 'model_epoch_{}.pt'.format(epoch))
    torch.save(model.state_dict(), path)

def create_dir(directory):
    """
    Creates a directory if it does not already exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_points_renderer(
    image_size=256, device=None, radius=0.01, background_color=(1, 1, 1)
):
    """
    Returns a Pytorch3D renderer for point clouds.

    Args:
        image_size (int): The rendered image size.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
        radius (float): The radius of the rendered point in NDC.
        background_color (tuple): The background color of the rendered image.
    
    Returns:
        PointsRenderer.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = PointsRasterizationSettings(image_size=image_size, radius=radius,)
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )
    return renderer


def viz_cls_seg (verts, labels, path, device, num_points, cls_seg, label_text):
    """
    visualize segmentation result
    output: a 360-degree gif
    """
    image_size=256
    background_color=(1, 1, 1)

    # chair:- red , vase:- green, lamp:- blue, 
    cls_colors_gt = [[1.0,0.0,0.0], [0.0,1.0,0.0], [0.0,0.0,1.0]]
    cls_colors_pred = [[0.7,0.0,0.0], [0.0,0.7,0.0], [0.0,0.0,0.7]]
    colors = [[1.0,1.0,1.0], [1.0,0.5,0.0], [1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0], [1.0,1.0,0.0]]

    # Construct various camera viewpoints
    dist = 3
    elev = 0
    azim = [180 - 12*i for i in range(30)]
    R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=dist, elev=elev, azim=azim, device=device)
    c = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, fov=60, device=device)

    sample_verts = verts.unsqueeze(0).repeat(30,1,1).to(torch.float)
    sample_labels = labels.unsqueeze(0)
    sample_colors = torch.zeros((1,num_points,3))

    if cls_seg == "cls_gt":
        # Colorize points based on classification labels
        for i in range(3):
            sample_colors[sample_labels==i] = torch.tensor(cls_colors_gt[i])
        sample_colors = sample_colors.repeat(sample_verts.shape[0],1,1).to(torch.float)
    elif cls_seg == "cls_pred":
        # Colorize points based on classification labels
        for i in range(3):
            sample_colors[sample_labels==i] = torch.tensor(cls_colors_pred[i])
        sample_colors = sample_colors.repeat(sample_verts.shape[0],1,1).to(torch.float)
    else:
        # Colorize points based on segmentation labels
        for i in range(6):
            sample_colors[sample_labels==i] = torch.tensor(colors[i])
        sample_colors = sample_colors.repeat(30,1,1).to(torch.float)

    point_cloud = pytorch3d.structures.Pointclouds(points=sample_verts, features=sample_colors).to(device)

    renderer = get_points_renderer(image_size=image_size, background_color=background_color, device=device)
    rend = renderer(point_cloud, cameras=c).cpu().numpy() # (30, 256, 256, 3)
    rend = (rend * 255).astype(np.uint8)

    # Add text labels to each frame
    for i in range(rend.shape[0]):
        img = Image.fromarray(rend[i])
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), label_text, (0, 0, 0))  # Text position and color
        rend[i] = np.array(img)
    imageio.mimsave(path, rend, duration=180, loop=0)
