import math
import json
import os

from tqdm import tqdm
from PIL import Image
import cv2
import numpy as np
import torch
import copy

from skimage.measure import block_reduce
# VALUES = [255, 150, 120, 90, 60, 30][::-1]
# EXTENT = [0, 0.2, 0.4, 0.6, 0.8, 1.0][::-1]


VALUES = [255]
EXTENT = [0]


def add_rect(img, loc, ori, box, value, pixels_per_meter, max_distance, color):
    """
    Rasterize an oriented bounding box onto a bird's-eye-view canvas.

    Args:
        img (np.ndarray): Image buffer to be updated in place.
        loc (np.ndarray): 2D center location in meters relative to ego.
        ori (np.ndarray): Unit heading vector pointing along the longitudinal axis.
        box (np.ndarray): Half-lengths along longitudinal and lateral directions (meters).
        value (float): Intensity multiplier applied to the color channels.
        pixels_per_meter (int): Resolution converting meters to pixels.
        max_distance (float): Half-size of the rendered square in meters.
        color (Sequence[float]): RGB scaling factors within [0, 1].

    Returns:
        np.ndarray: The modified image buffer containing the filled polygon.
    """
    img_size = max_distance * pixels_per_meter * 2
    vet_ori = np.array([-ori[1], ori[0]])
    hor_offset = box[0] * ori
    vet_offset = box[1] * vet_ori
    left_up = (loc + hor_offset + vet_offset + max_distance) * pixels_per_meter
    left_down = (loc + hor_offset - vet_offset + max_distance) * pixels_per_meter
    right_up = (loc - hor_offset + vet_offset + max_distance) * pixels_per_meter
    right_down = (loc - hor_offset - vet_offset + max_distance) * pixels_per_meter
    left_up = np.around(left_up).astype(np.int)
    left_down = np.around(left_down).astype(np.int)
    right_down = np.around(right_down).astype(np.int)
    right_up = np.around(right_up).astype(np.int)
    left_up = list(left_up)
    left_down = list(left_down)
    right_up = list(right_up)
    right_down = list(right_down)
    color = [int(x) for x in value * color]
    cv2.fillConvexPoly(img, np.array([left_up, left_down, right_down, right_up]), color)
    return img


def get_yaw_angle(forward_vector):
    """
    Convert a forward-direction vector into a yaw angle.

    Args:
        forward_vector (np.ndarray): 2D direction vector.

    Returns:
        float: Heading in radians within [0, 2π).
    """
    forward_vector = forward_vector / np.linalg.norm(forward_vector)
    yaw = math.acos(forward_vector[0])
    if forward_vector[1] < 0:
        yaw = 2 * np.pi - yaw
    return yaw


def generate_future_waypoints(measurements, pixels_per_meter=5, max_distance=30):
    """
    Render planned waypoints into a binary BEV image centered on the ego vehicle.

    Args:
        measurements (Dict[str, Any]): Contains keys `gps_x`, `gps_y`, `theta`, and `future_waypoints`.
        pixels_per_meter (int, optional): Resolution of the BEV grid. Defaults to 5.
        max_distance (float, optional): Half-width of the rendered map in meters. Defaults to 30.

    Returns:
        np.ndarray: Single-channel image marking future waypoints.
    """
    img_size = max_distance * pixels_per_meter * 2
    img = np.zeros((img_size, img_size), np.uint8)
    ego_x = measurements["gps_x"]
    ego_y = measurements["gps_y"]
    ego_theta = measurements["theta"] + np.pi / 2
    R = np.array(
        [
            [np.cos(ego_theta), -np.sin(ego_theta)],
            [np.sin(ego_theta), np.cos(ego_theta)],
        ]
    )
    for waypoint in measurements["future_waypoints"]:
        new_loc = R.T.dot(np.array([waypoint[0] - ego_x, waypoint[1] - ego_y]))
        if new_loc[0] ** 2 + new_loc[1] ** 2 > (max_distance + 3) ** 2 * 2:
            break
        # new_loc_2 = np.zeros(2)
        # new_loc_2[0] = new_loc[1]
        # new_loc_2[1] = new_loc[0]
        # new_loc = new_loc_2
        new_loc = new_loc * pixels_per_meter + pixels_per_meter * max_distance
        new_loc = np.around(new_loc)
        new_loc = tuple(new_loc.astype(np.int))
        img = cv2.circle(img, new_loc, 3, 255, -1)
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    return img


def generate_heatmap_multiclass(measurements, actors_data, max_distance=30, pixels_per_meter=8):
    """
    Produce separate occupancy heatmaps for each semantic actor class.

    Args:
        measurements (Dict[str, Any]): Ego measurement dictionary used for alignment.
        actors_data (Dict[int, Dict[str, Any]]): Actor metadata keyed by ID with `tpe` labels.
        max_distance (float, optional): Half-size of the rendered area in meters. Defaults to 30.
        pixels_per_meter (int, optional): Resolution of the heatmap. Defaults to 8.

    Returns:
        Dict[int, np.ndarray]: Mapping from class ID to occupancy heatmap.
    """
    actors_data_multiclass = {
        0: {}, 1: {}, 2: {}
    }
    for _id in actors_data.keys():
        actors_data_multiclass[actors_data[_id]['tpe']][_id] = actors_data[_id]
    heatmap_0 = generate_heatmap(measurements, actors_data_multiclass[0], max_distance, pixels_per_meter)
    heatmap_1 = generate_heatmap(measurements, actors_data_multiclass[1], max_distance, pixels_per_meter)
    return {0: heatmap_0, 1: heatmap_1}


def generate_heatmap(measurements, actors_data, max_distance=30, pixels_per_meter=8):
    """
    Render surrounding actors into a monochrome occupancy heatmap in ego coordinates.

    Args:
        measurements (Dict[str, Any]): Includes ego lidar pose, heading, and traffic light info.
        actors_data (Dict[int, Dict[str, Any]]): Actor state dictionaries (location, orientation, box, type).
        max_distance (float, optional): Half-width of the receptive field in meters. Defaults to 30.
        pixels_per_meter (int, optional): Map resolution. Defaults to 8.

    Returns:
        np.ndarray: Single-channel occupancy image.
    """
    img_size = max_distance * pixels_per_meter * 2
    img = np.zeros((img_size, img_size, 3), np.int)
    ego_x = measurements["lidar_pose_x"]
    ego_y = measurements["lidar_pose_y"]
    ego_theta = measurements["theta"]
    R = np.array(
        [
            [np.cos(ego_theta), -np.sin(ego_theta)],
            [np.sin(ego_theta), np.cos(ego_theta)],
        ]
    )
    ego_id = None
    for _id in actors_data:
        color = np.array([1, 1, 1])
        if actors_data[_id]["tpe"] == 2:
            if int(_id) == int(measurements["affected_light_id"]):
                if actors_data[_id]["sta"] == 0:
                    color = np.array([1, 1, 1])
                else:
                    color = np.array([0, 0, 0])
                yaw = get_yaw_angle(actors_data[_id]["ori"])
                TR = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
                actors_data[_id]["loc"] = np.array(
                    actors_data[_id]["loc"][:2]
                ) + TR.T.dot(np.array(actors_data[_id]["taigger_loc"])[:2])
                actors_data[_id]["ori"] = np.array(actors_data[_id]["ori"])
                actors_data[_id]["box"] = np.array(actors_data[_id]["trigger_box"]) * 2
            else:
                continue
        raw_loc = actors_data[_id]["loc"]
        if (raw_loc[0] - ego_x) ** 2 + (raw_loc[1] - ego_y) ** 2 <= 2:
            ego_id = _id
            color = np.array([0, 1, 1])
        new_loc = R.T.dot(np.array([raw_loc[0] - ego_x, raw_loc[1] - ego_y]))
        actors_data[_id]["loc"] = np.array(new_loc)
        raw_ori = actors_data[_id]["ori"]
        new_ori = R.T.dot(np.array([raw_ori[0], raw_ori[1]]))
        actors_data[_id]["ori"] = np.array(new_ori)
        actors_data[_id]["box"] = np.array(actors_data[_id]["box"])
        if int(_id) in measurements["is_vehicle_present"]:
            color = np.array([1, 1, 1])
        elif int(_id) in measurements["is_bike_present"]:
            color = np.array([1, 1, 1])
        elif int(_id) in measurements["is_junction_vehicle_present"]:
            color = np.array([1, 1, 1])
        elif int(_id) in measurements["is_pedestrian_present"]:
            color = np.array([1, 1, 1])
        actors_data[_id]["color"] = color

    if ego_id is not None and ego_id in actors_data:
        del actors_data[ego_id]  # Do not show ego car
    for _id in actors_data:
        if actors_data[_id]["tpe"] == 2:
            continue  # FIXME donot add traffix light
            if int(_id) != int(measurements["affected_light_id"]):
                continue
            if actors_data[_id]["sta"] != 0:
                continue
        act_img = np.zeros((img_size, img_size, 3), np.uint8)
        loc = actors_data[_id]["loc"][:2]
        ori = actors_data[_id]["ori"][:2]
        box = actors_data[_id]["box"]
        if box[0] < 1.5:
            box = box * 1.5  # FIXME enlarge the size of pedstrian and bike
        color = actors_data[_id]["color"]
        for i in range(len(VALUES)):
            act_img = add_rect(
                act_img,
                loc,
                ori,
                box + EXTENT[i],
                VALUES[i],
                pixels_per_meter,
                max_distance,
                color,
            )
        act_img = np.clip(act_img, 0, 255)
        img = img + act_img
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    img = img[:, :, 0]
    return img


def generate_relative_heatmap(measurements, actors_data, egp_pos, pixels_per_meter=5, max_distance=30, judge_visibility=False):
    """
    Render actors relative to an arbitrary observation pose (e.g., cooperative agent frame).

    Args:
        measurements (Dict[str, Any]): Measurement dictionary carrying traffic light context.
        actors_data (Dict[int, Dict[str, Any]]): Actor metadata with fields like `loc`, `ori`, `box`, `lidar_visible`.
        egp_pos (Dict[str, float]): Reference pose with keys `x`, `y`, and `theta`.
        pixels_per_meter (int, optional): Grid resolution. Defaults to 5.
        max_distance (float, optional): Half-width of the rendered area in meters. Defaults to 30.
        judge_visibility (bool, optional): Skip actors not visible to lidar when True. Defaults to False.

    Returns:
        np.ndarray: Single-channel occupancy map centered on `egp_pos`.
    """
    img_size = max_distance * pixels_per_meter * 2
    img = np.zeros((img_size, img_size, 3), np.int)
    ego_x = egp_pos["x"]
    ego_y = egp_pos["y"]
    ego_theta = egp_pos["theta"]
    R = np.array(
        [
            [np.cos(ego_theta), -np.sin(ego_theta)],
            [np.sin(ego_theta), np.cos(ego_theta)],
        ]
    )
    ego_id = None
    for _id in actors_data:
        color = np.array([1, 1, 1])
        if actors_data[_id]["tpe"] == 2:
            if int(_id) == int(measurements["affected_light_id"]):
                if actors_data[_id]["sta"] == 0:
                    color = np.array([1, 1, 1])
                else:
                    color = np.array([0, 0, 0])
                yaw = get_yaw_angle(actors_data[_id]["ori"])
                TR = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
                actors_data[_id]["loc"] = np.array(
                    actors_data[_id]["loc"][:2]
                ) + TR.T.dot(np.array(actors_data[_id]["taigger_loc"])[:2])
                actors_data[_id]["ori"] = np.array(actors_data[_id]["ori"])
                actors_data[_id]["box"] = np.array(actors_data[_id]["trigger_box"]) * 2
            else:
                continue
        raw_loc = actors_data[_id]["loc"]
        if (raw_loc[0] - measurements["x"]) ** 2 + (raw_loc[1] - measurements["y"]) ** 2 <= 1:
            ego_id = _id
            color = np.array([0, 1, 1])
        new_loc = R.T.dot(np.array([raw_loc[0] - ego_x, raw_loc[1] - ego_y]))
        actors_data[_id]["loc"] = np.array(new_loc)
        raw_ori = actors_data[_id]["ori"]
        new_ori = R.T.dot(np.array([raw_ori[0], raw_ori[1]]))
        actors_data[_id]["ori"] = np.array(new_ori)
        actors_data[_id]["box"] = np.array(actors_data[_id]["box"])
        color = np.array([1, 1, 1])
        actors_data[_id]["color"] = color

    if ego_id is not None and ego_id in actors_data:
        del actors_data[ego_id]  # Do not show ego car
    for _id in actors_data:
        if judge_visibility and actors_data[_id]["lidar_visible"]==0:
            continue
        if actors_data[_id]["tpe"] == 2:
            continue  # FIXME donot add traffix light
            if int(_id) != int(measurements["affected_light_id"]):
                continue
            if actors_data[_id]["sta"] != 0:
                continue
        act_img = np.zeros((img_size, img_size, 3), np.uint8)
        loc = actors_data[_id]["loc"][:2]
        ori = actors_data[_id]["ori"][:2]
        box = actors_data[_id]["box"]
        if box[0] < 1.5:
            box = box * 1.5  # FIXME enlarge the size of pedstrian and bike
        color = actors_data[_id]["color"]
        for i in range(len(VALUES)):
            act_img = add_rect(
                act_img,
                loc,
                ori,
                box + EXTENT[i],
                VALUES[i],
                pixels_per_meter,
                max_distance,
                color,
            )
        act_img = np.clip(act_img, 0, 255)
        img = img + act_img
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    img = img[:, :, 0]
    return img



def render_self_car(loc, ori, box, pixels_per_meter=5, max_distance=36, color=None):
    """
    Draw the ego vehicle's footprint as a filled rectangle in BEV space.

    Args:
        loc (np.ndarray): 2D position of the vehicle center in meters.
        ori (np.ndarray): Unit orientation vector pointing forward.
        box (np.ndarray): Half-lengths representing vehicle dimensions.
        pixels_per_meter (int, optional): Rasterization resolution. Defaults to 5.
        max_distance (float, optional): Half-size of the rendered map in meters. Defaults to 36.
        color (Optional[Sequence[float]]): RGB scaling coefficients; white if None.

    Returns:
        np.ndarray: Single-channel mask or RGB image depending on `color`.
    """
    img_size = max_distance * pixels_per_meter * 2
    img = np.zeros((img_size, img_size, 3), np.uint8)
    if color is None:
        color = np.array([1, 1, 1])
        new_img = add_rect(
            img, loc, ori, box, 255, pixels_per_meter, max_distance, color
        )
        return new_img[:, :, 0]
    else:
        color = np.array(color)
        new_img = add_rect(
            img, loc, ori, box, 255, pixels_per_meter, max_distance, color
        )
        return new_img
    


def convert_grid_to_xy(i, j, det_range):
    """
    Convert detection grid indices into metric coordinates.

    Args:
        i (int): Row index within the detection tensor.
        j (int): Column index within the detection tensor.
        det_range (Sequence[float]): `[front, back, right, left, resolution]` definition.

    Returns:
        Tuple[float, float]: `(x, y)` location in meters relative to ego.
    """
    x = det_range[4]*(j + 0.5) - det_range[2]
    y = det_range[0] - det_range[4]*(i+0.5)
    return x, y


def generate_det_data(
    heatmap, measurements, actors_data, det_range=[30,10,10,10, 0.8]
):
    """
    Assemble detection features (confidence, offsets, size, yaw, speed) per grid cell.

    Args:
        heatmap (np.ndarray): High-resolution occupancy heatmap.
        measurements (Dict[str, Any]): Ego lidar pose (`lidar_pose_x`, `lidar_pose_y`, `theta`).
        actors_data (Dict[int, Dict[str, Any]]): Actor annotations including `loc`, `ori`, `box`, and `vel`.
        det_range (Sequence[float], optional): `[front, back, right, left, resolution]` specification.

    Returns:
        np.ndarray: Detection tensor of shape `(H, W, 7)`.
    """
    res = det_range[4]
    max_distance = max(det_range)
    traffic_heatmap = block_reduce(heatmap, block_size=(int(8*res), int(8*res)), func=np.mean)
    traffic_heatmap = np.clip(traffic_heatmap, 0.0, 255.0)
    traffic_heatmap = traffic_heatmap[:int((det_range[0]+det_range[1])/res), int((max_distance-det_range[2])/res):int((max_distance+det_range[3])/res)]
    det_data = np.zeros((int((det_range[0]+det_range[1])/res), int((det_range[2]+det_range[3])/res), 7)) # (50,25,7)
    vertical, horizontal = det_data.shape[:2]

    ego_x = measurements["lidar_pose_x"]
    ego_y = measurements["lidar_pose_y"]
    ego_theta = measurements["theta"]
    R = np.array(
        [
            [np.cos(ego_theta), -np.sin(ego_theta)],
            [np.sin(ego_theta), np.cos(ego_theta)],
        ]
    )
    need_deleted_ids = []
    for _id in actors_data:
        raw_loc = actors_data[_id]["loc"]
        new_loc = R.T.dot(np.array([raw_loc[0] - ego_x, raw_loc[1] - ego_y]))
        new_loc[1] = -new_loc[1]
        actors_data[_id]["loc"] = np.array(new_loc)
        raw_ori = actors_data[_id]["ori"]
        new_ori = R.T.dot(np.array([raw_ori[0], raw_ori[1]]))
        dis = new_loc[0] ** 2 + new_loc[1] ** 2
        if (
            dis <= 2
            or dis >= (max_distance) ** 2 * 2
            or "box" not in actors_data[_id]
        ):
            need_deleted_ids.append(_id)
            continue
        actors_data[_id]["ori"] = np.array(new_ori)
        actors_data[_id]["box"] = np.array(actors_data[_id]["box"])

    for _id in need_deleted_ids:
        del actors_data[_id]

    for i in range(vertical):  # 50
        for j in range(horizontal):  # 25
            if traffic_heatmap[i][j] < 0.05 * 255.0:
                continue
            center_x, center_y = convert_grid_to_xy(i, j, det_range)
            min_dis = 1000
            min_id = None
            for _id in actors_data:
                loc = actors_data[_id]["loc"][:2]
                ori = actors_data[_id]["ori"][:2]
                box = actors_data[_id]["box"]
                dis = (loc[0] - center_x) ** 2 + (loc[1] - center_y) ** 2
                if dis < min_dis:
                    min_dis = dis
                    min_id = _id

            if min_id is None:
                continue

            loc = actors_data[min_id]["loc"][:2]
            ori = actors_data[min_id]["ori"][:2]
            box = actors_data[min_id]["box"]
            theta = (get_yaw_angle(ori) / np.pi + 2) % 2
            speed = np.linalg.norm(actors_data[min_id]["vel"])
            prob = np.power(0.5 / max(0.5, np.sqrt(min_dis)), 0.5)
            det_data[i][j] = np.array(
                [
                    prob,
                    (loc[0] - center_x) / 3.5,
                    (loc[1] - center_y) / 3.5,
                    theta / 2.0,
                    box[0] / 3.5,
                    box[1] / 2.0,
                    speed / 8.0,
                ]
            )
    return det_data



def generate_det_data_multiclass(
    heatmap, measurements, actors_data, det_range=[30,10,10,10, 0.8]
):  
    """
    Generate detection tensors separately for each semantic actor class.

    Args:
        heatmap (np.ndarray): Multi-channel heatmap indexed by actor class.
        measurements (Dict[str, Any]): Ego pose metadata used for alignment.
        actors_data (Dict[int, Dict[str, Any]]): Actor states with `tpe` denoting class.
        det_range (Sequence[float], optional): Detection coverage parameters. Defaults to `[30,10,10,10,0.8]`.

    Returns:
        np.ndarray: Array of per-class detection tensors `(num_classes, H, W, 7)`.
    """
    actors_data_multiclass = {
        0: {}, 1: {}, 2: {}
    }
    for _id in actors_data.keys():
        actors_data_multiclass[actors_data[_id]['tpe']][_id] = actors_data[_id]
    det_data = []
    for _class in range(2):
        det_data.append(generate_det_data(heatmap[_class], measurements, actors_data_multiclass[_class], det_range))
    
    return np.array(det_data)


def check_numpy_to_torch(x):
    """
    Convert NumPy arrays to Torch tensors while tracking conversion.

    Args:
        x (Union[np.ndarray, torch.Tensor]): Input array or tensor.

    Returns:
        Tuple[torch.Tensor, bool]: Torch tensor and a flag indicating conversion occurred.
    """
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False



def rotate_points_along_z(points, angle):
    """
    Rotate batched 3D points around the z-axis.

    Args:
        points (Union[np.ndarray, torch.Tensor]): Shape `(B, N, 3 + C)` containing xyz and extra features.
        angle (Union[np.ndarray, torch.Tensor]): Shape `(B,)` rotation angles in radians (positive rotates x→y).

    Returns:
        Union[np.ndarray, torch.Tensor]: Rotated points preserving the input container type.
    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa, sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3].float(), rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot



def boxes_to_corners_3d(boxes3d, order):
    """
    Convert parameterized boxes into their 3D corner coordinates.

    Args:
        boxes3d (Union[np.ndarray, torch.Tensor]): Shape `(N, 7)` array describing
            `[x, y, z, l, w, h, heading]` or `[x, y, z, h, w, l, heading]`.
        order (str): Either `'lwh'` or `'hwl'`, indicating the layout of `(l, w, h)`.

    Returns:
        Union[np.ndarray, torch.Tensor]: Array of shape `(N, 8, 3)` listing each box corner.

    Coordinate frame (OPV2V left-handed):

        4 -------- 5
       /|         /|
      7 -------- 6 .
      | |        | |
      . 0 -------- 1
      |/         |/
      3 -------- 2

        ^ z
        |
        |
        | . x
        |/
        +-------> y
    """

    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)
    boxes3d_ = boxes3d

    if order == 'hwl':
        boxes3d_ = boxes3d[:, [0, 1, 2, 5, 4, 3, 6]]

    template = boxes3d_.new_tensor((
        [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, -1],
        [1, -1, 1], [1, 1, 1], [-1, 1, 1], [-1, -1, 1],
    )) / 2

    corners3d = boxes3d_[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3),
                                                   boxes3d_[:, 6]).view(-1, 8,
                                                                        3)
    corners3d += boxes3d_[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d



def get_points_in_rotated_box_3d(p, box_corner):
    """
    Filter 3D points that lie inside a rotated bounding box.

    Args:
        p (np.ndarray): Input points with shape `(N, 3)`.
        box_corner (np.ndarray): Corner coordinates of the box with shape `(8, 3)`.

    Returns:
        np.ndarray: Subset of `p` that falls inside the box.
    """
    edge1 = box_corner[1, :] - box_corner[0, :]
    edge2 = box_corner[3, :] - box_corner[0, :]
    edge3 = box_corner[4, :] - box_corner[0, :]

    p_rel = p - box_corner[0, :].reshape(1, -1)

    l1 = get_projection_length_for_vector_projection(p_rel, edge1)
    l2 = get_projection_length_for_vector_projection(p_rel, edge2)
    l3 = get_projection_length_for_vector_projection(p_rel, edge3)
    # A point is within the box, if and only after projecting the
    # point onto the two edges s.t. p_rel = [edge1, edge2] @ [l1, l2]^T,
    # we have 0<=l1<=1 and 0<=l2<=1.
    mask1 = np.logical_and(l1 >= 0, l1 <= 1)
    mask2 = np.logical_and(l2 >= 0, l2 <= 1)
    mask3 = np.logical_and(l3 >= 0, l3 <= 1)

    mask = np.logical_and(mask1, mask2)
    mask = np.logical_and(mask, mask3)
    p_in_box = p[mask, :]

    return p_in_box

def get_projection_length_for_vector_projection(a, b):
    """
    Compute scalar projection lengths when projecting vectors `a` onto vector `b`.

    Args:
        a (np.ndarray): Collection of vectors with shape `(N, 2)`.
        b (np.ndarray): Reference vector of shape `(2,)`.

    Returns:
        np.ndarray: Projection lengths such that `a_proj = length * b`.
    """
    assert np.sum(b ** 2, axis=-1) > 1e-6
    length = a.dot(b) / np.sum(b ** 2, axis=-1)
    return length
