import numpy as np
import cv2
import math

# weights for channels: [confidence, x, y, yaw, l, w, speed]
reweight_array = np.array([1.0, 3.5, 3.5, 2.0, 3.5, 2.0, 8.0])


def add_rect(img, loc, ori, box, value, pixels_per_meter, max_distance, color):
    """
    Paint an oriented rectangle onto an occupancy image.

    Args:
        img (np.ndarray): Canvas image the rectangle is drawn onto (modified in place).
        loc (np.ndarray): Center position of the box in meters relative to ego.
        ori (np.ndarray): Unit vector indicating the longitudinal direction of the box.
        box (np.ndarray): Half-lengths of the box along longitudinal and lateral axes (meters).
        value (float): Scalar intensity multiplier applied to `color`.
        pixels_per_meter (int): Resolution that maps meters to pixels.
        max_distance (int): Half-size of the rendered map in meters.
        color (Sequence[float]): Base RGB color components in range [0, 1].

    Returns:
        np.ndarray: The updated image buffer with the rectangle filled in.
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


def convert_grid_to_xy(i, j):
    """
    Convert grid indices from the detector output into metric offsets.

    Args:
        i (int): Row index in detector space.
        j (int): Column index in detector space.

    Returns:
        Tuple[float, float]: Cartesian `(x, y)` coordinates in meters.
    """
    x = j - 9.5
    y = 17.5 - i
    return x, y


def find_peak_box(data):
    """
    Identify peak detections and categorize them by object type.

    Args:
        data (np.ndarray): Detector tensor of shape (20, 20, 7) scaled to highlight salient channels.

    Returns:
        Tuple[List[Tuple[int, int]], Dict[str, List[Tuple[int, int]]]]:
            All peak grid indices and a mapping of object categories to their indices.
    """
    det_data = np.zeros((22, 22, 7))
    det_data[1:21, 1:21] = data
    det_data[19:21, 1:21, 0] -= 0.1
    res = []
    for i in range(1, 21):
        for j in range(1, 21):
            if det_data[i, j, 0] > 0.9 or (
                det_data[i, j, 0] > 0.4         # Non-maximum suppression
                and det_data[i, j, 0] > det_data[i, j - 1, 0]
                and det_data[i, j, 0] > det_data[i, j + 1, 0]
                and det_data[i, j, 0] > det_data[i + 1, j + 1, 0]
                and det_data[i, j, 0] > det_data[i - 1, j + 1, 0]
                and det_data[i, j, 0] > det_data[i + 1, j - 1, 0]
                and det_data[i, j, 0] > det_data[i + 1, j + 1, 0]
                and det_data[i, j, 0] > det_data[i - 1, j, 0]
                and det_data[i, j, 0] > det_data[i + 1, j, 0]
            ):
                res.append((i - 1, j - 1))
    box_info = {"car": [], "bike": [], "pedestrian": []}
    for instance in res:
        i, j = instance
        box = np.array(det_data[i + 1, j + 1, 4:6])
        if box[0] > 2.0:
            box_info["car"].append((i, j))
        elif box[0] / box[1] > 1.5:
            box_info["bike"].append((i, j))
        else:
            box_info["pedestrian"].append((i, j))
    return res, box_info


def render_self_car(loc, ori, box, pixels_per_meter=5, max_distance=18, color=None):
    """
    Render the ego vehicle footprint into an image buffer.

    Args:
        loc (np.ndarray): Ego position in meters relative to image center.
        ori (np.ndarray): Unit vector representing vehicle heading.
        box (np.ndarray): Half-lengths of the ego bounding box (meters).
        pixels_per_meter (int, optional): Resolution scaling factor. Defaults to 5.
        max_distance (int, optional): Half-size of the rendered map in meters. Defaults to 18.
        color (Optional[Sequence[float]]): RGB color factors; defaults to white if omitted.

    Returns:
        np.ndarray: Single-channel occupancy image of the rendered ego car.
    """
    # Full image size
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


def render(det_data, pixels_per_meter=5, max_distance=18, t=0):
    """
    Produce an occupancy map of surrounding actors given detector predictions.

    Args:
        det_data (np.ndarray): Detector tensor of shape (20, 20, 7) describing nearby objects.
        pixels_per_meter (int, optional): Resolution scaling factor. Defaults to 5.
        max_distance (int, optional): Half-size of the rendered map in meters. Defaults to 18.
        t (int, optional): Future timestep used for simple motion extrapolation. Defaults to 0.

    Returns:
        Tuple[np.ndarray, Dict[str, int]]: Rendered occupancy image and counts per object category.
    """
    det_data = det_data * reweight_array
    box_ids, box_info = find_peak_box(det_data)
    img_size = max_distance * pixels_per_meter * 2
    img = np.zeros((img_size, img_size, 3), np.uint8)
    # point of interest
    for poi in box_ids:
        i, j = poi
        if poi in box_info['bike']:
            speed = max(4, det_data[i,j,6])
        else:
            speed = det_data[i, j, 6]
        center_x, center_y = convert_grid_to_xy(i, j)
        act_img = np.zeros((img_size, img_size, 3), np.uint8)
        theta = det_data[i, j, 3] * np.pi
        ori = np.array([math.cos(theta), math.sin(theta)])
        loc_x = center_x + det_data[i, j, 1] + t * speed * ori[0]
        loc_y = center_y + det_data[i, j, 2] - t * speed * ori[1]
        loc = np.array([loc_x, -loc_y])
        box = np.array(det_data[i, j, 4:6])
        box[1] = max(0.4, box[1])
        if box[0] < 1.5:
            box = box * 2
        color = np.array([1, 1, 1])
        new_img = add_rect(
            act_img, loc, ori, box, 255, pixels_per_meter, max_distance, color
        )
        act_img = np.clip(act_img, 0, 255)
        img = img + act_img
    img = np.clip(img, 0, 255)[:, :, 0]
    img = img.astype(np.uint8)

    box_info["car"] = len(box_info["car"])
    box_info["bike"] = len(box_info["bike"])
    box_info["pedestrian"] = len(box_info["pedestrian"])
    return img, box_info


def render_waypoints(waypoints, pixels_per_meter=5, max_distance=18, color=(0, 255, 0)):
    """
    Draw planned waypoints as circles in an occupancy image.

    Args:
        waypoints (Sequence[np.ndarray]): Iterable of 2D waypoint coordinates in meters.
        pixels_per_meter (int, optional): Resolution scaling factor. Defaults to 5.
        max_distance (int, optional): Half-size of the rendered map in meters. Defaults to 18.
        color (Tuple[int, int, int], optional): BGR color used for waypoint markers. Defaults to green.

    Returns:
        np.ndarray: Color image with waypoint markers rendered.
    """
    img_size = max_distance * pixels_per_meter * 2
    img = np.zeros((img_size, img_size, 3), np.uint8)
    for i in range(len(waypoints)):
        new_loc = waypoints[i]
        new_loc = new_loc * pixels_per_meter + pixels_per_meter * max_distance
        new_loc = np.around(new_loc)
        new_loc = tuple(new_loc.astype(np.int))
        img = cv2.circle(img, new_loc, 6, color, -1)
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    return img
