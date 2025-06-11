import numpy as np
from scipy.spatial.transform import Rotation as R

def replica2gsplat(point_in_replica, scene_name):
    final_transform = None

    if scene_name == "room_0":
        # Room_0 alignment
        tx, ty, tz = 0.95, -1.14, -0.24
        rot_x_deg, rot_y_deg, rot_z_deg = 0.0, -5.0, 106.0
        scale = 0.55

    elif scene_name == "office_3":
        # Office_3 alignment
        tx, ty, tz = -0.46, -0.51, -0.38
        rot_x_deg, rot_y_deg, rot_z_deg = -5.0, -3.0, 175.0
        scale = 0.49

    elif scene_name == "office_4":
        # Office_4 alignment
        tx, ty, tz = 1.18, 0.23, -0.29
        rot_x_deg, rot_y_deg, rot_z_deg = -2, 1, 168
        scale = 0.53

    else:
        print(f"Manual transform parameters for scene '{scene_name}' are not defined.")
        return None

    # --- Transformation Building ---
    rot_x_rad, rot_y_rad, rot_z_rad = np.deg2rad([rot_x_deg, rot_y_deg, rot_z_deg])
    rotation_matrix = R.from_euler('xyz', [rot_x_rad, rot_y_rad, rot_z_rad]).as_matrix()

    final_transform = np.eye(4)
    final_transform[:3, :3] = rotation_matrix * scale
    final_transform[:3, 3] = [tx, ty, tz]

    point_homogeneous = np.append(point_in_replica, 1.0)
    transformed_homogeneous = final_transform @ point_homogeneous
    return transformed_homogeneous[:3]