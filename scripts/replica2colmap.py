import os
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
import imageio.v2 as imageio
import tqdm

# ======== 1) CONFIG ======== 
input_rgb_dir = "/scratch/auj7tx/vmap/room_0/imap/00/rgb"
input_poses_file = "/scratch/auj7tx/vmap/room_0/imap/00/traj_w_c.txt"
output_dir = "/scratch/auj7tx/colmap/room_0/"

W, H = 1200, 680
fx = fy = 600.0
cx, cy = W / 2, H / 2

# ======== 2) PREP OUTPUT FOLDERS ======== 
images_out = Path(output_dir) / "images"
sparse_out = Path(output_dir) / "sparse" / "0"
images_out.mkdir(parents=True, exist_ok=True)
sparse_out.mkdir(parents=True, exist_ok=True)

# ======== 3) GATHER & RENAME RGB IMAGES ======== 
# Expect files "rgb_0.png" ... "rgb_N.png"
print("Gathering RGB files...")
rgb_files = sorted([f for f in os.listdir(input_rgb_dir) 
                    if f.startswith("rgb_") and f.endswith(".png")])
print(f"Found {len(rgb_files)} RGB images in {input_rgb_dir}")

image_names_colmap = []
print("Copying RGB images to output...")
for idx, fname in enumerate(tqdm.tqdm(rgb_files, desc="Copying images")):
    src = os.path.join(input_rgb_dir, fname)
    dst_name = f"{idx:05d}.png"
    dst_path = images_out / dst_name

    # Copy or read/write
    img = imageio.imread(src)
    imageio.imwrite(dst_path, img)

    image_names_colmap.append(dst_name)

print(f"✅ Copied {len(image_names_colmap)} images to {images_out}")

# ======== 4) LOAD POSES (World->Camera or Camera->World?) ========
print(f"Loading poses from {input_poses_file}...")
poses = []
with open(input_poses_file, "r") as f:
    lines = f.readlines()

# If your file is truly "world->camera," do NOT invert.
# If it's "camera->world," then do: mat = np.linalg.inv(mat).
for line in tqdm.tqdm(lines, desc="Reading poses"):
    mat = np.fromstring(line.strip(), sep=" ").reshape(4, 4)
    poses.append(mat)

print(f"Loaded {len(poses)} poses from file.")

if len(poses) != len(image_names_colmap):
    print(f"⚠ WARNING: Found {len(poses)} poses but {len(image_names_colmap)} images!")

# ======== 5) WRITE cameras.txt ========
print("Writing cameras.txt...")
with open(sparse_out / "cameras.txt", "w") as f:
    # Format: CAMERA_ID MODEL WIDTH HEIGHT fx fy cx cy
    f.write(f"1 PINHOLE {W} {H} {fx} {fy} {cx} {cy}\n")

# ======== 6) WRITE images.txt ========
print("Writing images.txt...")
with open(sparse_out / "images.txt", "w") as f:
    for i, (pose, img_name) in enumerate(zip(poses, image_names_colmap), start=1):
        Rmat = pose[:3, :3]
        tvec = pose[:3, 3]

        # Convert rotation to quaternion (x, y, z, w)
        quat_xyzw = R.from_matrix(Rmat).as_quat()
        # Reorder to (qw, qx, qy, qz)
        qw = quat_xyzw[3]
        qx = quat_xyzw[0]
        qy = quat_xyzw[1]
        qz = quat_xyzw[2]

        # Format: 
        #   IMAGE_ID qw qx qy qz tx ty tz CAMERA_ID IMAGE_NAME
        #   (blank line after)
        f.write(f"{i} {qw} {qx} {qy} {qz} {tvec[0]} {tvec[1]} {tvec[2]} 1 {img_name}\n\n")

# ======== 7) WRITE points3D.txt (EMPTY) ========
print("Writing empty points3D.txt...")
with open(sparse_out / "points3D.txt", "w") as f:
    f.write("# Empty minimal file, just for camera poses.\n")

print("Done! Check output:", output_dir)