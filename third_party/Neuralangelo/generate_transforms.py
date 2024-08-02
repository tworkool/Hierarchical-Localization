import numpy as np
from argparse import ArgumentParser
import os
import sys
from pathlib import Path
import json
import math
import pycolmap
import random
from PIL import Image

# from ..colmap.read_write_model import read_model, qvec2rotmat  # NOQA
from third_party.colmap.scripts.python.read_write_model import read_model, qvec2rotmat


class ExtraPose:
    def __init__(
        self,
        image: pycolmap.Image,
        camera: pycolmap.Camera,
        is_localized_camera=True,
        img_name=None,
    ):
        self.image = image
        self.camera = camera

        if is_localized_camera:
            self.id = 1337
            self.image.name = "MAIN" if img_name is None else img_name
        else:
            self.id = random.randint(1338, 9999)
            self.image.name = f"{self.id}"

        self.camera.camera_id = self.id
        self.image.camera_id = self.id
        self.image.image_id = self.id


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def safe_numpy_dict(d: dict or list):
    json_out = json.dumps(d, cls=NumpyEncoder)
    return json.loads(json_out)


def get_aux_and_loc_camera(
    cameras: dict, images: dict
) -> tuple[pycolmap.Camera, pycolmap.Camera]:
    LOC_CAM = None
    AUX_CAM = None

    for img in images.values():
        print(img.name, img.camera_id)
        if "main" in str(img.name).lower():
            # found loc image, now look for camera
            for camera in cameras.values():
                if camera.camera_id == img.camera_id:
                    print(
                        f"Found camera intrinsics for localization camera with ID {camera.camera_id}: {camera.params}"
                    )
                    LOC_CAM = camera
                else:
                    AUX_CAM = camera

                if LOC_CAM and AUX_CAM:
                    break
            break

    if LOC_CAM is None:
        raise Exception("Could not find camera intrinsics for localization camera")

    return AUX_CAM, LOC_CAM


def homogeous_transform_matrix(rotation: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    # rotation = qvec2rotmat(qvec)
    translation = tvec.reshape(3, 1)
    w2c = np.concatenate([rotation, translation], 1)
    w2c = np.concatenate([w2c, np.array([0, 0, 0, 1])[None]], 0)
    # c2w = np.linalg.inv(w2c)
    return w2c


def calculate_mean_frame_error(img: pycolmap.Image, pts):
    if not img.registered:
        return 999999
    frame_error_sum = 0
    frame_pts = 0
    skipped_pts = 0
    for pt_id, pt in pts.items():
        # print(pt, dir(pt))
        if pt_id == -1:
            skipped_pts += 1
        if not img.has_point3D(pt_id):
            continue
        frame_error_sum += pt.error
        frame_pts += 1
    frame_error = None if frame_pts == 0 else frame_error_sum / frame_pts
    if skipped_pts > 0:
        print(f"skipped unregistered 3D points: {skipped_pts}")
    return frame_error


def _cv_to_gl(cv):
    # convert to GL convention used in iNGP
    gl = cv * np.array([1, -1, -1, 1])
    return gl


def get_frames(images: dict, points3D) -> list:
    ret = []

    for img_id, img in images.items():
        world_t_camera = img.cam_from_world.inverse()
        rot = world_t_camera.rotation.matrix()
        # qvec = world_t_camera.rotation.quat
        tvec = world_t_camera.translation
        transform_matrix = homogeous_transform_matrix(rot, tvec)
        transform_matrix = _cv_to_gl(transform_matrix)
        r_err = calculate_mean_frame_error(img, points3D)
        # print(img.summary(), r_err)

        print(f"calculated frame error {r_err:.2f}px for image {img.name}")

        frame = {
            "file_path": img.name,
            "transform_matrix": transform_matrix,
            "error": r_err,
        }
        ret.append(frame)

    return ret


def data_to_json(
    reconstruction: pycolmap.Reconstruction,
    image_path: Path,
    data_path: Path,
    extra_pose: ExtraPose = None,
):
    cameras = reconstruction.cameras
    images = reconstruction.images
    points3D = reconstruction.points3D

    # print(cameras)
    # print(images)
    # print(points3D)

    if extra_pose is not None:
        # print(extra_pose.image)
        # print(extra_pose.camera)
        print(dir(extra_pose.image))
        cameras[extra_pose.camera.camera_id] = extra_pose.camera
        images[extra_pose.image.image_id] = extra_pose.image

    # find representing auxilliary and localized camera from all cameras and images
    aux_cam, loc_cam = get_aux_and_loc_camera(cameras, images)
    loc_cam_intrinsics = loc_cam.params

    # f,cx,cy,k = focal length, principal point, distortion koefficient
    fl_x = loc_cam_intrinsics[0]
    fl_y = loc_cam_intrinsics[0]
    cx = loc_cam_intrinsics[1]
    cy = loc_cam_intrinsics[2]
    k = loc_cam_intrinsics[3]
    w = loc_cam.width
    h = loc_cam.height
    angle_x = math.atan(w / (fl_x * 2)) * 2
    angle_y = math.atan(h / (fl_y * 2)) * 2

    frames = get_frames(images, points3D)
    points = [{"rgb": p.color, "xyz": p.xyz} for _, p in list(points3D.items())]

    # add metadata to frames
    for f in frames:
        full_img_path = image_path / f["file_path"]
        if full_img_path.exists():
            pil_image = Image.open(full_img_path)
            width, height = pil_image.size
            f["metadata"] = {
                "full_path": str(full_img_path.resolve()),
                "w": int(width),
                "h": int(height),
                "focal_length": 26,
            }

    model = {
        "aux_cam_params": {
            "f": aux_cam.params[0],
        },
        "camera_angle_x": angle_x,
        "camera_angle_y": angle_y,
        "fl_x": fl_x,
        "fl_y": fl_y,
        "cx": cx,
        "cy": cy,
        "w": int(w),
        "h": int(h),
        "frames": frames,
        "points": points,
    }

    # export json
    export_path = data_path / "transforms.json"
    with open(export_path, "w+") as outputfile:
        json.dump(model, outputfile, indent=2, cls=NumpyEncoder)

    return safe_numpy_dict(model)
