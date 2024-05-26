import json
import bpy
import os
from random import uniform
import time
import numpy as np
import re
from pprint import pp
from mathutils import Matrix, Euler

root_dir = os.getcwd()
transforms = None

with open(
    "D:/dev/python/Hierarchical-Localization/outputs/sfm/framlingham_loftr/transforms.json"
) as f:
    transforms = json.load(f)


def get_cameras(match_name=None, parent=None):
    cams = []
    if not parent:
        parent = bpy.context.scene
    for obj in parent.objects:
        if obj.type != "CAMERA":
            continue
        if match_name and not re.search(match_name, obj.name):
            continue
        cams.append(obj)
    return cams
    return [
        obj
        for obj in bpy.context.scene.objects
        if obj.type == "CAMERA" and obj.name == name.replace("**", obj.name)
    ]


# m2 is subset of m1 names! m1=GT, m2=SYNTH
def map_cameras(m1, m2, data):
    assert len(m1) == len(m2)
    map = []
    for c1 in m1:
        for c2 in m2:
            if c1.name in c2.name:
                print(f"INFO: matched {c1.name} and {c2.name}")
                # append error
                for frame in data["frames"]:
                    f_name = frame["file_path"]
                    if f_name == c2.name:
                        print(f"INFO: added error metric for {c2.name} from frame")
                        map.append({"c1": c1, "c2": c2, "error": frame["error"]})
                        break
                break
    assert len(m1) == len(map)
    return map


MAX_REPROJECTION_ERROR_THESHOLD = 1.0
MAX_CAMS = 3


def get_top_cams(cam_map):
    sorted_cams = sorted(cam_map, key=lambda x: x["error"])
    if len(cam_map) < MAX_CAMS:
        print(
            "cannot select top cameras for optimal alignment. There need to be at least N cameras available!"
        )
    else:
        sorted_cams = sorted_cams[:MAX_CAMS]
    mean_error = np.asarray([x["error"] for x in sorted_cams]).mean()
    if mean_error > MAX_REPROJECTION_ERROR_THESHOLD:
        print(
            f"WARNING: camera exceeds max error theshold and could lead to inaccurately aligned cameras: {mean_error}"
        )

    c1c2_d = (sorted_cams[0]["c2"].location - sorted_cams[1]["c2"].location).length
    c1c3_d = (sorted_cams[0]["c2"].location - sorted_cams[2]["c2"].location).length
    if c1c3_d > c1c2_d:
        # switch sorted cams location so that the second cam is further away
        # further distance = finer scaling
        sorted_cams[1], sorted_cams[2] = sorted_cams[2], sorted_cams[1]

    # 1: set position of first cam to GT first cam
    # 2: scale to fit 2nd cam with GT 2nd cam
    # 3: rotate to fit 2nd cam with GT 2nd cam
    # 4: rotate to fit 3rd cam with GT 3rd cam
    return sorted_cams


def helmert_transform(ground_truth_points, estimated_points):
    # Compute scale
    centered_gt = ground_truth_points - ground_truth_points[0]
    centered_est = estimated_points - estimated_points[0]
    scale = np.sqrt(np.sum(centered_gt**2) / np.sum(centered_est**2))
    print("Scale:", scale)

    # Compute rotation using SVD
    H = np.dot(centered_est.T, centered_gt)
    U, _, Vt = np.linalg.svd(H)
    rotation = np.dot(Vt.T, U.T)
    print("Initial Rotation Matrix:\n", rotation)

    # Ensure a proper rotation matrix (det(R) = 1)
    if np.linalg.det(rotation) < 0:
        Vt[-1, :] *= -1
        rotation = np.dot(Vt.T, U.T)
    print("Adjusted Rotation Matrix:\n", rotation)

    # Compute translation
    translation = ground_truth_points[0] - scale * np.dot(rotation, estimated_points[0])
    print("Translation:", translation)

    return scale, rotation, translation


def apply_helmert_transform(points, scale, rotation, translation):
    transformed_points = scale * np.dot(points, rotation.T) + translation
    return transformed_points


def apply_helmert_transform_blender(cam, scale, rotation, translation):
    # Apply the scale, rotation, and translation to the camera location using Blender's API
    est_position = np.array(cam.location)
    transformed_position = scale * np.dot(rotation, est_position) + translation
    cam.location = transformed_position

    # Convert the rotation matrix to Euler angles and set the rotation
    rotation_matrix = Matrix(rotation).transposed()  # Transpose the rotation matrix
    rotation_euler = rotation_matrix.to_euler()
    cam.rotation_euler = rotation_euler


def align_estimated_to_ground_truth(est_cams, gt_cams, top_cams, apply_to=None):
    # Extract the top cameras' positions
    gt_positions = np.array([cam["c1"].location for cam in top_cams])
    est_positions = np.array([cam["c2"].location for cam in top_cams])

    # Compute the Helmert transformation
    scale, rotation, translation = helmert_transform(gt_positions, est_positions)

    # Apply the transformation to all estimated camera poses
    apply_to = est_cams if not apply_to else apply_to.all_objects
    for cam in apply_to:
        est_position = np.array(cam.location)
        transformed_position = apply_helmert_transform(
            est_position, scale, rotation, translation
        )
        cam.location = transformed_position
        # apply_helmert_transform_blender(cam, scale, rotation, translation)

    print("Alignment complete")


def analyse_poses(cam_map):
    total_translation_error = 0.0
    total_rotation_error = 0.0
    num_cameras = len(cam_map)

    for entry in cam_map:
        c1 = entry["c1"]
        c2 = entry["c2"]

        # Calculate translation error (Euclidean distance between positions)
        translation_error = np.linalg.norm(
            np.array(c1.location) - np.array(c2.location)
        )
        total_translation_error += translation_error

        # Calculate rotation error (angle difference in degrees)
        euler_c1 = np.array(c1.rotation_euler)
        euler_c2 = np.array(c2.rotation_euler)

        # Compute the difference in Euler angles
        euler_diff = euler_c1 - euler_c2

        # Convert Euler angles difference to degrees
        euler_diff_deg = np.degrees(euler_diff)

        # Calculate the rotation error as the norm of the Euler angles difference
        rotation_error_deg = np.linalg.norm(euler_diff_deg)
        total_rotation_error += rotation_error_deg

        print(f"Camera: {c1.name} -> {c2.name}")
        print(f"  Translation Error: {translation_error:.3f} meters")
        print(f"  Rotation Error: {rotation_error_deg:.3f} degrees")
        print(f"  Reprojection Error: {entry['error']}")

    avg_translation_error = total_translation_error / num_cameras
    avg_rotation_error = total_rotation_error / num_cameras

    print("\nAverage Errors:")
    print(f"  Average Translation Error: {avg_translation_error:.3f} meters")
    print(f"  Average Rotation Error: {avg_rotation_error:.3f} degrees")


est_coll = bpy.data.collections.get("Collection 3")
gt_coll = bpy.data.collections.get("Scene7")
est_cams = get_cameras(parent=est_coll)
gt_cams = get_cameras(parent=gt_coll)

cam_map = map_cameras(gt_cams, est_cams, transforms)
top_cams = get_top_cams(cam_map)

# Align estimated cameras to ground truth
align_estimated_to_ground_truth(est_cams, gt_cams, top_cams)

analyse_poses(cam_map)
