import json
import bpy
import sys
from random import uniform
import time
import numpy as np
import re
from pprint import pp
from mathutils import Matrix, Euler, Quaternion, Vector
import bmesh
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve()
transforms = None
DEFAULT_FOCAL_LENGTH = 27
DEFAULT_SENSOR_WIDTH = 35
MAX_REPROJECTION_ERROR_THESHOLD = 1.0  # warning if avg is above
MAX_CAMS = 3  # for transformation alignment


def eval_args():
    args = sys.argv[sys.argv.index("--") + 1 :]
    TRANSFORMS_FILE = Path(args[0])
    if not (TRANSFORMS_FILE.exists()):
        raise Exception(f"File not found: {TRANSFORMS_FILE}")

    IMAGES_PATH = Path(args[1])
    if not (IMAGES_PATH.exists()):
        raise Exception(f"File not found: {IMAGES_PATH}")
    return {"transforms": TRANSFORMS_FILE, "images": IMAGES_PATH}


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


# m2 is subset of m1 names! m1=GT, m2=SYNTH
def map_cameras(m1, m2, data):
    m2_names = [c.name.split(".")[0] for c in m2]
    m1_names = [c.name.split(".")[0] for c in m1]
    diff = set(m1_names) - set(m2_names)
    if len(diff) > 0:
        raise Exception(f"These cameras are missing in the estimated camera poses! {diff}")

    map = []
    for c1 in m1:
        for c2 in m2:
            if c1.name in c2.name or ("MAIN" in c1.name and "MAIN" in c2.name):
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
    #print("Scale:", scale)

    # Compute rotation using SVD
    H = np.dot(centered_est.T, centered_gt)
    U, _, Vt = np.linalg.svd(H)
    rotation = np.dot(Vt.T, U.T)
    #print("Initial Rotation Matrix:\n", rotation)

    # Ensure a proper rotation matrix (det(R) = 1)
    if np.linalg.det(rotation) < 0:
        Vt[-1, :] *= -1
        rotation = np.dot(Vt.T, U.T)
    #print("Adjusted Rotation Matrix:\n", rotation)

    # Compute translation
    translation = ground_truth_points[0] - scale * np.dot(rotation, estimated_points[0])
    #print("Translation:", translation)

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


def align_estimated_to_ground_truth(est_cams, top_cams, apply_to=None):
    # Extract the top cameras' positions
    gt_positions = np.array([cam["c1"].location for cam in top_cams])
    est_positions = np.array([cam["c2"].location for cam in top_cams])

    # Compute the Helmert transformation
    scale, rotation, translation = helmert_transform(gt_positions, est_positions)

    try:
        local_rot = Matrix(np.linalg.inv(rotation)).transposed().to_euler()
    except np.linalg.LinAlgError as e:
        raise Exception(
            f"Failed to apply local rotation due to not inversible rotation matrix: {e}"
        )

    # Apply the transformation to all estimated camera poses
    apply_to = est_cams if not apply_to else apply_to.all_objects
    for cam in apply_to:
        est_position = np.array(cam.location)
        transformed_position = apply_helmert_transform(
            est_position, scale, rotation, translation
        )

        # apply transform for new position
        cam.location = transformed_position

        # apply local rotation by calculating the inverse of the rotation matrix
        cam.rotation_euler.rotate(local_rot)

    print("Alignment complete")


def analyse_poses(cam_map):
    total_translation_error = 0.0
    total_rotation_error = 0.0
    num_cameras = len(cam_map)
    analysed_poses = []

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

        print(f"Camera: {c1.name} -> {c2.name} - [t_err {translation_error:.3f} m] [r_err {rotation_error_deg:.3f} deg] [re_err {entry['error']:.3f}]")

        analysed_poses.append(
            {
                "name": c1.name,
                "t_err_m": translation_error,
                "r_err_deg": rotation_error_deg,
                "reprojection_err": entry["error"],
            }
        )

    avg_translation_error = total_translation_error / num_cameras
    avg_rotation_error = total_rotation_error / num_cameras

    print(f"\nAverage Errors: - [t_err {avg_translation_error:.3f} m] [r_err {avg_rotation_error:.3f} deg]")

    return {
        "avg_translation_error": avg_translation_error,
        "avg_rotation_error": avg_rotation_error,
        "analysed_poses": analysed_poses,
    }


class SceneUtils:
    BACKUP_SCENE_NAME = "BACKUP_SCENE"

    def __init__(self) -> None:
        pass

    def lock_scene_objects():
        # Iterate through all objects in the current scene
        for obj in bpy.context.scene.objects:
            # Check if the object is a camera
            # if obj.type == 'CAMERA':
            # Lock transformations for location, rotation, and scale
            obj.lock_location = (True, True, True)
            obj.lock_rotation = (True, True, True)
            obj.lock_scale = (True, True, True)
        print("SceneUtils: locked all scene objects")

    def backup_linked_scene():
        # Get the name of the current scene
        original_scene = bpy.context.scene
        original_scene_name = original_scene.name

        # Create a new scene for backup
        bpy.ops.scene.new(type="EMPTY")
        backup_scene = bpy.context.scene
        backup_scene.name = SceneUtils.BACKUP_SCENE_NAME

        # Link objects from the original scene to the backup scene
        for obj in bpy.data.scenes[original_scene_name].objects:
            backup_scene.collection.objects.link(obj)

        bpy.context.window.scene = bpy.data.scenes[SceneUtils.BACKUP_SCENE_NAME]
        print("SceneUtils: created scene backup")

    def restore_scene():
        backup_scene_name = SceneUtils.BACKUP_SCENE_NAME

        # Check if the backup scene exists
        if backup_scene_name in bpy.data.scenes:
            bpy.data.scenes.remove(bpy.data.scenes[backup_scene_name])
            print("SceneUtils: Original scene restored successfully!")
        else:
            print("SceneUtils: Backup scene not found!")


def colmap_to_blender4(c2w):
    # Conversion from COLMAP/OpenCV to Blender coordinate system
    colmap_to_blender_transform = np.array(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    )

    # -90 degree rotation around the x-axis
    rotation_neg_90_x = np.array(
        [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
    )

    # Apply both transformations
    return rotation_neg_90_x @ colmap_to_blender_transform @ c2w


def adjust_scene(data):
    w = data["w"]
    h = data["h"]
    bpy.context.scene.render.resolution_x = w
    bpy.context.scene.render.resolution_y = h
    return


def draw_frames(data, image_path: Path):
    # Extract the data
    frames = data["frames"]

    # FRAMES
    for frame in frames:
        # transform COLMAP coords to blender
        blender_transform = colmap_to_blender4(frame["transform_matrix"])
        transform_matrix = Matrix(blender_transform.tolist())

        # Extract the position from the last column of the matrix
        position = transform_matrix.translation

        # Create a new camera
        camera = bpy.data.cameras.new("Camera")
        # Create a new object and link the camera to it
        camera_obj = bpy.data.objects.new("Camera", camera)
        bpy.context.collection.objects.link(camera_obj)
        # Set the location of the camera object to the translation part of the matrix
        camera_obj.location = position
        # Set the rotation of the camera object to the rotation part of the matrix
        camera_obj.rotation_euler = transform_matrix.to_euler("XYZ")
        # Scale the camera object
        # camera_obj.scale = (0.5, 0.5, 0.5)
        camera_obj.name = frame["file_path"]
        camera_obj.show_name = True

        # set intrinsics
        frame_metadata = frame["metadata"] if "metadata" in frame else None
        if frame_metadata:
            # Set the background image for the camera
            # Set the path to your background image
            background_image_path = image_path / frame["file_path"]
            #print(background_image_path)
            camera_obj.data.show_background_images = True
            background_image = camera_obj.data.background_images.new()
            background_image.image = bpy.data.images.load(
                str(background_image_path.resolve())
            )
            # background_image.image.use_alpha = False  # Set to True if your image has an alpha channel

            # other camera intrinsics
            camera_obj.data.lens = (
                "focal_length" in frame_metadata and frame_metadata["focal_length"]
                if frame_metadata["focal_length"]
                else float(DEFAULT_FOCAL_LENGTH)
            )
            camera_obj.data.sensor_width = (
                frame_metadata["lens_sensor_size"]
                if "lens_sensor_size" in frame_metadata
                and frame_metadata["lens_sensor_size"]
                else float(DEFAULT_SENSOR_WIDTH)
            )


def main(args):
    TRANSFORMS_FILE = args["transforms"]
    IMAGES_PATH = args["images"]
    with open(
        TRANSFORMS_FILE  # "D:/dev/python/Hierarchical-Localization/outputs/sfm/framlingham_loftr/transforms.json"
    ) as f:
        transforms = json.load(f)

    SceneUtils.backup_linked_scene()
    SceneUtils.lock_scene_objects()

    analysis = {}
    analysis_path = TRANSFORMS_FILE.parent.resolve()

    try:
        # new collection and set active to import frames here
        est_coll = bpy.data.collections.new("IMPORT")
        bpy.context.scene.collection.children.link(est_coll)
        view_layer = bpy.context.view_layer
        view_layer.active_layer_collection = view_layer.layer_collection.children[
            est_coll.name
        ]
        # get gt cameras b4 import
        gt_cams = get_cameras()

        # start import
        draw_frames(transforms, IMAGES_PATH)

        # transformation
        est_cams = get_cameras(parent=est_coll)

        cam_map = map_cameras(gt_cams, est_cams, transforms)
        top_cams = get_top_cams(cam_map)

        # align estimated cameras to ground truth
        align_estimated_to_ground_truth(est_cams, top_cams)

        # analysis
        analysis = analyse_poses(cam_map)
        analysis["result"] = "ok"
        analysis["message"] = "ok"
    except Exception as e:
        #raise Exception(e)
        analysis["result"] = "error"
        analysis["message"] = str(e)
        print(f"ERROR: {e}")
    finally:
        # restore scene
        with open(analysis_path / "stats.json", "w+", encoding="utf8") as f:
            json.dump(analysis, f)
        SceneUtils.restore_scene()
    return 1


if __name__ == "__main__":
    """ parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transforms",
        type=Path,
        required=True,
        help="path to the transforms json file",
    )

    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="path to the images",
    )
    args = parser.parse_args()
    print(args) """
    args = eval_args()
    main(args)
