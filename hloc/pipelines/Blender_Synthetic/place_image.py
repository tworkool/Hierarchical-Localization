import bpy
import numpy as np
import re
from pprint import pp

# from mathutils import Matrix, Euler, Quaternion, Vector
import mathutils
from pathlib import Path

# from PIL import Image
import math

ROOT = Path(__file__).resolve()
LOC_CAM_NAME = "MAIN"
EST_COLL_NAME = "IMPORT"
IMG_PATH = Path()


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


def get_loc_camera():
    est_coll = bpy.data.collections.get(EST_COLL_NAME)
    cams = get_cameras(LOC_CAM_NAME, est_coll)
    return cams[0] if len(cams) > 0 else None


"""def prepare_cam_intrinsics(image: Path):
    _image = Image.open(image.resolve())
    w, h = _image.size
    bpy.context.scene.render.resolution_x = w
    bpy.context.scene.render.resolution_y = h
    # TODO: load calibration matrix
    # workaround: define by hand
    print(w, h)
    f = 40 #700 #mm
    px = w/2
    py = h/2
    skew = 0
    K = np.array(
        [
            [f, skew, 300],
            [0, f, 190.5],
            [0, 0, 1]
        ]
    )
    return K
"""


def get_camera_clip_planes(camera, clip_distance):
    """
    Calculate the four corner points of the camera's clipping plane.

    :param camera: bpy.types.Object, the camera object
    :return: dict, dictionary containing the four corner points
    """
    scene = bpy.context.scene
    cam_data = camera.data

    # Get camera properties
    focal_length = cam_data.lens
    sensor_width = cam_data.sensor_width
    sensor_height = cam_data.sensor_height
    sensor_fit = cam_data.sensor_fit
    aspect_ratio = scene.render.resolution_x / scene.render.resolution_y

    # Calculate the horizontal and vertical FOV
    if sensor_fit == "VERTICAL":
        vertical_fov = 2 * math.atan((sensor_height / (2 * focal_length)))
        horizontal_fov = 2 * math.atan(
            (sensor_width * aspect_ratio / (2 * focal_length))
        )
    else:  # HORIZONTAL or AUTO
        horizontal_fov = 2 * math.atan((sensor_width / (2 * focal_length)))
        vertical_fov = 2 * math.atan(
            (sensor_height / (2 * focal_length * aspect_ratio))
        )

    # Get the near clipping distance
    # clip_distance = cam_data.clip_start

    # Calculate the width and height of the near clipping plane
    near_height = 2 * math.tan(vertical_fov / 2) * clip_distance
    near_width = 2 * math.tan(horizontal_fov / 2) * clip_distance

    # Define the corner points in camera space
    top_left = mathutils.Vector((-near_width / 2, -near_height / 2, -clip_distance))
    top_right = mathutils.Vector((near_width / 2, -near_height / 2, -clip_distance))
    bottom_left = mathutils.Vector((-near_width / 2, near_height / 2, -clip_distance))
    bottom_right = mathutils.Vector((near_width / 2, near_height / 2, -clip_distance))

    # Convert the points to world space
    cam_matrix_world = camera.matrix_world
    top_left_world = cam_matrix_world @ top_left
    top_right_world = cam_matrix_world @ top_right
    bottom_left_world = cam_matrix_world @ bottom_left
    bottom_right_world = cam_matrix_world @ bottom_right

    return {
        "top_left": top_left_world,
        "top_right": top_right_world,
        "bottom_left": bottom_left_world,
        "bottom_right": bottom_right_world,
    }


def create_plane_at_clip_plane(camera, clip_distance):
    """
    Create or update a plane in the world at the camera's clipping plane.

    :param camera: bpy.types.Object, the camera object
    """
    clip_plane_corners = get_camera_clip_planes(camera, clip_distance)

    # Check if a mesh named "ClipPlane" exists
    mesh = bpy.data.meshes.get("ClipPlane")
    if mesh is None:
        # Create a new mesh and object if it doesn't exist
        mesh = bpy.data.meshes.new(name="ClipPlane")
        plane = bpy.data.objects.new("ClipPlane", mesh)
        bpy.context.collection.objects.link(plane)
    else:
        # Get the existing object
        plane = bpy.data.objects.get("ClipPlane")
        if plane is None:
            # Create the object if it somehow doesn't exist
            plane = bpy.data.objects.new("ClipPlane", mesh)
            bpy.context.collection.objects.link(plane)

    # Get the corner points
    verts = [
        clip_plane_corners["top_left"],
        clip_plane_corners["top_right"],
        clip_plane_corners["bottom_right"],
        clip_plane_corners["bottom_left"],
    ]

    # Create faces
    faces = [(0, 1, 2, 3)]

    # Clear existing mesh data
    mesh.clear_geometry()

    # Assign vertices and faces to the mesh
    mesh.from_pydata(verts, [], faces)
    mesh.update()


# camera = bpy.context.object
camera = get_loc_camera()
create_plane_at_clip_plane(camera, 6)
# clip_plane_corners = get_camera_clip_planes(camera)
# print(clip_plane_corners)
