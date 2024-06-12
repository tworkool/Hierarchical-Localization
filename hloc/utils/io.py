from pathlib import Path
from typing import Tuple

import cv2
import h5py
import numpy as np

from .parsers import names_to_pair, names_to_pair_old, parse_retrieval
from .colmap import read_cameras_binary


def read_image(path, grayscale=False):
    if grayscale:
        mode = cv2.IMREAD_GRAYSCALE
    else:
        mode = cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise ValueError(f"Cannot read image {path}.")
    if not grayscale and len(image.shape) == 3:
        image = image[:, :, ::-1]  # BGR to RGB
    return image


def list_h5_names(path):
    names = []
    with h5py.File(str(path), "r", libver="latest") as fd:

        def visit_fn(_, obj):
            if isinstance(obj, h5py.Dataset):
                names.append(obj.parent.name.strip("/"))

        fd.visititems(visit_fn)
    return list(set(names))


def get_keypoints(
    path: Path, name: str, return_uncertainty: bool = False
) -> np.ndarray:
    with h5py.File(str(path), "r", libver="latest") as hfile:
        dset = hfile[name]["keypoints"]
        p = dset.__array__()
        uncertainty = dset.attrs.get("uncertainty")
    if return_uncertainty:
        return p, uncertainty
    return p


def find_pair(hfile: h5py.File, name0: str, name1: str):
    pair = names_to_pair(name0, name1)
    if pair in hfile:
        return pair, False
    pair = names_to_pair(name1, name0)
    if pair in hfile:
        return pair, True
    # older, less efficient format
    pair = names_to_pair_old(name0, name1)
    if pair in hfile:
        return pair, False
    pair = names_to_pair_old(name1, name0)
    if pair in hfile:
        return pair, True
    raise ValueError(
        f"Could not find pair {(name0, name1)}... "
        "Maybe you matched with a different list of pairs? "
    )


def get_matches(path: Path, name0: str, name1: str) -> Tuple[np.ndarray]:
    with h5py.File(str(path), "r", libver="latest") as hfile:
        pair, reverse = find_pair(hfile, name0, name1)
        matches = hfile[pair]["matches0"].__array__()
        scores = hfile[pair]["matching_scores0"].__array__()
    idx = np.where(matches != -1)[0]
    matches = np.stack([idx, matches[idx]], -1)
    if reverse:
        matches = np.flip(matches, -1)
    scores = scores[idx]
    return matches, scores

def generate_query_list(colmap_cameras_path: Path, output_path: Path):
    cameras = read_cameras_binary(colmap_cameras_path)
    with open(output_path, "w+") as f:
        for camera_id, camera in cameras.items():
            # Extract the necessary parameters
            model = camera.model
            width = camera.width
            height = camera.height
            params = camera.params

            # For the SIMPLE_RADIAL model, params = [f, cx, cy, k]
            if model == "SIMPLE_RADIAL":
                fx = params[0]
                cx = params[1]
                cy = params[2]
                k = params[3]
            else:
                raise ValueError(f"Unsupported camera model: {model}")

            # Write to the query list file
            f.write(f"{camera_id} {model} {width} {height} {fx} {cx} {cy} {k}\n")

def generate_query_list(colmap_cameras, image_names, output_path: Path):
    assert len(colmap_cameras) == len(image_names)
    cameras = colmap_cameras
    with open(output_path, "w+") as f:
        for i, camera in enumerate(cameras):
            # Extract the necessary parameters
            model = str(camera.model).replace("CameraModelId.", "")
            width = camera.width
            height = camera.height
            params = camera.params

            # For the SIMPLE_RADIAL model, params = [f, cx, cy, k]
            if model == "SIMPLE_RADIAL":
                fx = params[0]
                cx = params[1]
                cy = params[2]
                k = params[3]
            else:
                raise ValueError(f"Unsupported camera model: {model}")

            # Write to the query list file
            f.write(f"{image_names[i]} {model} {width} {height} {fx} {cx} {cy} {k}\n")


def generate_localization_pairs(reloc, num, ref_pairs, out_path):
    """Create the matching pairs for the localization.
    We simply lookup the corresponding reference frame
    and extract its `num` closest frames from the existing pair list.
    """
    relocs = [reloc]
    query_to_ref_ts = {}
    for reloc in relocs:
        with open(reloc, "r") as f:
            for line in f.readlines():
                line = line.rstrip("\n")
                if line[0] == "#" or line == "":
                    continue
                ref_ts, q_ts = line.split()[:2]
                query_to_ref_ts[q_ts] = ref_ts

    ts_to_name = "cam0/{}.png".format
    ref_pairs = parse_retrieval(ref_pairs)
    loc_pairs = []
    for q_ts, ref_ts in query_to_ref_ts.items():
        ref_name = ts_to_name(ref_ts)
        selected = [ref_name] + ref_pairs[ref_name][: num - 1]
        loc_pairs.extend([" ".join((ts_to_name(q_ts), s)) for s in selected])
    with open(out_path, "w") as f:
        f.write("\n".join(loc_pairs))