import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pycolmap

logger = logging.getLogger(__name__)


def parse_image_list_dir(
    path: Path,
    image_extensions={".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"},
    recursive=True
):
    if recursive:
        # Get all files in the directory and its subdirectories
        all_files = path.rglob("*")
    else:
        all_files = path.glob("*")
    # Filter image files by their extensions, case-insensitively
    image_files = [
        file for file in all_files if file.suffix.lower() in image_extensions
    ]
    print(image_files)
    # Convert to absolute paths
    #images = [str(file.resolve()) for file in image_files]
    images = [str(file.name) for file in image_files]

    assert len(images) > 0
    logger.info(f"Imported {len(images)} images from {path.name}")
    return images


def parse_image_list(path, with_intrinsics=False):
    images = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip("\n")
            if len(line) == 0 or line[0] == "#":
                continue
            name, *data = line.split()
            if with_intrinsics:
                model, width, height, *params = data
                params = np.array(params, float)
                cam = pycolmap.Camera(
                    model=model, width=int(width), height=int(height), params=params
                )
                images.append((name, cam))
            else:
                images.append(name)

    assert len(images) > 0
    logger.info(f"Imported {len(images)} images from {path.name}")
    return images


def parse_image_lists(paths, with_intrinsics=False):
    images = []
    files = list(Path(paths.parent).glob(paths.name))
    assert len(files) > 0
    for lfile in files:
        images += parse_image_list(lfile, with_intrinsics=with_intrinsics)
    return images


def parse_retrieval(path):
    retrieval = defaultdict(list)
    with open(path, "r") as f:
        for p in f.read().rstrip("\n").split("\n"):
            if len(p) == 0:
                continue
            q, r = p.split()
            retrieval[q].append(r)
    return dict(retrieval)


def names_to_pair(name0, name1, separator="/"):
    return separator.join((name0.replace("/", "-"), name1.replace("/", "-")))


def names_to_pair_old(name0, name1):
    return names_to_pair(name0, name1, separator="_")
