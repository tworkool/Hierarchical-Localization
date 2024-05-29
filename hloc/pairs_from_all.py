import argparse
from pathlib import Path

from . import logger
from .utils.parsers import parse_image_list_dir


def main(
    output: Path,
    image_list: Path,
):
    if image_list is not None:
        names_q = parse_image_list_dir(image_list)
    else:
        raise ValueError(f"Unknown type for image list: {image_list}")

    sorted_image_ids = sorted(names_q)
    pairs = []
    for i in range(len(sorted_image_ids)):
        for j in range(i + 1, len(sorted_image_ids)):
            pairs.append((sorted_image_ids[i], sorted_image_ids[j]))

    logger.info(f"Found {len(pairs)} pairs.")
    with open(output, "w+") as f:
        f.write("\n".join(" ".join([i, j]) for i, j in pairs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image_list", required=True, type=Path)
    args = parser.parse_args()
    main(**args.__dict__)
