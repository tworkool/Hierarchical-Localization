from pathlib import Path
import argparse
import logging

from hloc import (
    extract_features,
    match_features,
    match_dense,
    localize_sfm,
    match_features,
    pairs_from_all,
    triangulation,
    reconstruction,
)

from hloc.pipelines.Blender_Synthetic import analyse_dataset

"""
Pipeline for testing multiple HLOC pipelines on multiple datasets
call like this: python pipeline.py --dataset datasets/test.blend
"""
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
log = logging.getLogger("HlocBlenderSynthPipeline")

config = [
    {
        "extractor": "superpoint_max",
        "matcher": "superglue",
    },
    {
        "extractor": "disk",
        "matcher": "lightglue",
    },
    {
        "matcher": "loftr",
    },
    {
        "extractor": "superpoint_max",
        "matcher": "lightglue",
    },
    {
        "extractor": "s2dnet",  # TODO: add s2d
        "matcher": "lightglue",
    },
    {
        "extractor": "r2d2",  # TODO: add r2d2
        "matcher": "lightglue",
    },
    # TODO: add HP (Does not work...)
]


def ABORT(msg):
    log.error(msg)
    print("aborting")
    exit()


def get_extractor_config(key):
    return extract_features.confs.get(key, None)


def get_matcher_config(key):
    return match_dense.confs.get(key, None) or match_features.confs.get(key, None)


def validate():
    required_keys = ["matcher"]
    for c in config:
        if not all(key in c for key in required_keys):
            ABORT(f"Validation: config items require these items: {required_keys}")
        if not get_extractor_config(c["extractor"]):
            log.warn(f"Validation: extractor is missing or invalid")
        if not get_matcher_config(c["matcher"]):
            ABORT(f"Validation: {c['matcher']} is not a valid matcher")
    log.info("Validation Succeeded")
    return


# runs selected component on dataset
def run_component(component):
    return


def main(args):
    validate()

    if not (args.dataset.is_file() and args.dataset.name.split(".")[-1] == "blend"):
        raise Exception("Provided dataset is not a .blend file")

    ds_analysis = analyse_dataset.main()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to the dataset (.blend file)",
    )
    args = parser.parse_args()
    main(args)
