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
    }  # add more matching pipelines!
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
    required_keys = ["extractor", "matcher"]
    for c in config:
        if not all(key in c for key in required_keys):
            ABORT(
                f"VALIDATION FAILED: config items require these items: {required_keys}"
            )
        if not get_extractor_config(c["extractor"]):
            ABORT(f"VALIDATION FAILED: {c['extractor']} is not a valid extractor")
        if not get_matcher_config(c["matcher"]):
            ABORT(f"VALIDATION FAILED: {c['matcher']} is not a valid matcher")
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
