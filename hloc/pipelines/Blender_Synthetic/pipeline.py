from pathlib import Path
import argparse
import logging
import shutil
import subprocess

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

from third_party.Neuralangelo.convert_data_to_json import data_to_json

# from hloc.pipelines.Blender_Synthetic import analyse_dataset

"""
Pipeline for testing multiple HLOC pipelines on multiple datasets
call like this: python pipeline.py --dataset datasets/Bartholomew+evening_field_8k/
"""
logging.basicConfig(format="  >> %(levelname)s: %(message)s", level=logging.INFO)
log = logging.getLogger("HlocBlenderSynthPipeline")
ROOT = Path(__file__).parent.resolve()

config = [
    {
        "extractor": "superpoint_max",
        "matcher": "superglue",
        "name": "superpoint+superglue",
    },
    {"extractor": "disk", "matcher": "disk+lightglue", "name": "disk+lightglue"},
    {"matcher": "loftr", "name": "loftr"},
    {
        "extractor": "superpoint_max",
        "matcher": "superpoint+lightglue",
        "name": "superpoint+lightglue",
    },
    {"extractor": "s2dnet", "matcher": "loftr", "name": "..."},  # TODO: add s2d
    {"extractor": "r2d2", "matcher": "loftr", "name": "..."},  # TODO: add r2d2
    # TODO: add HP (Does not work...)
]


def ABORT(msg):
    log.error(msg)
    print("aborting...")
    exit()


def get_extractor_config(key):
    return extract_features.confs.get(key, None)


def get_matcher_config(key):
    is_dense = match_dense.confs.get(key, None)
    config = match_dense.confs.get(key, None) or match_features.confs.get(key, None)
    return (config, is_dense)


def validate():
    required_keys = ["matcher", "name"]
    for c in config:
        if not all(key in c for key in required_keys):
            ABORT(f"Validation: config items require these items: {required_keys}")
        if "extractor" not in c or not get_extractor_config(c["extractor"]):
            log.warning(f"Validation: extractor is missing or invalid")
        matcher_conf, _ = get_matcher_config(c["matcher"])
        if not matcher_conf:
            ABORT(f"Validation: {c['matcher']} is not a valid matcher")
    log.info("Validation Succeeded")
    return


def component_id(component: dict, dataset_name: str):
    return f'{dataset_name}-{component["name"]}'


# runs selected component on dataset
def run_component(component, dataset) -> Path:
    id = component_id(component, dataset)
    OUT = Path(ROOT / "out" / dataset / id)
    if OUT.exists():
        log.info(f"  SKIPPING Component - '{id}' already ran for dataset '{dataset}'")
        return None

    extractor_conf = get_extractor_config(component["extractor"])
    matcher_conf, is_dense = get_matcher_config(component["matcher"])

    IN = Path(ROOT / "datasets" / dataset)
    IMAGES = Path(IN / "images")
    sfm_pairs = OUT / f"{id}-pairs.txt"
    sfm_dir = OUT / id
    sfm_images = sfm_dir / "images"

    # get image pairs
    pairs_from_all.main(sfm_pairs, IMAGES)

    # extract features
    feature_path = extract_features.main(extractor_conf, IMAGES, OUT)

    # match features
    if is_dense:
        feature_path, match_path = match_dense.main(
            conf=matcher_conf,
            pairs=sfm_pairs,
            image_dir=IMAGES,
            export_dir=OUT,
            features_ref=feature_path,
        )
    else:
        match_path = match_features.main(
            conf=matcher_conf,
            pairs=sfm_pairs,
            features=extractor_conf["output"],
            export_dir=OUT,
        )

    # start reconstruction
    model = reconstruction.main(
        sfm_dir, sfm_images, sfm_pairs, feature_path, match_path
    )

    # TODO: localization pipeline?

    # export transforms and copy images
    args = {
        "data_dir": str(sfm_dir.resolve()),
        "scene_type": "outdoor",
        "image_dir": str(IMAGES.resolve()),
        "name": "transforms.json",
    }
    data_to_json(args)

    if not sfm_images.exists():
        # copy files from input images to self contained project path
        # os.mkdir(new_input_path)
        log.info("copied input images to self contained SfM reconstruction folder")
        shutil.copytree(IMAGES, sfm_images)

    return sfm_dir


def run_analysis(blender_file: Path, transforms_json: Path):
    subprocess.run(
        [
            "blender",
            blender_file.resolve(),
            "-b",
            "--python",
            (ROOT / "analyse_dataset.py").resolve(),
            "--",  # script args
            transforms_json.resolve(),
        ],
        check=True,
    )
    # todo: read analysis summary for reconstruction


def main(args):
    validate()

    if not args.dataset.exists():
        ABORT(
            "Please provide the path to the dataset containing 'images' and 'test.blend'"
        )

    image_path = args.dataset / "images"
    if not image_path.exists():
        ABORT(
            "Please provide an 'images' folder containing the images for reconstruction in the dataset"
        )

    blender_file = args.dataset / "test.blend"
    if not blender_file.exists():
        ABORT("Please provide a blender file 'test.blend' for running stats")

    """ if not (args.dataset.is_file() and args.dataset.name.split(".")[-1] == "blend"):
        raise Exception("Provided dataset is not a .blend file") """

    dataset_name = "Test"

    log.info("Step 1: reconstructions")
    for c in config:
        # implement timer!
        run_component(c, dataset_name)

    # ds_analysis = analyse_dataset.main()
    # run_analysis()
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
