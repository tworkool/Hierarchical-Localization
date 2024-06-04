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
call like this: 
* python pipeline.py --dataset Bartholomew+evening_field_8k --validator datasets/Bartholomew+evening_field_8k/test.blend
* python pipeline.py --dataset Chateu-img1+evening_field_8k --validator datasets/Chateu-img1+evening_field_8k/chateu1.blend
"""
logging.basicConfig(format="  >> %(levelname)s: %(message)s", level=logging.INFO)
log = logging.getLogger("HlocBlenderSynthPipeline")
ROOT = Path(__file__).parent.resolve()
IMAGES_FOLDER_NAME = "images"

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
    #{"extractor": "s2dnet", "matcher": "loftr", "name": "..."},  # TODO: add s2d
    #{"extractor": "r2d2", "matcher": "loftr", "name": "r2d2+loftr"},  # TODO: add r2d2
    # TODO: add HP (Does not work...)
]

config1 = [
    {
        "extractor": "superpoint_max",
        "matcher": "superglue",
        "name": "superpoint+superglue",
    },
]

_datasets = [
    (
        "Bartholomew+evening_field_8k",
        "datasets/Bartholomew+evening_field_8k/test.blend",
    ),
    (
        "Chateu-img1+evening_field_8k",
        "datasets/Chateu-img1+evening_field_8k/chateu1.blend",
    ),
    (
        "Chateu-img2+evening_field_8k",
        "datasets/Chateu-img2+evening_field_8k/chateu2.blend",
    ),
    (
        "Chateu-img3+evening_field_8k",
        "datasets/Chateu-img3+evening_field_8k/chateu3.blend",
    ),
    (
        "Framlingham+evening_field_8k",
        "datasets/Framlingham+evening_field_8k/framlingham.blend",
    ),
    (
        "Pelegrina+evening_field_8k",
        "datasets/Pelegrina+evening_field_8k/pelegrina.blend",
    )
]


def ABORT(msg):
    log.error(msg)
    print("aborting...")
    exit()


def get_extractor_config(key):
    if not key:
        return None
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

    OUT.mkdir(parents=True, exist_ok=True)

    extractor_conf = get_extractor_config(component.get("extractor", None))
    matcher_conf, is_dense = get_matcher_config(component["matcher"])

    IN = Path(ROOT / "datasets" / dataset)
    IMAGES = Path(IN / IMAGES_FOLDER_NAME)
    sfm_pairs = OUT / f"{id}-pairs.txt"
    sfm_dir = OUT / id

    # get image pairs
    pairs_from_all.main(sfm_pairs, IMAGES)

    feature_path = None
    # extract features
    if extractor_conf:
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
    model = reconstruction.main(sfm_dir, IMAGES, sfm_pairs, feature_path, match_path)

    # TODO: localization pipeline?

    # export transforms and copy images
    args = {
        "data_dir": str(sfm_dir.resolve()),
        "scene_type": "outdoor",
        "image_dir": str(IMAGES.resolve()),
        "name": "transforms.json",
    }
    data_to_json(args)

    sfm_images = sfm_dir / IMAGES_FOLDER_NAME
    if not sfm_images.exists():
        # copy files from input images to self contained project path
        # os.mkdir(new_input_path)
        log.info("copied input images to self contained SfM reconstruction folder")
        shutil.copytree(IMAGES, sfm_images)

    return sfm_dir


def run_analysis(blender_file: Path, transforms_json: Path, images_path: Path):
    subprocess.run(
        [
            "blender",
            blender_file.resolve(),
            "-b",
            "--python",
            (ROOT / "analyse_dataset.py").resolve(),
            "--",  # script args
            transforms_json.resolve(),
            images_path.resolve(),
        ],
        check=True,
    )
    # todo: read analysis summary for reconstruction


def main(args):
    validate()
    dataset_name = args.dataset
    dataset_path = ROOT / "datasets" / dataset_name
    image_path = dataset_path / IMAGES_FOLDER_NAME
    blender_file = args.validator

    if not dataset_path.exists():
        ABORT("Please provide the path to the dataset containing 'images'")

    if not image_path.exists():
        ABORT(
            "Please provide an 'images' folder containing the images for reconstruction in the dataset"
        )

    if not blender_file.exists():
        ABORT("Please provide a blender file 'test.blend' for running stats")

    log.info(f"Loaded dataset under {dataset_path}")
    log.info("Step 1: reconstructions")
    for c in config:
        # implement timer!
        run_component(c, dataset_name)

    log.info("Step 2: analysis")
    for c in config:
        id = component_id(c, dataset_name)
        out = ROOT / "out" / dataset_name / id / id
        transforms_json = Path(out / "transforms.json").resolve()
        stats_json = Path(out / "stats.json").resolve()  # saved stats
        if stats_json.exists():
            log.info(f"  SKIPPING Analysis - {id}")
            continue
        images = Path(out / "images").resolve()
        run_analysis(blender_file, transforms_json, images)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        help="Name of the dataset in the 'dataset' folder",
    )
    parser.add_argument(
        "--validator",
        type=Path,
        required=False,
        help="Blender validation file path",
    )
    args = parser.parse_args()
    if not args.dataset or not args.validator:
        # if not set, use datasets provided in this file
        for ds in _datasets:
            args.dataset = ds[0]
            args.validator = ROOT / ds[1]
            main(args)
    else:
        main(args)
