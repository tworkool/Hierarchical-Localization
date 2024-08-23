from pathlib import Path
import argparse
import logging
import shutil
import subprocess
import traceback

from hloc import (
    extract_features,
    match_features,
    match_dense,
    localize_sfm,
    match_features,
    pairs_from_all,
    triangulation,
    reconstruction,
    pairs_from_exhaustive,
    pairs_from_retrieval,
)

from hloc.utils.parsers import parse_image_list_dir
#from third_party.Neuralangelo.convert_data_to_json import data_to_json
from third_party.Neuralangelo.generate_transforms import data_to_json, ExtraPose
import pycolmap
from hloc.localize_sfm import QueryLocalizer, pose_from_cluster
import numpy as np

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
LOCALIZATION_CAMERA_NAME = "MAIN"

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
    {"extractor": "r2d2", "matcher": "NN-ratio", "name": "r2d2+nn_ratio"},
    {"extractor": "r2d2", "matcher": "NN-mutual", "name": "r2d2+nn_mutual"},
    {"extractor": "r2d2", "matcher": "adalam", "name": "r2d2+adalam"},
    # {
    #    "extractor": "superpoint_max",
    #    "matcher": "loftr_superpoint",
    #    "name": "superpoint+loftr",
    # },
    # {"extractor": "d2net-ss", "matcher": "NN-ratio", "name": "d2net+nn_ratio"},
    # {
    #    "extractor": "s2dnet",
    #    "matcher": "NN-ratio",
    #    "name": "s2d+nn_ratio",
    # },  # TODO: add s2d
    # TODO: add HP (Does not work...)
]

""" config = [
    {
        "extractor": "superpoint_max",
        "matcher": "superpoint+lightglue",
        "name": "TESTER",
    },
] """
_datasets = [
    (
        "LOC-Framlingham+evening_field_8k",
        "datasets/LOC-Framlingham+evening_field_8k/framlingham.blend",
    ),
]
"""
datasets_loc = [
    (
        "LOC-Chateu-img1+evening_field_8k",
        "datasets/LOC-Chateu-img1+evening_field_8k/chateu1.blend",
    ),
]
"""

datasets_loc = [
    (
        "LOC-Bartholomew+evening_field_8k",
        "datasets/LOC-Bartholomew+evening_field_8k/test.blend",
    ),
    (
        "LOC-Chateu-img1+evening_field_8k",
        "datasets/LOC-Chateu-img1+evening_field_8k/chateu1.blend",
    ),
    (
        "LOC-Chateu-img2+evening_field_8k",
        "datasets/LOC-Chateu-img2+evening_field_8k/chateu2.blend",
    ),
    (
        "LOC-Chateu-img3+evening_field_8k",
        "datasets/LOC-Chateu-img3+evening_field_8k/chateu3.blend",
    ),
    (
        "LOC-Framlingham+evening_field_8k",
        "datasets/LOC-Framlingham+evening_field_8k/framlingham.blend",
    ),
    (
        "LOC-Pelegrina+evening_field_8k",
        "datasets/LOC-Pelegrina+evening_field_8k/pelegrina.blend",
    ),
]


datasets_sfm = [
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
    ),
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


def is_loc_dataset(ds: Path):
    return (ds / "images" / "query").exists()


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


# temporary construct for images processed in transform file generator
class TempImage:
    def __init__(self, qvec, tvec, name, point3d_ids=None):
        self.qvec = np.array(qvec)
        self.tvec = np.array(tvec)
        self.name = name
        if point3d_ids:
            self.point3D_ids = point3d_ids

    def todict(self):
        return {
            "qvec": self.qvec,
            "tvec": self.tvec,
            "name": self.name,
        }


# runs selected component on dataset
def run_component(component, dataset) -> Path:
    id = component_id(component, dataset)
    OUT = Path(ROOT / "out" / dataset / id)
    LOC_CHECKPOINT = OUT / ".LOC_CHECKPOINT"

    if OUT.exists():
        log.info(f"  SKIPPING Component - '{id}' already ran for dataset '{dataset}'")
        return None

    log.info(f"  RUNNING Component - '{id}' for dataset '{dataset}'")

    extractor_conf = get_extractor_config(component.get("extractor", None))
    matcher_conf, is_dense = get_matcher_config(component["matcher"])

    IN = Path(ROOT / "datasets" / dataset)
    IMAGES = Path(IN / IMAGES_FOLDER_NAME)
    sfm_pairs = OUT / f"{id}-pairs.txt"
    sfm_dir = OUT / id  # "sparse"

    OUT.mkdir(parents=True, exist_ok=True)

    # get image pairs
    pairs_from_all.main(
        sfm_pairs, IMAGES, recursive=False
    )  # IMPORTANT: non recursive to only load first layer

    feature_path = None

    if is_loc_dataset(IN):
        # use only first layer of images
        image_list = parse_image_list_dir(IMAGES, recursive=False)
    else:
        # use all images
        image_list = parse_image_list_dir(IMAGES, recursive=True)

    # extract features
    if extractor_conf:
        feature_path = extract_features.main(
            extractor_conf, IMAGES, OUT, image_list=image_list
        )

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
    # TODO: run post-init SQL command to set prior_focal_length in MAIN cam to "1.25 * max(width_in_px, height_in_px)"
    # as described in https://colmap.github.io/tutorial.html#feature-detection-and-extraction
    gt_focal_length = 26  # mm
    POST_IMAGE_IMPORT_SQL = f"UPDATE cameras SET prior_focal_length = {gt_focal_length} WHERE camera_id IN (SELECT c.camera_id FROM cameras c JOIN images i ON c.camera_id = i.camera_id WHERE i.name NOT LIKE '%{LOCALIZATION_CAMERA_NAME}%')"
    reconstruction_args = {
        "sfm_dir": sfm_dir,
        "image_dir": IMAGES,
        "pairs": sfm_pairs,
        "features": feature_path,
        "matches": match_path,
        "image_list": image_list,
        # "post_image_import_sql": POST_IMAGE_IMPORT_SQL,
    }
    # if is_loc_dataset(IN):
    #    reconstruction_args["camera_mode"] = pycolmap.CameraMode.PER_FOLDER
    #    # single camera for CAI

    model = reconstruction.main(**reconstruction_args)

    # export transforms args
    """ transforms_args = {
        "data_dir": str(sfm_dir.resolve()),
        "scene_type": "outdoor",
        "image_dir": str(IMAGES.resolve()),
        "name": "transforms.json",
    } """
    transforms_args = {
        "reconstruction": model,
        "image_path": IMAGES,
        "data_path": sfm_dir,
    }

    # TODO: localization pipeline?
    if is_loc_dataset(IN):
        from hloc.utils.io import generate_query_list, generate_localization_pairs

        QUERY_IMAGE = parse_image_list_dir(IMAGES / "query", recursive=False)
        if len(QUERY_IMAGE) == 0:
            raise Exception("no query images found")
        QUERY_IMAGE = f"query/{QUERY_IMAGE[0]}"
        QUERY_IMAGE_PATH = IMAGES / QUERY_IMAGE
        retrieval_conf = extract_features.confs["netvlad"]

        query_list = OUT / "queries_with_intrinsics.txt"
        query_camera = pycolmap.infer_camera_from_image(QUERY_IMAGE_PATH)
        generate_query_list([query_camera], [QUERY_IMAGE], query_list)

        netvlad_features = extract_features.main(
            retrieval_conf, IMAGES, OUT, overwrite=True
        )
        retrieval_path = OUT / "netvlad_retrieval_pairs.txt"
        pairs_from_retrieval.main(netvlad_features, retrieval_path, num_matched=8)
        # pairs_from_all.main(retrieval_path, IMAGES, recursive=True)

        feature_path = None
        # extract features again for the query image and match them
        if extractor_conf:
            feature_path = extract_features.main(
                extractor_conf, IMAGES, OUT, image_list=[QUERY_IMAGE], overwrite=True
            )
        # match features
        if is_dense:
            feature_path, match_path = match_dense.main(
                matcher_conf,
                retrieval_path,
                image_dir=IMAGES,
                export_dir=OUT,
                features_ref=feature_path,
                overwrite=True,
            )
        else:
            match_path = match_features.main(
                matcher_conf,
                retrieval_path,
                features=feature_path,
                matches=match_path,
                overwrite=True,
            )

        results = OUT / "localize_results.txt"
        ret = localize_sfm.main(
            reference_sfm=sfm_dir,  # sparse reconstruction colmap directory
            queries=query_list,  # txt file
            retrieval=retrieval_path,  # txt file
            features=feature_path,  # h5 file
            matches=match_path,  # h5 file
            results=results,  # output txt file
        )  # not required with SuperPoint+SuperGlue

        """
        # set checkpoint
        with open(LOC_CHECKPOINT, "w+") as f:
            f.write("")
        # apply localization
        QUERY_IMAGE_PATH = IMAGES / "query"
        QUERY_IMAGE = parse_image_list_dir(QUERY_IMAGE_PATH, recursive=False)
        if len(QUERY_IMAGE) == 0:
            raise Exception("no query images found")
        QUERY_IMAGE = f"query/{QUERY_IMAGE[0]}"

        references_registered = [model.images[i].name for i in model.reg_image_ids()]

        # extract features and match query image again
        loc_pairs = OUT / f"{id}-pairs_loc.txt"
        extract_features.main(
            extractor_conf,
            IMAGES,
            image_list=[QUERY_IMAGE],
            feature_path=feature_path,
            overwrite=True,
        )
        '''retrieval_conf = extract_features.confs["netvlad"]
        #retrieval_path = OUT / retrieval_conf["output"]
        retrieval_path = extract_features.main(
            retrieval_conf,
            IMAGES,
            export_dir=OUT,
            image_list=[QUERY_IMAGE],
            overwrite=True,
        )
        loc_pairs = OUT / f"{id}-pairs_loc.txt"
        pairs_from_retrieval.main(retrieval_path, loc_pairs, num_matched=5) '''
        pairs_from_exhaustive.main(
            loc_pairs, image_list=[QUERY_IMAGE], ref_list=references_registered
        )
        match_features.main(
            matcher_conf,
            loc_pairs,
            features=feature_path,
            matches=match_path,
            overwrite=True,
        )

        # register new camera and localize it
        camera = pycolmap.infer_camera_from_image(IMAGES / QUERY_IMAGE)
        ref_ids = [
            model.find_image_with_name(n).image_id for n in references_registered
        ]
        conf = {
            "estimation": {"ransac": {"max_error": 12}},
            "refinement": {"refine_focal_length": True, "refine_extra_params": True},
        }
        localizer = QueryLocalizer(model, conf)
        ret, logs = pose_from_cluster(
            localizer, QUERY_IMAGE, camera, ref_ids, feature_path, match_path
        )
        """

        # extract localized camera
        """ localized_image = TempImage(
            ret["cam_from_world"].rotation.todict()[
                "quat"
            ],  # do this to as a workaround to extract numpy array instead of Rotation3D quaternion
            ret["cam_from_world"].translation,
            QUERY_IMAGE,
            None,  # [i + 1 if e else -1 for i, e in enumerate(ret["inliers"])],
        )
        transforms_args["extra_cam"] = localized_image """
        transforms_args["extra_pose"] = ExtraPose(
            image=pycolmap.Image(cam_from_world=ret['cam_from_world']),
            camera=query_camera,
            img_name=QUERY_IMAGE
        )

    # export transforms and copy images
    #data_to_json(transforms_args)
    data_to_json(**transforms_args)

    sfm_images = sfm_dir / IMAGES_FOLDER_NAME
    if not sfm_images.exists():
        # copy files from input images to self contained project path
        # os.mkdir(new_input_path)
        log.info("copied input images to self contained SfM reconstruction folder")
        shutil.copytree(IMAGES, sfm_images)

    return sfm_dir


def run_analysis(blender_file: Path, transforms_json: Path, images_path: Path):
    if not blender_file.exists():
        raise Exception("Blender file missing")
    if not transforms_json.exists():
        raise Exception("Transforms file missing")
    if not images_path.exists():
        raise Exception("Images path missing")

    dst = transforms_json.parent
    local_blender_file = dst / blender_file.name
    if not local_blender_file.exists():
        log.info(f"  Copying Blender file to local folder for analysis")
        # copy blender file to local folder
        shutil.copy(blender_file, dst)

    log.info(f"  Running Analysis")
    subprocess.run(
        [
            "blender",
            local_blender_file.resolve(),
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
        # run_component(c, dataset_name)
        # continue
        try:
            run_component(c, dataset_name)
        except Exception as E:
            log.error(
                f"  There was a problem with the component run: {E} {traceback.format_exc()}"
            )

    log.info("Step 2: analysis")
    for c in config:
        id = component_id(c, dataset_name)
        out = ROOT / "out" / dataset_name / id / id
        transforms_json = Path(out / "transforms.json").resolve()
        stats_json = Path(out / "stats.json").resolve()  # saved stats
        if stats_json.exists() and not args.overwrite_analysis:
            log.info(f"  SKIPPING Analysis - {id}")
            continue

        log.info(f"  RUNNING Analysis - {id}")
        images = Path(out / "images").resolve()
        try:
            run_analysis(blender_file, transforms_json, images)
        except Exception as E:
            log.error(f"  There was a problem with the component analysis: {E}")
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
    parser.add_argument(
        "--overwrite_analysis",
        type=bool,
        required=False,
        help="Overwrite Analysis Results",
        default=False,
    )
    args = parser.parse_args()
    if not args.dataset or not args.validator:
        # if not set, use datasets provided in this file
        datasets = []
        # build datasets list
        datasets.extend(datasets_loc)
        #datasets.extend(datasets_sfm)
        #datasets.extend(_datasets)
        for ds in datasets:
            args.dataset = ds[0]
            args.validator = ROOT / ds[1]
            main(args)
    else:
        main(args)
