from pathlib import Path
import sys
import shutil

from hloc import (
    extract_features,
    match_features,
    match_dense,
    reconstruction,
    visualization,
    pairs_from_all,
    localize_sfm,
)

ROOT = Path().absolute()
# add root to path in order to execute scripts
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def main():
    images = Path("datasets/framlingham2/images/")

    outputs = Path("outputs/sfm/")
    sfm_pairs = outputs / "pairs.txt"
    sfm_dir = outputs / "framlingham_s2d_nn"
    sfm_images = sfm_dir / "images"

    # retrieval_conf = extract_features.confs["netvlad"]
    feature_conf = extract_features.confs["s2dnet"]
    matcher_conf = match_features.confs["NN-mutual"]
    matcher_conf = match_dense.confs["loftr"]

    pairs_from_all.main(sfm_pairs, images)

    feature_path = extract_features.main(feature_conf, images, outputs)

    '''
    match_path = match_features.main(
        conf=matcher_conf,
        pairs=sfm_pairs,
        features=feature_conf["output"],
        export_dir=outputs,
    )
    '''

    feature_path, match_path = match_dense.main(
        conf=matcher_conf,
        pairs=sfm_pairs,
        image_dir=images,
        export_dir=outputs,
        #features_ref=feature_path,
    )
    return 1


if __name__ == "__main__":
    main()
