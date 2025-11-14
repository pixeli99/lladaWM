import copy
import os
import yaml
import glob

from liauto.projects.end2end.common import (
    DEBUG,
    FINETUNE,
    EVAL_WITH_PNP,
    data_config,
    input_modality,
    use_BDA,
    point_cloud_range,
    det_infer_cfg,
    with_motion_states,
    with_light_states,
    ONLY_INFER_PNP,
    at128_point_cloud_range,
    at128_voxel_size,
    cfg_root,
)

dataset_type = "AllinOneDataset"
yml_path = os.path.join(cfg_root, f"allinone_datasets_val.yaml")
dataset_config = yaml.load(open(yml_path), Loader=yaml.FullLoader)
assert "allinone" in dataset_config
data_mode = "full" if bool(int(os.environ.get("FULL_TRAIN", False))) else "success"
assert data_mode in dataset_config["allinone"]

if DEBUG:
    txt_root = dataset_config["allinone"][data_mode].get("debug", None)
elif FINETUNE:
    txt_root = dataset_config["allinone"][data_mode].get("finetune", None)
else:
    stage_key_map = dict(float="train", qat="train", qatpnp="train")
    if os.environ.get("PNP_FINETUNE", None) == "True":
        key = stage_key_map["qatpnp"]
    elif os.environ.get("TRAINING_STEP", None) is not None:
        stage = os.environ.get("TRAINING_STEP", None)
        key = stage_key_map[stage]
    else:
        key = stage_key_map["float"]
    txt_root = dataset_config["allinone"][data_mode].get(key, None)
    assert txt_root is not None


blacklist_files = glob.glob(os.path.join(cfg_root, "blacklist", "*.txt"))
whitelist_files = glob.glob(os.path.join(cfg_root, "whitelist", "*.txt"))
blacklist_files = os.environ.get("BLACK_LIST", blacklist_files)
whitelist_files = os.environ.get("WHITE_LIST", whitelist_files)