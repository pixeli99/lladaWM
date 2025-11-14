from dataclasses import dataclass, field
from typing import List, Dict

import numpy as np
import copy
from shapely.geometry import Polygon
from shapely.strtree import STRtree
import math


class Obstacle:
    def __init__(
        self,
        x: float,
        y: float,
        z: float,
        l: float,
        w: float,
        h: float,
        yaw: float,
        vx: float,
        vy: float,
        label: int,
    ):
        self.x = x
        self.y = y
        self.z = z
        self.l = l
        self.w = w
        self.h = h
        self.yaw = yaw
        self.vx = vx
        self.vy = vy
        self.label = label
        self.geometry = None
        self._compute_polygon()

    def _compute_polygon(self):
        if self.geometry is not None:
            return self.geometry

        rot_matrix = np.array(
            [
                [np.cos(self.yaw), -np.sin(self.yaw)],
                [np.sin(self.yaw), np.cos(self.yaw)],
            ]
        )
        half_length = self.l / 2.0
        half_width = self.w / 2.0
        local_corners = np.array(
            [
                [half_length, half_width],
                [half_length, -half_width],
                [-half_length, -half_width],
                [-half_length, half_width],
            ]
        )

        corners = []
        for corner in local_corners:
            rotated = np.dot(corner, rot_matrix.T)
            global_x = self.x + rotated[0]
            global_y = self.y + rotated[1]
            pt = (global_x, global_y)
            corners.append(pt)

        self.geometry = Polygon(corners)
        return self.geometry


class Frame:
    def __init__(self, info: dict):
        if info is None:
            info = {}

        self.scene_token = info.get("scene_token", "unknown_vid")
        self.timestamp = info.get("timestamp", -1)
        self.frame_idx = info.get("frame_idx", -1)
        self.ego_state = info.get("ego_info", {})
        self.static_map = info.get("map_gt", {})
        # ！ 注意不是sd_v2_navigation_information
        self.sd_v2 = info.get("sd_env", {})
        self.obs_str_tree = None
        self.obs = self.create_obstacles(info)
        self.ego_obs = self.create_ego_obstacle()
        self.future_obs = self.create_future_obstacles(info)

        self.image_path = {
            "CAM_FRONT": {},
            "CAM_BACK": {},
            "CAM_BACK_LEFT": {},
            "CAM_FRONT_LEFT": {},
            "CAM_BACK_RIGHT": {},
            "CAM_FRONT_RIGHT": {},
        }
        for key in self.image_path.keys():
            self.image_path[key]["data_path"] = (
                info.get("cams", {}).get(key, {}).get("data_path", "")
            )
            self.image_path[key]["data_s3_path"] = (
                info.get("cams", {}).get(key, {}).get("data_s3_path", "")
            )

    def create_obstacles(self, info: Dict) -> List[Obstacle]:
        gt_boxes = info.get("gt_boxes", [])
        gt_labels = info.get("gt_label", [])

        if len(gt_boxes) != len(gt_labels) or len(gt_boxes) == 0:
            self.obs_str_tree = STRtree([])
            return []

        obs_list = [
            Obstacle(
                x=box[0],
                y=box[1],
                z=box[2],
                l=box[3],
                w=box[4],
                h=box[5],
                yaw=box[6],
                vx=box[7],
                vy=box[8],
                label=gt_labels[i],
            )
            for i, box in enumerate(gt_boxes)
        ]

        geometries = [obs._compute_polygon() for obs in obs_list]
        self.obs_str_tree = STRtree(geometries)
        return obs_list

    def calculate_abs_radian_diff(self, radian1, radian2):
        diff = radian2 - radian1
        diff = (diff + math.pi) % (2 * math.pi) - math.pi
        return math.fabs(diff)

    def create_future_obstacles(self, info: Dict) -> List:
        gt_boxes = info.get("gt_boxes", [])
        gt_labels = info.get("gt_label", [])
        gt_ids = info.get("gt_inds", [])
        dynamic_obstacals = info.get("dynamic_info", {}).get("obstacles", {})

        if len(gt_boxes) != len(gt_labels) or len(gt_boxes) == 0:
            return []

        obs_list = [
            Obstacle(
                x=box[0],
                y=box[1],
                z=box[2],
                l=box[3],
                w=box[4],
                h=box[5],
                yaw=box[6],
                vx=box[7],
                vy=box[8],
                label=gt_labels[i],
            )
            for i, box in enumerate(gt_boxes)
        ]

        dynamic_obs_dict = {}
        for obs_id, dynamic_info in dynamic_obstacals.items():
            if "future_traj" in dynamic_info.keys():
                dynamic_obs_dict[obs_id] = []
                for time_idx in range(10):
                    dynamic_obs_timestamp_dict = {}
                    for future_traj_step in dynamic_info["future_traj"]:
                        relative_time = future_traj_step.get("relative_time", -1)
                        if round(relative_time * 10) == (
                            time_idx + 1
                        ) and future_traj_step.get("is_valid", False):
                            try:
                                dynamic_obs_timestamp_dict[
                                    "relative_time"
                                ] = future_traj_step["relative_time"]
                                dynamic_obs_timestamp_dict["x"] = future_traj_step[
                                    "position"
                                ]["x"]
                                dynamic_obs_timestamp_dict["y"] = future_traj_step[
                                    "position"
                                ]["y"]
                                dynamic_obs_timestamp_dict["vx"] = future_traj_step[
                                    "velocity"
                                ]["x"]
                                dynamic_obs_timestamp_dict["vy"] = future_traj_step[
                                    "velocity"
                                ]["y"]
                                dynamic_obs_timestamp_dict["yaw"] = future_traj_step[
                                    "theta"
                                ]
                            except:
                                pass
                    dynamic_obs_dict[obs_id].append(dynamic_obs_timestamp_dict)

        # dynamic_obs_dict_refine = copy.deepcopy(dynamic_obs_dict)
        dynamic_obs_dict_refine = {}
        for obs_id, dynamic_obs_timestamp_list in dynamic_obs_dict.items():
            string_id_list = [str(element) for element in gt_ids]
            if obs_id in string_id_list:
                obs_index = string_id_list.index(obs_id)
                pre_yaw = obs_list[obs_index].yaw
                new_dynamic_obs_timestamp_list = []
                for item in dynamic_obs_timestamp_list:
                    if len(item) != 0:
                        yaw_diff = self.calculate_abs_radian_diff(pre_yaw, item["yaw"])
                        if yaw_diff <= math.radians(10):
                            new_dynamic_obs_timestamp_list.append(item)
                            pre_yaw = item["yaw"]
                        else:
                            item["yaw"] = pre_yaw
                            new_dynamic_obs_timestamp_list.append(item)
                    else:
                        new_dynamic_obs_timestamp_list.append(item)
                dynamic_obs_dict_refine[obs_id] = copy.deepcopy(
                    new_dynamic_obs_timestamp_list
                )

        future_obs_list = []
        for time_idx in range(10):
            future_obs_dict = {}
            for value in dynamic_obs_dict_refine.values():
                if len(value[time_idx]) != 0:
                    future_obs_dict["relative_time"] = value[time_idx]["relative_time"]
                    break
            new_obs_list = []
            for index, obs_info in enumerate(obs_list):
                new_obs_id = str(gt_ids[index])
                if new_obs_id in dynamic_obs_dict_refine.keys():
                    if len(dynamic_obs_dict_refine[new_obs_id][time_idx]) != 0:
                        new_obs = Obstacle(
                            x=dynamic_obs_dict_refine[new_obs_id][time_idx]["x"],
                            y=dynamic_obs_dict_refine[new_obs_id][time_idx]["y"],
                            z=obs_info.z,
                            l=obs_info.l,
                            w=obs_info.w,
                            h=obs_info.h,
                            yaw=dynamic_obs_dict_refine[new_obs_id][time_idx]["yaw"],
                            vx=dynamic_obs_dict_refine[new_obs_id][time_idx]["vx"],
                            vy=dynamic_obs_dict_refine[new_obs_id][time_idx]["vy"],
                            label=obs_info.label,
                        )
                        new_obs_list.append(new_obs)
            if len(new_obs_list) > 0 and "relative_time" in future_obs_dict.keys():
                curr_geometries = [obs._compute_polygon() for obs in new_obs_list]
                future_obs_dict["obs_list"] = new_obs_list
                future_obs_dict["obs_str_tree"] = STRtree(curr_geometries)
                future_obs_list.append(future_obs_dict)

        return future_obs_list

    def create_ego_obstacle(self) -> Obstacle:
        """
        Creates an obstacle representing the ego vehicle.

        Note:
            There is an issue. The (x, y) coordinates should be recalculated using the length from the rear axle to the front bumper of the ego vehicle.

        Returns:
            Obstacle: An obstacle object with the ego vehicle's state parameters.
        """
        return Obstacle(
            x=self.ego_state["position"]["x"],
            y=self.ego_state["position"]["y"],
            z=0,
            l=self.ego_state["length"],
            w=self.ego_state["width"],
            h=1.8,
            yaw=0,
            vx=self.ego_state["velocity"]["x"],
            vy=self.ego_state["velocity"]["y"],
            label=0,
        )


class Observation:
    def __init__(self):
        self.frames = []

    def update(self, raw_info: dict):
        cur_frame = Frame(raw_info)
        self.frames.append(cur_frame)
        if len(self.frames) > 1:
            self.frames.pop(0)

    def __getitem__(self, idx: int):
        return self.frames[idx]

    def get_latest(self):
        return self.frames[-1]
