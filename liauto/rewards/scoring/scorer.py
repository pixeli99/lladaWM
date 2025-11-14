from ..observation import Observation, Frame, Obstacle
from typing import Dict, List, Optional
from shapely.geometry import LineString, Point
import numpy as np
from itertools import groupby

import copy


class Scorer:
    def __init__(self):
        self.multiplicative_scores = []
        self.score_dict = {}

    def score_frame(self, frame: Frame):
        # initialize score_dict
        self.score_dict = {}
        self.multiplicative_scores = []

        self._calculate_is_not_overspeed(frame.ego_state, frame.sd_v2)
        self._calculate_collision(frame)
        self._calculate_solid_line_violation(frame)
        self._calculate_drivable_area_compliance(frame)
        self._calculate_ttc(frame)

        ego_future_traj = frame.ego_state.get("sdv_future_traj", [])
        ego_future_traj_pts = [
            [item["position"]["x"], item["position"]["y"]] for item in ego_future_traj
        ]
        self._calculate_progress(ego_future_traj_pts, frame.sd_v2)

        self._aggregate_scores()
        return copy.deepcopy(self.score_dict)

    def score_episode(self):
        pass

    def _calculate_collision(self, frame: Frame) -> None:
        ego_geometry = frame.ego_obs.geometry

        # calculate collision with obstacles
        res = frame.obs_str_tree.query(ego_geometry)

        score = 1.0 if len(res) == 0 else 0.0

        self.score_dict["collision"] = score
        self.multiplicative_scores.append(score)

    def _calculate_is_not_overspeed(self, ego_state: Dict, sd_v2: Dict) -> None:
        road_profile = sd_v2.get("road_profile")
        if not road_profile:
            return None

        speed_limits = road_profile.get("speed_limits")
        if not speed_limits:
            return None

        first_limit = speed_limits[0]
        if first_limit.get("type") == 7:
            return None
        navi_vel_limit = first_limit.get("speed_limit")

        velocity_x_m_s = ego_state.get("velocity", {}).get("x", 0)
        velocity_x_kmh = velocity_x_m_s * 3.6

        score = 1.0 if velocity_x_kmh <= navi_vel_limit else 0.0

        self.score_dict["overspeed"] = score
        self.multiplicative_scores.append(score)

    def _calculate_solid_line_violation(self, frame: Frame) -> None:
        """
        LANE_DIVIDER_TYPES:
            0: unset - 未设置类型
            1: solid - 实线
            2: dashed - 虚线
            3: short_dashed - 短虚线
            4: left_solid_right_dashed - 左实右虚
            5: left_dashed_right_solid - 左虚右实
            6: fishbone_solid - 鱼骨实线
            7: fishbone_dashed - 鱼骨虚线
            8: virtual - 虚拟线
            9: shaded_area - 阴影区域
        """
        ego_polygon = frame.ego_obs.geometry
        lane_dividers = frame.static_map.get("lane_divider", [])
        if not lane_dividers:
            return

        def extract_segments(
            points: np.ndarray, labels: list, target_label: int
        ) -> list:
            segments = [
                np.array([points[i] for i in indices])
                for value, indices in groupby(
                    range(len(labels)), key=lambda i: labels[i] == target_label
                )
                if value
            ]
            return segments

        has_violation = False
        for divider in lane_dividers:
            solid_segments = extract_segments(divider["points"], divider["type"], 1)
            for segment in solid_segments:
                if len(segment) < 2:
                    continue
                curve_2d = LineString(segment)
                if ego_polygon.intersects(curve_2d):
                    has_violation = True
                    break
        score = 0 if has_violation else 1.0
        self.score_dict["solid_line_violation"] = score
        self.multiplicative_scores.append(score)

    # TODO: 自车polygon需要转到vcs系下
    def _calculate_drivable_area_compliance(self, frame: Frame) -> None:
        ego_polygon = frame.ego_obs.geometry
        road_boundarys = frame.static_map.get("road_boundary", [])

        has_violation = False
        if road_boundarys:
            for boundary in road_boundarys:
                boundary_2d = boundary["points"][:, :2]
                curve_boundary_2d = LineString(boundary_2d)
                if ego_polygon.intersects(curve_boundary_2d):
                    print(f"find boundary ({boundary['id']}) and ego intersect")
                    has_violation = True
                    break

        score = 0 if has_violation else 1.0
        self.score_dict["drivable_area"] = score
        self.multiplicative_scores.append(score)

    def _calculate_ttc(self, frame: Frame) -> None:
        future_obs_list = frame.future_obs
        tcc_score = 1.0
        for future_obs_info in future_obs_list:
            curr_relative_time = future_obs_info["relative_time"]
            current_ego_x = frame.ego_obs.x + frame.ego_obs.vx * curr_relative_time
            current_ego_y = frame.ego_obs.y + frame.ego_obs.vy * curr_relative_time
            curr_ego_obs = Obstacle(
                x=current_ego_x,
                y=current_ego_y,
                z=frame.ego_obs.z,
                l=frame.ego_obs.l,
                w=frame.ego_obs.w,
                h=frame.ego_obs.h,
                yaw=frame.ego_obs.yaw,
                vx=frame.ego_obs.vx,
                vy=frame.ego_obs.vy,
                label=frame.ego_obs.label,
            )
            curr_ego_geometry = curr_ego_obs._compute_polygon()
            # calculate collision with obstacles
            res = future_obs_info["obs_str_tree"].query(curr_ego_geometry)
            score = 1.0 if len(res) == 0 else 0.0
            if score == 0.0:
                tcc_score = 0.0
                break
        self.score_dict["ttc"] = tcc_score
        self.multiplicative_scores.append(tcc_score)

    def _calculate_progress(self, traj_pts: List, sd_v2: Dict) -> None:
        navi_future_traj = sd_v2.get("coordinates_vcs", [])
        if len(navi_future_traj) < 2 or len(traj_pts) < 2:
            return None
        navi_future_pts = [(item[0], item[1]) for item in navi_future_traj]
        navi_future_line = LineString(navi_future_pts)
        start_point = Point(traj_pts[0][0], traj_pts[0][1])
        end_point = Point(traj_pts[-1][0], traj_pts[-1][1])
        progress_start = navi_future_line.project(start_point)
        progress_end = navi_future_line.project(end_point)
        progress_in_meter = progress_end - progress_start
        if progress_in_meter < 0:
            return None
        self.score_dict["progress"] = round(progress_in_meter, 1)

    def _aggregate_scores(self):
        multiplicative_metric_scores = 1.0
        for _ in self.multiplicative_scores:
            multiplicative_metric_scores *= _

        self.score_dict["full"] = multiplicative_metric_scores
