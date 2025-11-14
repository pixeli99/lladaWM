# Copyright (c) OpenMMLab. All rights reserved.
import json
import math
import os
import os.path
import pickle
import random
from collections import defaultdict
from math import ceil
from random import randint as randint

import cv2
import mmcv
import numpy as np
import torch
from mmcv.image.photometric import imnormalize
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile
from mmdet3d.core.points import BasePoints, get_points_type
from PIL import Image
from pyquaternion import Quaternion
import open3d as o3d
from pypcd import pypcd
import json
import random

import av
import cv2
import decord
import imageio
import math
from decord import VideoReader

from mmdet.datasets.builder import PIPELINES

import numpy as np
import os
from functools import reduce
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix

import math
from copy import deepcopy
from collections import deque
from scipy import ndimage
from shapely.ops import nearest_points
from shapely.geometry import (
    Point,
    MultiPoint,
    LineString,
    Polygon,
    MultiLineString,
    box,
)

class LoadNavigation(object):
    def __init__(
        self,
        start_distur=[-5, 10],
        angle_distur=[-2, 2],
        left_dis=[8, 16],
        right_dis=[1, 8],
        use_ego_angle=True,
        sample_pts_num=20,
        pc_range=[-18.0, -24.0, -5.0, 120.0, 24.0, 3.0],
    ):
        self.train = True
        self.thres_angle = 70
        self.pc_range = pc_range
        self.sample_pts_num = sample_pts_num
        self.use_ego_angle = use_ego_angle
        self.angle_distur = angle_distur
        self.start_distur = start_distur
        self.left_dis = left_dis
        self.right_dis = right_dis
        self.navi_patchbox = Polygon(
            [
                (0.0, pc_range[1]),
                (0.0, pc_range[4]),
                (pc_range[3], pc_range[4]),
                (pc_range[3], pc_range[1]),
                (0.0, pc_range[1]),
            ]
        )

    # Functions below originally authored by xulu3
    @staticmethod
    def angle_protect(angle):
        return angle % 360

    @staticmethod
    def cond1_angle(vector1, vector2):
        angle = 0
        product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        cos_angle = np.dot(vector1, vector2) / product
        if product and (not np.isnan(product)) and -1 <= cos_angle <= 1:
            angle = np.degrees(np.arccos(cos_angle))
        return angle

    @staticmethod
    def angle_with_x_axis(vector):
        vector = np.array(vector)
        angle_radians = np.arctan2(vector[1], vector[0])
        angle_degrees = np.degrees(angle_radians)
        if angle_degrees < 0:
            angle_degrees += 360
        return angle_degrees

    @staticmethod
    def sample_pts_from_line_sdprior(line, sample_pts_num=20):
        line = LineString(line)
        distances = np.linspace(0, line.length, sample_pts_num)
        sampled_points = np.array(
            [list(line.interpolate(distance).coords) for distance in distances]
        ).reshape(-1, 2)
        return sampled_points

    @staticmethod
    def sd_measurement(sd_prior, navi_gt):
        navi_gt_ls = LineString(navi_gt)
        start_d = navi_gt_ls.project(Point(sd_prior[0]))
        end_d = navi_gt_ls.project(Point(sd_prior[-1]))
        if start_d >= end_d:
            return 0.0, None
        sd_ls = LineString(sd_prior)
        projected_segment = LineString(
            navi_gt_ls.interpolate(t, normalized=False) for t in [start_d, end_d]
        )
        return 1.0 * (end_d - start_d) / sd_ls.length, projected_segment

    @staticmethod
    def find_correct_order(sd_prior, navi_gt):
        navi_gt_ls = LineString(navi_gt)
        start_d = navi_gt_ls.project(Point(sd_prior[0]))
        end_d = navi_gt_ls.project(Point(sd_prior[-1]))
        if start_d <= end_d:
            return True, sd_prior
        else:
            return False, sd_prior[::-1]

    @staticmethod
    def extend_sd_to_BEV_range(polyline_points, pc_range):
        if len(polyline_points) < 2:
            return polyline_points

        # navi_pc_range = [-18.0, -24.0, -5.0, 120.0, 24.0, 3.0]
        polyline = LineString(polyline_points)
        bbox = [0, pc_range[1], pc_range[3], pc_range[4]]

        last_point = polyline_points[-1]
        second_last_point = polyline_points[-2]
        delta_x = last_point[0] - second_last_point[0]
        delta_y = last_point[1] - second_last_point[1]

        candidates = []

        # 判断延长方向，只做正向延长
        def is_forward_extension(candidate, last, second_last):
            vec_to_candidate = (candidate[0] - last[0], candidate[1] - last[1])
            vec_last_line = (last[0] - second_last[0], last[1] - second_last[1])
            dot_product = np.dot(vec_to_candidate, vec_last_line)
            return dot_product >= 0

        # 避免除零错误
        if delta_x != 0:
            slope = delta_y / delta_x
            # 计算与上下边界的交点
            for y in [bbox[1], bbox[3]]:  # ymin, ymax
                x = (y - last_point[1]) / slope + last_point[0]
                if bbox[0] <= x <= bbox[2]:
                    candidate = (x, y)
                    if is_forward_extension(candidate, last_point, second_last_point):
                        candidates.append(candidate)

        # 如果斜率不是垂直的（即delta_x不为0），计算与左右边界的交点
        for x in [bbox[0], bbox[2]]:  # xmin, xmax
            if delta_x != 0:
                y = slope * (x - last_point[0]) + last_point[1]
            else:  # 如果是垂直线，直接使用最后一个点的y坐标
                y = last_point[1]
            if bbox[1] <= y <= bbox[3]:  # 如果交点在bbox的上下边界之内
                candidate = (x, y)
                if is_forward_extension(candidate, last_point, second_last_point):
                    candidates.append(candidate)

        # 找到距离最后一个点最近的交点作为延长的目标点
        if candidates:
            extended_point = min(
                candidates, key=lambda pt: Point(pt).distance(Point(last_point))
            )
            extended_polyline_points = np.vstack(
                (polyline_points, np.array(extended_point)[np.newaxis, :])
            )
            extended_polyline = LineString(extended_polyline_points)
            return np.array(extended_polyline.coords)

        return np.array(polyline.coords)

    @staticmethod
    def cut_linestring_at_p(ls, p, first_half=True):
        # if multi_points use point with min x
        if isinstance(p, MultiPoint):
            p = min(p.geoms, key=lambda point: point.x)
        if first_half:
            d = ls.project(p)
        else:
            d = ls.length - ls.project(p)
        points_count = 20
        distance_delta = 1.0 * d / points_count
        if first_half:
            distances = (distance_delta * i for i in range(points_count))
        else:
            distances = (
                distance_delta * i + ls.project(p) for i in range(points_count)
            )
        points = [ls.interpolate(distance) for distance in distances]
        return LineString(points)

    def navi_close_angle(self, prior, navi, angle_thr):
        _, closest_seg = self.sd_measurement(prior, navi)
        if closest_seg is not None:
            closest_seg_navi = np.array(closest_seg.coords)
        else:
            return False
        if len(prior) < 2 or len(closest_seg_navi) < 2:
            return False
        angles_prior = prior[-1] - prior[0]
        angles_navi = closest_seg_navi[-1] - closest_seg_navi[0]
        angle_diff = self.cond1_angle(angles_prior, angles_navi)
        if angle_diff < angle_thr:
            return True
        return False

    def update_sd_dicts(
        self, sd_prior, same_order, ordered_cut_sd, connect_dict, link_dict
    ):
        link_id = sd_prior["link_id"]
        if same_order:
            snode = sd_prior["snode_id"]
            enode = sd_prior["enode_id"]
        else:
            snode = sd_prior["enode_id"]
            enode = sd_prior["snode_id"]

        if link_id not in link_dict.keys():
            link_dict[link_id] = deepcopy(sd_prior)
            cut_points = self.sample_pts_from_line_sdprior(
                ordered_cut_sd, self.sample_pts_num
            )
            link_dict[link_id]["cut_points"] = cut_points[np.newaxis, ...]
            link_dict[link_id]["snode_id"] = snode
            link_dict[link_id]["enode_id"] = enode

        if connect_dict.get(snode, {}):
            if connect_dict[snode].get(enode, []):
                connect_dict[snode][enode].append(link_id)
            else:
                connect_dict[snode][enode] = [link_id]
        else:
            connect_dict[snode] = {enode: [link_id]}

        return deepcopy(connect_dict), deepcopy(link_dict)

    def approx_same_angle(self, line1, line2, thres_angle):
        vector1 = line1[-1] - line1[0]
        vector2 = line2[-1] - line2[0]
        if self.cond1_angle(vector1, vector2) < thres_angle:
            return True
        else:
            return False

    def angle_adjust(self, angle, ego_angle):
        distur_angle = angle
        if self.train:
            distur_angle += np.random.uniform(
                self.angle_distur[0], self.angle_distur[1]
            )
        if angle > 0:
            if self.train and ego_angle > 0:
                ego_angle += np.random.uniform(
                    self.angle_distur[0], self.angle_distur[1]
                )
        else:
            ego_angle = 0
        # ego direction convert to math direction

        distur_angle = distur_angle * -1
        ego_angle = ego_angle * -1
        distur_angle = self.angle_protect(distur_angle)
        ego_angle = self.angle_protect(ego_angle)
        distur_angle = distur_angle - ego_angle
        distur_angle = self.angle_protect(distur_angle)
        return distur_angle

    def calculate_endpoint(self, distur_angle, turning_pivot_x, turning_pivot_y):
        if distur_angle == 0 or distur_angle == 180 or distur_angle == 360:
            final_x = self.pc_range[3]
            final_y = turning_pivot_y
        elif distur_angle > 0 and distur_angle < 90:
            end_y = math.tan(math.radians(distur_angle)) * (
                self.pc_range[3] - turning_pivot_x
            )
            if end_y > self.pc_range[4]:
                final_x = (
                    self.pc_range[4] / math.tan(math.radians(distur_angle))
                    + turning_pivot_x
                )
                final_y = self.pc_range[4]
            else:
                final_x = self.pc_range[3]
                final_y = end_y + turning_pivot_y
        elif distur_angle == 90:
            final_x = turning_pivot_x
            final_y = self.pc_range[4]
        elif distur_angle > 90 and distur_angle < 180:
            end_y = math.tan(math.radians(distur_angle)) * (
                self.pc_range[0] - turning_pivot_x
            )
            if end_y >= self.pc_range[4]:
                final_x = (
                    self.pc_range[4] / math.tan(math.radians(distur_angle))
                    + turning_pivot_x
                )
                final_y = self.pc_range[4]
            else:
                final_x = self.pc_range[0]
                final_y = end_y + turning_pivot_y
        elif distur_angle > 180 and distur_angle < 270:
            end_y = math.tan(math.radians(distur_angle)) * (
                self.pc_range[0] - turning_pivot_x
            )
            if end_y < self.pc_range[1]:
                final_x = (
                    self.pc_range[1] / math.tan(math.radians(distur_angle))
                    + turning_pivot_x
                )
                final_y = self.pc_range[1]
            else:
                final_x = self.pc_range[0]
                final_y = end_y + turning_pivot_y

        elif distur_angle == 270:
            final_x = turning_pivot_x
            final_y = self.pc_range[1]
        elif distur_angle > 270 and distur_angle < 360:
            end_y = math.tan(math.radians(distur_angle)) * (
                self.pc_range[3] - turning_pivot_x
            )
            if end_y < self.pc_range[1]:
                final_x = (
                    self.pc_range[1] / math.tan(math.radians(distur_angle))
                    + turning_pivot_x
                )
                final_y = self.pc_range[1]
            else:
                final_x = self.pc_range[3]
                final_y = end_y + turning_pivot_y
        else:
            raise NotImplementedError

        return [[final_x, final_y]]

    def before_turning(self, distur_angle, angle, turning_start, s_y):
        middle_point = []

        if angle > 10 and angle < 170:
            # turnning right
            if self.train:
                intersection = np.random.uniform(self.right_dis[0], self.right_dis[1])
            else:
                intersection = (self.right_dis[0] + self.right_dis[1]) / 2.0
        elif angle > 190 and angle < 350:
            if self.train:
                intersection = np.random.uniform(self.left_dis[0], self.left_dis[1])
            else:
                intersection = (self.left_dis[0] + self.left_dis[1]) / 2.0
        else:
            intersection = 0

        if turning_start <= self.start_distur[1]:
            # 距离路口较近
            distur_turning_start_x = turning_start
            turning_pivot_x = distur_turning_start_x + intersection
        elif turning_start >= self.pc_range[3] - intersection:
            # 距离路口较远
            distur_turning_start_x = turning_start
            intersection = self.pc_range[3] - distur_turning_start_x
            turning_pivot_x = self.pc_range[3] - 2
        else:
            distur_turning_start_x = turning_start
            if self.train:
                distur_turning_start_x += np.random.uniform(
                    self.start_distur[0], self.start_distur[1]
                )
            distur_turning_start_x = min(distur_turning_start_x, self.pc_range[3] - 2)
            turning_pivot_x = distur_turning_start_x + intersection
            turning_pivot_x = min(turning_pivot_x, self.pc_range[3] - 2)
        middle_point.append([distur_turning_start_x, s_y])
        middle_point.append([turning_pivot_x, s_y])

        end_point = self.calculate_endpoint(distur_angle, turning_pivot_x, s_y)
        middle_point.extend(end_point)
        return middle_point

    def sample_pts_from_line(self, points):
        points = np.array(points)
        points[:, 0] = np.clip(points[:, 0], self.pc_range[0], self.pc_range[3])
        points[:, 1] = np.clip(points[:, 1], self.pc_range[1], self.pc_range[4])
        line = LineString(points)
        distances = np.linspace(0, line.length, self.sample_pts_num)
        sampled_points = (
            np.array(
                [list(line.interpolate(distance).coords) for distance in distances]
            )
            .reshape(-1, 2)
            .astype(np.float32)
        )
        sampled_points[:, 0] = np.clip(
            sampled_points[:, 0], self.pc_range[0], self.pc_range[3]
        )
        sampled_points[:, 1] = np.clip(
            sampled_points[:, 1], self.pc_range[1], self.pc_range[4]
        )
        return sampled_points

    def check_sd_navi_same_direction(self, prior, navi, angle_thr):
        navi_pos = np.array([(x, y) for x, y in navi if x > 0])
        if len(prior) < 2 or len(navi_pos) < 2:
            return False
        angles_prior = np.diff(prior, axis=0)
        angles_navi = np.diff(navi_pos, axis=0)
        angle_diff_prior = self.angle_with_x_axis(angles_prior[-1])
        angle_diff_navi = self.angle_with_x_axis(angles_navi[-1])
        angle_diff_prior_s = self.angle_with_x_axis(angles_prior[0])
        angle_diff_navi_s = self.angle_with_x_axis(angles_navi[0])
        angle_dist_start = min(
            abs(angle_diff_navi - angle_diff_prior),
            abs(360 - angle_diff_navi - angle_diff_prior),
        )
        angle_dist_end = min(
            abs(angle_diff_navi_s - angle_diff_prior_s),
            abs(360 - angle_diff_navi_s - angle_diff_prior_s),
        )
        if angle_dist_start < angle_thr and angle_dist_end < angle_thr:
            return True
        return False

    def choose_next_sd(self, candidates_dict, navi_gt, link_dict, invalid_thres=0.05):
        candidates = []
        for enode in candidates_dict.keys():
            candidates += candidates_dict[enode]
        if len(candidates) == 1:
            points = link_dict[candidates[0]]["cut_points"][0]
            score, project_seg = self.sd_measurement(points, navi_gt)
            if score < invalid_thres:
                return -1
            return candidates[0]

        best_measurement = 0
        best_candidate = -1
        for candidate in candidates:
            if link_dict.get(candidate, {}):
                points = link_dict[candidate]["cut_points"][0]
                score, project_seg = self.sd_measurement(points, navi_gt)
                if score > best_measurement:
                    if self.check_sd_navi_same_direction(
                        np.array(project_seg.coords), points, self.thres_angle
                    ):
                        best_measurement = score
                        best_candidate = candidate

        if best_measurement < invalid_thres:
            best_candidate = -1
        return best_candidate

    def find_best_sd_path(self, closest_sd_id, navi_gt, connect_dict, link_dict):
        # connect_dict: {node_in: {node_out: [link_ids]}}
        # link_dict: {link_id: link_content}
        cur_path = [closest_sd_id]
        continue_extend_path = True
        counter = 0

        while continue_extend_path:
            counter += 1
            if counter >= 100:
                continue_extend_path = False
                break

            last_link = link_dict.get(cur_path[-1], {})
            if len(last_link) < 1:
                continue_extend_path = False
                break

            last_enode = last_link.get("enode_id", -1)
            if connect_dict.get(last_enode, {}):
                candidates_dict = connect_dict[last_enode]
                if len(candidates_dict) < 1:
                    continue_extend_path = False
                    break
                best_next_id = self.choose_next_sd(candidates_dict, navi_gt, link_dict)
                if best_next_id in cur_path:
                    continue_extend_path = False
                    break
                if best_next_id == -1:
                    continue_extend_path = False
                    break
                if cur_path[-1] != best_next_id:
                    cur_path.append(best_next_id)
                else:
                    continue_extend_path = False
                    break
            else:
                continue_extend_path = False
                break

        return cur_path

    def best_match_sd_to_navigation_prior(
        self, navi, best_path, connect_dict, link_dict
    ):
        # navi_sd_ls = LineString(navi)
        shooting_line = LineString(np.array([[0, 0], [self.pc_range[3], 0]]))
        best_path_ls = []
        for id in best_path:
            points = link_dict.get(id, {}).get("cut_points", [])
            if len(points) < 1:
                continue
            best_path_ls.append(LineString(points[0]))

        intersection_index = -1  # the first segment intersecting with shooting_line
        for i, sd_ls in enumerate(best_path_ls):
            if sd_ls.intersects(shooting_line):
                intersection_index = i
                break

        if intersection_index != -1:
            # there's intersection with shooting_line
            intersected_sd = best_path_ls[intersection_index]
            p_intersec = shooting_line.intersection(intersected_sd)
            first_half = np.array(
                self.cut_linestring_at_p(
                    shooting_line, p_intersec, first_half=True
                ).coords
            )
            second_half = np.array(
                self.cut_linestring_at_p(
                    intersected_sd, p_intersec, first_half=False
                ).coords
            )
            map_prior = np.concatenate((first_half, second_half), axis=0)
            for i in range(intersection_index + 1, len(best_path_ls)):
                map_prior = np.concatenate(
                    (map_prior, np.array(best_path_ls[i].coords)), axis=0
                )
            combined_halves = self.extend_sd_to_BEV_range(map_prior, self.pc_range)
            sampled_combined_halves = self.sample_pts_from_line(combined_halves)

        else:
            # no intersection with shooting_line
            # shift the path to pass (0,0)
            original_laster = np.array(best_path_ls[0].coords)
            for i in range(1, len(best_path_ls)):
                original_laster = np.concatenate(
                    (original_laster, np.array(best_path_ls[i].coords)), axis=0
                )

            original_laster_ls = LineString(original_laster)
            # if this line passed x=0, shift this whole line
            y_shifter = 0.0
            x_intersected = False
            y_axis = LineString(
                np.array([[0, self.pc_range[4]], [0, self.pc_range[1]]])
            )
            if original_laster_ls.intersects(y_axis):
                y_intersection = original_laster_ls.intersection(y_axis)
                if isinstance(y_intersection, Point):
                    y_shifter = y_intersection.y
                    x_intersected = True
                    new_laster = np.array(
                        self.cut_linestring_at_p(
                            original_laster_ls, y_intersection, False
                        ).coords
                    )
                    new_laster[:, 1] -= y_shifter
                    combined_halves = new_laster

            # else, shift the line according to y of the first point in second half
            if not x_intersected:
                y_shifter = original_laster[0][1]
                original_laster[:, 1] -= y_shifter
                combined_halves = np.concatenate(
                    (
                        np.array(
                            [[0, 0], [original_laster[0][0], original_laster[0][1]]]
                        ),
                        original_laster,
                    ),
                    axis=0,
                )
            combined_halves = self.extend_sd_to_BEV_range(
                combined_halves, self.pc_range
            )
            sampled_combined_halves = self.sample_pts_from_line(combined_halves)
        return sampled_combined_halves

    def patch_cut_indi(self, sd_geom):
        patch_box = box(
            self.pc_range[0], self.pc_range[1], self.pc_range[3], self.pc_range[4]
        )
        linestring = LineString(sd_geom)
        if linestring.is_empty:
            return False, sd_geom

        clipped = linestring.intersection(patch_box)
        if clipped.is_empty:
            is_start_inside = patch_box.contains(Point(linestring.coords[0]))
            is_end_inside = patch_box.contains(Point(linestring.coords[-1]))
            if is_start_inside and is_end_inside:
                return True, sd_geom
            else:
                return False, sd_geom
        if "Multi" in clipped.geom_type:
            clipped = max(clipped.geoms, key=lambda part: part.length)
        if clipped.length > 0:
            return True, np.array(clipped.coords)
        return False, sd_geom

    def sd_preprocess(self, results):
        # Build-up node-link relations, patch_cut the candidates
        # Return {node_in:{node_out:[link_ids]}}, {link_id:link_content}
        angle_thr = 90
        connect_dict = {}
        link_dict = {}

        navi_gt = results["map_gt"].get("maptr_navigation_gt", [])
        sd_prior_all = results.get("sd_v2_navigation_information", {})
        if sd_prior_all.get("data"):
            for sd_prior in sd_prior_all.get("data", []):
                in_patch, cut_sd = self.patch_cut_indi(sd_prior["geom"][:, :2])
                if in_patch:
                    if (
                        (len(navi_gt) > 0)
                        and (
                            (sd_prior["direction"] == 1)
                            or (sd_prior["road_class"] in [0, 6, 7, 8])
                        )
                        and (sd_prior["multiply_digitized"] in [0, 1])
                    ):
                        same_order, ordered_cut_sd = self.find_correct_order(
                            cut_sd, navi_gt[0]["points"][0]
                        )
                        if self.navi_close_angle(
                            ordered_cut_sd, navi_gt[0]["points"][0], angle_thr
                        ):
                            connect_dict, link_dict = self.update_sd_dicts(
                                sd_prior,
                                same_order,
                                ordered_cut_sd,
                                connect_dict,
                                link_dict,
                            )
                    elif sd_prior["direction"] == 3:
                        if (len(navi_gt) > 0) and self.navi_close_angle(
                            cut_sd[::-1], navi_gt[0]["points"][0], angle_thr
                        ):
                            connect_dict, link_dict = self.update_sd_dicts(
                                sd_prior, False, cut_sd[::-1], connect_dict, link_dict
                            )
                        elif (len(navi_gt) > 0) and self.navi_close_angle(
                            cut_sd, navi_gt[0]["points"][0], angle_thr
                        ):
                            connect_dict, link_dict = self.update_sd_dicts(
                                sd_prior, True, cut_sd, connect_dict, link_dict
                            )
                    else:
                        if (len(navi_gt) > 0) and self.navi_close_angle(
                            cut_sd, navi_gt[0]["points"][0], angle_thr
                        ):
                            connect_dict, link_dict = self.update_sd_dicts(
                                sd_prior, True, cut_sd, connect_dict, link_dict
                            )
            return connect_dict, link_dict
        return {}, {}

    def find_closest_sd(self, link_dict, navi_gt, is_first_seg):
        # Find the closest link from current location
        p = Point((0, 0))
        bounding_box = box(0, self.pc_range[1], self.pc_range[3], self.pc_range[4])
        if len(navi_gt) > 0:
            if is_first_seg:
                p_i = LineString(navi_gt).intersection(
                    LineString(np.array([[0, self.pc_range[1]], [0, self.pc_range[4]]]))
                )
                if isinstance(p_i, Point):
                    p = p_i
            else:
                p = Point(navi_gt[-1])
        min_dist = 10000.0
        best_linkid = -1
        for link_id in link_dict.keys():
            link_ls = LineString(link_dict[link_id]["cut_points"][0])
            clipped_link_ls = link_ls.intersection(bounding_box)
            if clipped_link_ls.is_empty:
                continue
            if clipped_link_ls.geom_type == "MultiLineString":
                closest_line_ls = min(
                    clipped_link_ls.geoms,
                    key=lambda line: nearest_points(line, p)[0].distance(p),
                )
            elif clipped_link_ls.geom_type == "LineString":
                closest_line_ls = clipped_link_ls
            else:
                continue
            cur_dist = closest_line_ls.distance(p)
            _, closest_seg = self.sd_measurement(np.array(link_ls.coords), navi_gt)
            curr_navi_seg = [point for point in navi_gt if point[0] > 0]
            if len(curr_navi_seg) < 2 or len(link_dict[link_id]["cut_points"][0]) < 2:
                continue
            if is_first_seg:
                curr_navi_seg = np.array(curr_navi_seg[:2])
                curr_sd_geom = link_dict[link_id]["cut_points"][0][:2]
            else:
                curr_navi_seg = np.array(curr_navi_seg[-2:])
                curr_sd_geom = link_dict[link_id]["cut_points"][0][-2:]
            if closest_seg is not None and self.approx_same_angle(
                curr_navi_seg, curr_sd_geom, 30
            ):
                if not self.check_sd_navi_same_direction(
                    np.array(closest_seg.coords),
                    link_dict[link_id]["cut_points"][0],
                    self.thres_angle,
                ):
                    continue
                if cur_dist < min_dist:
                    min_dist = cur_dist
                    best_linkid = link_id
        if min_dist == 10000.0:
            return False, best_linkid
        return True, best_linkid

    def find_all_paths_bfs(self, links, start_link_id, end_link_id):
        queue = deque([(start_link_id, [start_link_id])])
        all_link_ids = [start_link_id]
        all_end_link_ids = set()
        visited = {start_link_id}
        if end_link_id != -1 and start_link_id == end_link_id:
            return all_link_ids
        counter = 0
        while queue:
            counter += 1
            if counter >= 100:
                break
            current_link_id, current_path = queue.popleft()
            if current_link_id == -1:
                return all_link_ids
            enode_id = links[current_link_id].get("enode_id", -1)

            next_links = [
                link_id
                for link_id in links
                if links[link_id]["snode_id"] == enode_id
                and link_id not in visited
                and link_id not in current_path
            ]

            for next_link_id in next_links:
                if next_link_id not in all_link_ids:
                    if end_link_id == -1:
                        all_link_ids.append(next_link_id)
                    else:
                        if next_link_id == end_link_id:
                            all_end_link_ids.update(set(current_path + [next_link_id]))

                new_path = current_path + [next_link_id]
                queue.append((next_link_id, new_path))
                visited.add(next_link_id)
        if end_link_id != -1:
            return list(all_end_link_ids)
        return all_link_ids

    def select_used_link(self, connect_dict, link_dict, start_link_id, end_link_id=-1):
        if len(link_dict) == 0:
            return {}, {}
        all_used_ids = self.find_all_paths_bfs(link_dict, start_link_id, end_link_id)
        select_connect_dict = {}
        select_link_dict = {}
        # select connects
        for snode_id, sub_dict in connect_dict.items():
            sub_connect_dict = {}
            for enode_id, link_id in sub_dict.items():
                if link_id[0] in all_used_ids:
                    sub_connect_dict[enode_id] = link_id
            if len(sub_connect_dict) > 0:
                select_connect_dict[snode_id] = sub_connect_dict
        # select links
        select_link_dict = {
            key: value for key, value in link_dict.items() if key in all_used_ids
        }
        return select_connect_dict, select_link_dict

    def mock_sd_to_navigation_prior(self, navigation_prior):
        points = []
        s_x = 0
        s_y = 0
        points.append([s_x, s_y])
        turning_start = max(navigation_prior["start"], 0.0)
        if turning_start >= 150:
            turning_start = self.pc_range[3]
            navigation_prior["angle"] = 0
        elif turning_start >= self.pc_range[3] and turning_start < 150:
            turning_start = self.pc_range[3]
        angle = self.angle_protect(navigation_prior["angle"])
        if self.use_ego_angle:
            ego_angle = navigation_prior.get("ego_angle", 0.0)
        else:
            ego_angle = 0
        distur_angle = self.angle_adjust(angle, ego_angle)
        if angle > 0:
            # turning left or right
            end_point = self.before_turning(distur_angle, angle, turning_start, s_y)
        else:
            # straight
            end_point = self.calculate_endpoint(distur_angle, turning_start, s_y)

        points.extend(end_point)
        sample_points = self.sample_pts_from_line(points)
        return sample_points

    def navigation_angel_bin(self, angle, num_bins=12):
        angle = angle % 360
        if angle < 15 or angle >= 345:
            angle_bin = 0
        else:
            adjusted_angle = angle - 15
            bin_size = 360 // num_bins
            angle_bin = int(adjusted_angle // bin_size) + 1
        return np.array([angle_bin]).astype(int)

    def crop_at_navi_boundaries(self, uncropped_pts):
        pre_ls = LineString(uncropped_pts)
        intersection = pre_ls.intersection(self.navi_patchbox)
        if intersection.is_empty:
            return False, None
        else:
            if intersection.geom_type == "LineString":
                return True, intersection
            elif intersection.geom_type == "MultiLineString":
                origin_point = Point(0.0, 0.0)
                near_part = min(
                    (line for line in intersection.geoms),
                    key=lambda line: line.distance(origin_point),
                )
                return True, near_part
            else:
                return False, None

    def shift_n_extend(self, cropped_ls):
        tmp_output = np.array(cropped_ls)
        # if this line passed x=0, shift this whole line
        y_shifter = 0.0
        x_intersected = False
        y_axis = LineString(np.array([[0, self.pc_range[4]], [0, self.pc_range[1]]]))
        if cropped_ls.intersects(y_axis):
            y_intersection = cropped_ls.intersection(y_axis)
            if isinstance(y_intersection, Point):
                y_shifter = y_intersection.y
                x_intersected = True
                new_laster = np.array(
                    self.cut_linestring_at_p(cropped_ls, y_intersection, False).coords
                )
                new_laster[:, 1] -= y_shifter
                tmp_output = new_laster

        # else, shift the line according to y of the first point in second half
        if not x_intersected:
            cropped_pts =np.array(cropped_ls.coords)
            y_shifter = cropped_pts[0][1]
            cropped_pts[:, 1] -= y_shifter
            tmp_output = np.concatenate(
                (
                    np.array([[0, 0], [cropped_pts[0][0], cropped_pts[0][1]]]),
                    cropped_pts,
                ),
                axis=0,
            )
        tmp_output = self.extend_sd_to_BEV_range(tmp_output, self.pc_range)
        return self.sample_pts_from_line(tmp_output)

    def __call__(self, results):
        mock_sd_prior = results["navigation_info"]
        if not results.get("navi_best_path"):
            results["navi_best_path"] = []

        # generate navigation angle bin from mock sd
        angle_junction = mock_sd_prior["angle"]
        results["navigation_angle_bin"] = self.navigation_angel_bin(angle_junction)

        # Update 12.24.2024 Use targetv2 as navi
        target_v2_navi_path = results["sd_v2_navigation_information"][
            "target_v2_navi_path"
        ]
        if len(target_v2_navi_path) > 0:
            target_list = []
            for t in target_v2_navi_path:
                if "original_vcs_pts" in t.keys():
                    target_list.append(t["original_vcs_pts"][:, :2])

            if len(target_list) > 0:
                targets_vcs_pts = np.concatenate(target_list, axis=0)
                # Crop at the boundary, and resample the targets
                crop_flag, cropped_ls = self.crop_at_navi_boundaries(targets_vcs_pts)
                if crop_flag:
                    results["diffusion_navi"] = self.shift_n_extend(cropped_ls)
                    return results

        # build-up node-link relations, patch_cut the candidates
        connect_dict, link_dict = self.sd_preprocess(results)

        navi_gt = results["map_gt"].get("maptr_navigation_gt", [])
        if len(navi_gt) < 1:
            results["diffusion_navi"] = self.mock_sd_to_navigation_prior(mock_sd_prior)
            return results
        # find the closest link from current location
        found_start_id, closest_start_id = self.find_closest_sd(
            link_dict, navi_gt[0]["points"][0], is_first_seg=True
        )
        # find all used links
        if found_start_id:
            connect_dict, link_dict = self.select_used_link(
                connect_dict, link_dict, closest_start_id
            )
        # update link dict
        found_end_id, closest_end_id = self.find_closest_sd(
            link_dict, navi_gt[0]["points"][0], is_first_seg=False
        )
        # find all used links
        if found_end_id:
            connect_dict, link_dict = self.select_used_link(
                connect_dict, link_dict, closest_start_id, closest_end_id
            )

        if not found_start_id or not found_end_id:
            results["diffusion_navi"] = self.mock_sd_to_navigation_prior(mock_sd_prior)
            return results

        # greedily loop over the graph to find the best matched path
        # best_path: list of link_id in order
        best_path = self.find_best_sd_path(
            closest_start_id, navi_gt[0]["points"][0], connect_dict, link_dict
        )
        results["navi_best_path"] = best_path
        if len(best_path) > 0:
            # intersect (or connect) sd prior using best_path
            results["diffusion_navi"] = self.best_match_sd_to_navigation_prior(
                navi_gt[0]["points"][0], best_path, connect_dict, link_dict
            )
            # check sd prior direction
            if results[
                "diffusion_navi"
            ] is not None and self.check_sd_navi_same_direction(
                results["diffusion_navi"],
                navi_gt[0]["points"][0],
                self.thres_angle,
            ):
                results["diffusion_navi"] = results["diffusion_navi"].astype(np.float32)
                return results

        results["diffusion_navi"] = self.mock_sd_to_navigation_prior(mock_sd_prior)
        return results

class LoadPointsFromLiautoFiles(object):
    def __init__(
        self,
        coord_type,
        pcd_root=None,
        load_dim=6,
        use_dim=[0, 1, 2],
        shift_height=False,
        file_client_args=dict(backend="disk"),
    ):
        self.shift_height = shift_height
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert (
            max(use_dim) < load_dim
        ), f"Expect all used dimensions < {load_dim}, got {use_dim}"
        assert coord_type in ["CAMERA", "LIDAR", "DEPTH"]

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.pcd_root = pcd_root

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        mmcv.check_file_exist(pts_filename)
        try:
            pc = pypcd.PointCloud.from_path(pts_filename)
        except Exception as e:
            print("pts file loading error", e)
            invalid_data = pts_filename
            pts_name = pts_filename.replace("/", "_")
            save_path = os.path.join("/lpai/output/data", f"{pts_name}.txt")
            with open(save_path, "w", encoding="utf-8") as f:
                f.writelines(pts_name)
            print("invalid pts filename: ", invalid_data)
            return np.zeros((1, 3)), np.zeros((1, 1))
        x = np.expand_dims(pc.pc_data["x"], axis=1)
        y = np.expand_dims(pc.pc_data["y"], axis=1)
        z = np.expand_dims(pc.pc_data["z"], axis=1)
        intensity = np.expand_dims(pc.pc_data["intensity"], axis=1)
        points = np.concatenate((x, y, z), axis=1)
        return points, intensity

    def __call__(self, results):
        pts_filename = results["at128_pts_filename"]
        pts_path = (
            os.path.join(self.pcd_root, pts_filename) if self.pcd_root else pts_filename
        )
        if pts_path.endswith(".bin"):
            points_with_intensity = torch.from_numpy(
                np.fromfile(pts_path, dtype=np.float32).reshape(-1, 4)
            )
            results["points"] = BasePoints(points_with_intensity, points_dim=4)
            return results
        points, intensity = self._load_points(pts_path)
        # translate
        rotation = results["lidar2ego_rotation_at128"]
        translate = results["lidar2ego_translation_at128"]
        lidar_rotation_mat = Quaternion(rotation).rotation_matrix
        lidar_rots = np.array(lidar_rotation_mat).reshape((3, 3))
        vcs_T_lidar = np.eye(4)
        vcs_T_lidar[:3, :3] = lidar_rots
        vcs_T_lidar[:3, 3] = translate
        points_4 = np.hstack([points, np.ones((points.shape[0], 1))])
        points_vcs_4 = points_4.dot(vcs_T_lidar.T)
        points_vcs_4 = points_vcs_4[:, :3] / points_vcs_4[:, 3:]
        points = points_vcs_4[:, :3]
        intensity = np.ascontiguousarray(intensity)
        points_with_intensity = torch.concat(
            (torch.from_numpy(points), torch.from_numpy(intensity)), axis=-1
        )
        results["points"] = BasePoints(points_with_intensity, points_dim=4)
        return results

    def __repr__(self) -> str:
        repr_str = super().__repr__()
        return repr_str