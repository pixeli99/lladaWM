from ..observation import Observation, Frame
from typing import List, Dict

import cv2
import re
import numpy as np
import os
from shapely.geometry import Point, Polygon
from PIL import Image, ImageDraw, ImageFont
import math
import copy
from itertools import groupby
import matplotlib.image as mpimg
import asset


class Visualizer:
    def __init__(self):
        self.output_folder_root = "/lpai/output/score_visual/test2"

        self.bev_imgx_low_ = -18.0
        self.bev_imgx_high_ = 118.8
        self.bev_imgy_low_ = -24.0
        self.bev_imgy_high_ = 24.0

        self.bev_w_ = 0
        self.bev_h_ = 0
        self.pc_range = [-16.2, -24.0, -5.0, 119.8, 24.0, 3.0]

        self.Color_map = {
            "LaneDivider": (0, 127, 255),
            "Boundary": (255, 255, 0),
            "Stopline": (0, 0, 255),
            "Crosswalk": (0, 255, 0),
            "Polygon": (255, 0, 255),
            "LaneCenter": (255, 48, 155),
            "Junction": (255, 0, 0),
            "SpeedBump": (147, 20, 255),
            "VirtualLane": (100, 100, 100),
            "RoadMark": (128, 128, 0),
        }

        self.Toll_Station_Color_map = {
            "Man": (0, 0, 255),
            "Unknow": (0, 0, 0),
            "No": (147, 20, 255),
            "ETC/Man": (0, 255, 0),
            "ETC": (0, 255, 0),
        }

    def visualize_frame(self, frame: Frame, score_dict: dict, set_output_dir=None):
        meter_h_ = 400.0
        meter_w_ = 140.0
        all_h_ = 1440
        all_w_ = int(all_h_ * meter_w_ / meter_h_)
        self.bev_h_ = int(all_h_ * 0.8)
        self.bev_w_ = int(all_h_ * meter_w_ / meter_h_)

        vcsimage = np.ones((self.bev_h_, self.bev_w_, 3), dtype="uint8") * 255
        self.PlotGridAndEgo(vcsimage)

        static_map = frame.static_map
        self.ShowBev(vcsimage, static_map)

        navi_traj = frame.sd_v2.get("coordinates_vcs", [])
        self.ShowNaviTraj(vcsimage, navi_traj)

        ego_future_traj = frame.ego_state.get("sdv_future_traj", [])
        self.ShowEgoFutureTraj(vcsimage, ego_future_traj)

        obstacles = frame.obs
        self.ShowObstables(vcsimage, obstacles)

        ego_obstacle = frame.ego_obs
        self.ShowEGOObstables(vcsimage, ego_obstacle)

        future_1s_obstacles = frame.future_obs
        self.ShowFutureObstables(vcsimage, future_1s_obstacles)
        self.ShowFutureEGOObstables(vcsimage, ego_obstacle)

        self.ShowScore(vcsimage, score_dict)

        image_path = frame.image_path
        vcsimage = self.CamImageStitching(vcsimage, image_path)

        if set_output_dir is None:
            output_dir = os.path.join(self.output_folder_root, str(frame.scene_token))
        else:
            output_dir = os.path.join(set_output_dir, str(frame.scene_token))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_pic_path = os.path.join(output_dir, str(frame.timestamp) + ".jpg")
        cv2.imwrite(output_pic_path, vcsimage)

    def CamImageStitching(self, vcsimage, image_path):
        jwt_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjdWlzb25nQGxpeGlhbmcuY29tIiwiaXNzIjoibHBhaSIsImlhdCI6MTcwNDE2MTc5OCwianRpIjoiMmU3ODQ5ODgtNTc3Ni00ODk3LTliOTUtOWM4ZDkwMzg4ZTRiIn0.CijUMFHBtc4gdTdUTF4TdUH_LCmhJjLe8xyUKFTX7_w"
        config = asset.config(
            env="prod",
            jwt_token=jwt_token,
        )

        if image_path["CAM_FRONT"]["data_path"]:
            dataset_name = image_path["CAM_FRONT"]["data_path"].split("/")[3]
            dataset_version = image_path["CAM_FRONT"]["data_path"].split("/")[4]
            dataset_resource = (
                "datasets/" + dataset_name + "/versions/" + dataset_version
            )
            dataset = asset.resource(dataset_resource, config=config)
        else:
            return vcsimage

        top_img_path = image_path["CAM_FRONT"]["data_s3_path"]
        bottom_img_path = image_path["CAM_BACK"]["data_s3_path"]
        top_left_img_path = image_path["CAM_FRONT_LEFT"]["data_s3_path"]
        bottom_left_img_path = image_path["CAM_BACK_LEFT"]["data_s3_path"]
        top_right_img_path = image_path["CAM_FRONT_RIGHT"]["data_s3_path"]
        bottom_right_img_path = image_path["CAM_BACK_RIGHT"]["data_s3_path"]

        def create_black_image(width, height):
            return np.zeros((height, width, 3), dtype="uint8")

        def load_or_create_black_image(path, width, height):
            if path:
                try:
                    image_data = dataset.get_file(s3_path=path).read()
                    image_array = np.frombuffer(image_data, np.uint8)
                    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                    return cv2.resize(img, (width, height))
                except:
                    return create_black_image(width, height)
            else:
                return create_black_image(width, height)

        cam_img_width = 3840
        cam_img_height = 2160
        img_width = vcsimage.shape[1]
        img_height = (img_width * cam_img_height) // cam_img_width

        top, bottom, left, right = img_height, img_height, img_width, img_width
        vcsimage = cv2.copyMakeBorder(
            vcsimage, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

        top_image = load_or_create_black_image(top_img_path, img_width, img_height)
        bottom_image = load_or_create_black_image(
            bottom_img_path, img_width, img_height
        )
        top_left_image = load_or_create_black_image(
            top_left_img_path, img_width, img_height
        )
        bottom_left_image = load_or_create_black_image(
            bottom_left_img_path, img_width, img_height
        )
        top_right_image = load_or_create_black_image(
            top_right_img_path, img_width, img_height
        )
        bottom_right_image = load_or_create_black_image(
            bottom_right_img_path, img_width, img_height
        )

        vcsimage[0:img_height, (img_width + 1) : (2 * img_width + 1)] = top_image
        vcsimage[-img_height:, (img_width + 1) : (2 * img_width + 1)] = bottom_image
        vcsimage[(img_height + 1) : (2 * img_height + 1), 0:img_width] = top_left_image
        vcsimage[
            (-2 * img_height - 1) : (-img_height - 1), 0:img_width
        ] = bottom_left_image
        vcsimage[(img_height + 1) : (2 * img_height + 1), -img_width:] = top_right_image
        vcsimage[
            (-2 * img_height - 1) : (-img_height - 1), -img_width:
        ] = bottom_right_image

        return vcsimage

    def ShowScore(self, vcsimage, score_dict):
        score_str = "score: \n"
        for key, value in score_dict.items():
            score_str += str(key) + "=" + str(value) + ";\n"
        text = score_str
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        font_color = (150, 150, 150)
        line_height = (
            cv2.getTextSize("Sample", font, font_scale, font_thickness)[0][1] + 5
        )
        x = 10
        y = 20
        for line in text.split("\n"):
            cv2.putText(
                vcsimage, line, (x, y), font, font_scale, font_color, font_thickness
            )
            y += line_height

    def getpix_x(self, pt_y):
        return int(
            (-self.bev_imgy_low_ - pt_y)
            * ((self.bev_w_) / (self.bev_imgy_high_ - self.bev_imgy_low_))
        )

    def getpix_y(self, pt_x):
        return int(
            self.bev_h_
            - (-self.bev_imgx_low_ + pt_x)
            * ((self.bev_h_) / (self.bev_imgx_high_ - self.bev_imgx_low_))
        )

    def PlotGridAndEgo(self, vcsimage):
        color = (200, 200, 200)
        for x in [0.0, 20.0, 40.0, 60.0, 80.0, 100.0]:
            v = self.getpix_y(x)
            cv2.line(
                vcsimage,
                (self.getpix_x(self.bev_imgy_low_), v),
                (self.getpix_x(self.bev_imgy_high_), v),
                color,
                1,
            )
            cv2.putText(
                vcsimage,
                str(int(x)),
                (0, v + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

        for y in [-20.0, -10.0, 10.0, 20.0]:
            u = self.getpix_x(y)
            cv2.line(
                vcsimage,
                (u, self.getpix_y(self.bev_imgx_low_)),
                (u, self.getpix_y(self.bev_imgx_high_)),
                color,
                1,
            )
            cv2.putText(
                vcsimage,
                str(int(-y)),
                (u - 10, 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

        # car_w = 1.998
        # car_h = 5.128
        # car_pts = [(car_h / 2.0, car_w / 2.0), (car_h / 2.0, -car_w / 2.0),
        #         (-car_h / 2.0, -car_w / 2.0), (-car_h / 2.0, car_w / 2.0)]
        # self.BevdrawBox(vcsimage, car_pts, (0, 0, 0))

    def Bevdrawline(self, vcsimage, pts, color, width=2):
        if len(pts) == 0:
            return
        cv2.circle(
            vcsimage, (self.getpix_x(pts[0][1]), self.getpix_y(pts[0][0])), 5, color, -1
        )
        for i in range(1, len(pts)):
            last_pt = (self.getpix_x(pts[i - 1][1]), self.getpix_y(pts[i - 1][0]))
            next_pt = (self.getpix_x(pts[i][1]), self.getpix_y(pts[i][0]))
            cv2.line(vcsimage, last_pt, next_pt, color, width, cv2.LINE_AA)
            if i == len(pts) // 2:
                length = math.hypot(next_pt[0] - last_pt[0], next_pt[1] - last_pt[1])
                if length < 0.01:
                    continue
                tipLength = 10.0 / length
                cv2.arrowedLine(
                    vcsimage, last_pt, next_pt, color, 2, cv2.LINE_AA, 0, tipLength
                )
        cv2.circle(
            vcsimage,
            (self.getpix_x(pts[-1][1]), self.getpix_y(pts[-1][0])),
            5,
            color,
            -1,
        )

    def BevdrawBox(self, vcsimage, pts, color):
        if len(pts) == 0:
            return
        for i in range(1, len(pts)):
            last_pt = (self.getpix_x(pts[i - 1][1]), self.getpix_y(pts[i - 1][0]))
            next_pt = (self.getpix_x(pts[i][1]), self.getpix_y(pts[i][0]))
            cv2.line(vcsimage, last_pt, next_pt, color, 2, cv2.LINE_AA)
        last_pt = (self.getpix_x(pts[-1][1]), self.getpix_y(pts[-1][0]))
        next_pt = (self.getpix_x(pts[0][1]), self.getpix_y(pts[0][0]))
        cv2.line(vcsimage, last_pt, next_pt, color, 2, cv2.LINE_AA)

    def DrawDynamicObstacle(
        self, vcsimage, position, heading, length, width, color=(128, 128, 128)
    ):
        x, y = position
        rot_matrix = np.array(
            [[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]]
        )
        half_length = length / 2.0
        half_width = width / 2.0
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
            global_x = x + rotated[0]
            global_y = y + rotated[1]
            pixel = (self.getpix_x(global_y), self.getpix_y(global_x))
            corners.append(pixel)

        for i in range(4):
            cv2.line(vcsimage, corners[i], corners[(i + 1) % 4], color, 1, cv2.LINE_AA)
        cv2.circle(vcsimage, (self.getpix_x(y), self.getpix_y(x)), 1, (0, 0, 0), -1)

    def DrawEgoObstacle(
        self, vcsimage, position, heading, length, width, color=(0, 0, 0)
    ):
        x, y = position
        rot_matrix = np.array(
            [[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]]
        )
        half_length = length / 2.0
        half_width = width / 2.0
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
            global_x = x + rotated[0]
            global_y = y + rotated[1]
            pixel = (self.getpix_x(global_y), self.getpix_y(global_x))
            corners.append(pixel)

        for i in range(4):
            cv2.line(vcsimage, corners[i], corners[(i + 1) % 4], color, 2, cv2.LINE_AA)
        cv2.circle(vcsimage, (self.getpix_x(y), self.getpix_y(x)), 2, (0, 0, 0), -1)

    def ShowLineExplane(self, vcsimage):
        legend = {
            "LaneCenter": (153, 0, 76),
            "LaneCenter_Virtual": (100, 100, 100),
            "RoadBoundary": (255, 255, 0),
            "Stopline": (0, 0, 255),
            "LaneDivider_Solid": (7, 143, 14),
            "LaneDivider_Other": (0, 127, 255),
            "Junction": (255, 0, 0),
            "Navigation_Line": (23, 255, 248),
            "Ego_Future_Traj": (255, 132, 248),
        }
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.3
        font_thickness = 1
        line_length = 30
        line_thickness = 2

        y_offset = vcsimage.shape[0] - 10 * len(legend) - 5
        x_text_offset = vcsimage.shape[1] - 150

        for i, (text, color) in enumerate(legend.items()):
            y = y_offset + i * 10
            cv2.line(
                vcsimage,
                (x_text_offset, y),
                (x_text_offset + line_length, y),
                color,
                line_thickness,
            )
            cv2.putText(
                vcsimage,
                text,
                (x_text_offset + line_length + 10, y + 5),
                font,
                font_scale,
                color,
                font_thickness,
            )

    def ShowBev(self, vcsimage, static_map):
        for j_lanes in static_map["lane_center"]:
            if len(j_lanes["points"]) < 2:
                continue
            lane = [[pt[axis] for axis in [0, 1]] for pt in j_lanes["points"]]
            color = self.Color_map["LaneCenter"]
            if j_lanes["is_virutal"]:
                color = self.Color_map["VirtualLane"]
            else:
                color = (153, 0, 76)  # purple

            self.Bevdrawline(vcsimage, lane, color)

            pt0 = lane[0]
            pt = lane[len(lane) // 2]
            if not j_lanes["is_virutal"]:
                cv2.putText(
                    vcsimage,
                    str(j_lanes["lane_type"]),
                    (self.getpix_x(pt0[1]) + 5, self.getpix_y(pt0[0]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                )

        for j_lanes in static_map["road_boundary"]:
            lane = [[pt[axis] for axis in [0, 1]] for pt in j_lanes["points"]]
            color = self.Color_map["Boundary"]
            self.Bevdrawline(vcsimage, lane, color)

        for j_lanes in static_map["stopline"]:
            lane = [[pt[axis] for axis in [0, 1]] for pt in j_lanes["points"]]
            color = self.Color_map["Stopline"]
            self.Bevdrawline(vcsimage, lane, color)

        for j_lanes in static_map["lane_divider"]:
            if j_lanes["points"][0][0] == "NaN":
                continue
            solid_divide_lanes = [
                [j_lanes["points"][i] for i in indices]
                for value, indices in groupby(
                    range(len(j_lanes["type"])), key=lambda i: j_lanes["type"][i] == 1
                )
                if value
            ]
            other_divide_lanes = [
                [j_lanes["points"][i] for i in indices]
                for value, indices in groupby(
                    range(len(j_lanes["type"])), key=lambda i: j_lanes["type"][i] != 1
                )
                if value
            ]
            for divide_lane in solid_divide_lanes:
                if len(divide_lane) < 2:
                    continue
                lane = [[pt[axis] for axis in [0, 1]] for pt in divide_lane]
                color = (7, 143, 14)
                self.Bevdrawline(vcsimage, lane, color)
            for divide_lane in other_divide_lanes:
                if len(divide_lane) < 2:
                    continue
                lane = [[pt[axis] for axis in [0, 1]] for pt in divide_lane]
                color = self.Color_map["LaneDivider"]
                self.Bevdrawline(vcsimage, lane, color)

        #   for j_lanes in j_frame['polygons']:
        #     lane = [[pt[axis] for axis in ['x', 'y']] for pt in j_lanes['points']]
        #     color = Color_map["Crosswalk"]
        #     Bevdrawline(vcsimage, lane, color)

        for j_lanes in static_map["junction"]:
            lane = [[pt[axis] for axis in [0, 1]] for pt in j_lanes["points"]]
            color = self.Color_map["Junction"]
            self.Bevdrawline(vcsimage, lane, color)

        self.ShowLineExplane(vcsimage)

    def ShowNaviTraj(self, vcsimage, navi_traj):
        if len(navi_traj) < 2:
            return None
        navi_future_pts = [[item[0], item[1]] for item in navi_traj]
        color = (23, 255, 248)  # yellow
        self.Bevdrawline(vcsimage, navi_future_pts, color)

    def ShowEgoFutureTraj(self, vcsimage, ego_future_traj):
        if len(ego_future_traj) < 2:
            return None
        ego_future_traj_pts = [
            [item["position"]["x"], item["position"]["y"]] for item in ego_future_traj
        ]
        color = (255, 132, 248)  # purple
        self.Bevdrawline(vcsimage, ego_future_traj_pts, color)

    def ShowObstables(self, vcsimage, obstacles):
        for obj in obstacles:
            self.DrawDynamicObstacle(
                vcsimage=vcsimage,
                position=(obj.x, obj.y),
                heading=obj.yaw,
                length=obj.l,
                width=obj.w,
            )

    def ShowFutureObstables(self, vcsimage, future_1s_obstacles):
        for future_obs_info in future_1s_obstacles:
            for obj in future_obs_info["obs_list"]:
                self.DrawDynamicObstacle(
                    vcsimage=vcsimage,
                    position=(obj.x, obj.y),
                    heading=obj.yaw,
                    length=obj.l,
                    width=obj.w,
                    color=(210, 210, 210),
                )

    def ShowEGOObstables(self, vcsimage, obstacle):
        # draw current ego
        self.DrawEgoObstacle(
            vcsimage=vcsimage,
            position=(obstacle.x, obstacle.y),
            heading=obstacle.yaw,
            length=obstacle.l,
            width=obstacle.w,
        )

    def ShowFutureEGOObstables(self, vcsimage, obstacle):
        # draw future 1s ego
        for i in range(10):
            self.DrawEgoObstacle(
                vcsimage=vcsimage,
                position=(
                    obstacle.x + obstacle.vx * (i + 1) * 0.1,
                    obstacle.y + obstacle.vy * (i + 1) * 0.1,
                ),
                heading=obstacle.yaw,
                length=obstacle.l,
                width=obstacle.w,
                color=(100, 100, 100),
            )
