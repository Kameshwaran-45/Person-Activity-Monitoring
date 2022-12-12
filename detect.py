import sys
import json
import torch
import cv2
import math

import numpy as np

from torchvision import transforms as T

from pose.models import get_pose_model
from pose.utils.boxes import letterbox, non_max_suppression, xyxy2xywh
from pose.utils.decode import get_final_preds, get_simdr_final_preds
from pose.utils.utils import setup_cudnn, get_affine_transform

sys.path.insert(0, 'yolov5')
from yolov5.utils.general import (non_max_suppression, scale_coords, cv2,xyxy2xywh)
from yolov5.models.experimental import attempt_load

sys.path.insert(0, 'strong_sort')
from strong_sort import StrongSORT


with open("config/config.json") as json_file:
    config_file = json.load(json_file)


def getConfigParam(strKey, dict):
    try:
        if strKey in dict.keys():
            return dict[strKey]
        else:
            return ''
    except:
        print(str(strKey) + "not available")


class Pose:
    def __init__(self,
                 det_model,
                 pose_model,
                 img_size=640,
                 conf_thres=0.25,
                 iou_thres=0.45,
                 ) -> None:
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.det_model = attempt_load(det_model, device=self.device)
        self.det_model = self.det_model.to(self.device)

        self.model_name = pose_model
        self.pose_model = get_pose_model(pose_model)
        self.pose_model.load_state_dict(torch.load(pose_model, map_location=self.device))
        self.pose_model = self.pose_model.to(self.device)
        self.pose_model.eval()

        self.patch_size = (192, 256)

        self.pose_transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        self.coco_skeletons = [
            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8],
            [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]
        ]
        self.body_labels = {0: 'nose', 1: 'left_eye', 2: 'right_eye', 3: 'left_ear', 4: 'right_ear',
                            5: 'left_shoulder', 6: 'right_shoulder', 7: 'left_elbow', 8: 'right_elbow', 9: 'left_wrist',
                            10: 'right_wrist', 11: 'left_hip', 12: 'right_hip', 13: 'left_knee', 14: 'right_knee',
                            15: 'left_ankle', 16: 'right_ankle'
                            }
        self.body_idx = dict([[v, k] for k, v in self.body_labels.items()])

    def preprocess(self, image):
        img = letterbox(image, new_shape=self.img_size)
        img = np.ascontiguousarray(img.transpose((2, 0, 1)))
        img = torch.from_numpy(img).to(self.device)
        img = img.float() / 255.0
        img = img[None]
        return img

    def box_to_center_scale(self, boxes, pixel_std=200):
        boxes = xyxy2xywh(boxes)
        r = self.patch_size[0] / self.patch_size[1]
        mask = boxes[:, 2] > boxes[:, 3] * r
        boxes[mask, 3] = boxes[mask, 2] / r
        boxes[~mask, 2] = boxes[~mask, 3] * r
        boxes[:, 2:] /= pixel_std
        boxes[:, 2:] *= 1.25
        return boxes

    def euclidian(self,point1, point2):
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def angle_calc(self,p0, p1, p2):
        '''
            p1 is center point from where we measured angle between p0 and
        '''
        try:
            a = (p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2
            b = (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
            c = (p2[0] - p0[0]) ** 2 + (p2[1] - p0[1]) ** 2
            angle = math.acos((a + b - c) / math.sqrt(4 * a * b)) * 180 / math.pi
        except:
            return 0
        return int(angle)

    def predict_poses(self, boxes, img):
        image_patches = []
        for cx, cy, w, h in boxes:
            trans = get_affine_transform(np.array([cx, cy]), np.array([w, h]), self.patch_size)
            img_patch = cv2.warpAffine(img, trans, self.patch_size, flags=cv2.INTER_LINEAR)
            img_patch = self.pose_transform(img_patch)
            image_patches.append(img_patch)

        image_patches = torch.stack(image_patches).to(self.device)
        return self.pose_model(image_patches)

    def humans_with_body_label(self, body_labels, humans, bbox):

        body_list = []
        bboxes = []
        for i, human in enumerate(humans):
            body_dict = {}
            bbox_value = [bbox[i][0], bbox[i][1], bbox[i][2], bbox[i][3]]
            for i in range(len(self.body_labels.keys())):
                body_part = human[i]
                center = body_part[0], body_part[1]
                body_dict[body_labels[i]] = center

            body_list.append(body_dict)
            bboxes.append((bbox_value, body_dict))
        return body_list, bboxes

    def postprocess(self, pred, img1, img0):
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=0)
        bbox = []
        for det in pred:
            if len(det):
                det[:, :4] = scale_coords(img1.shape[2:], det[:, :4], img0.shape).round()
                for box in det[:, :4].cpu().numpy():
                    pt1, pt2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                    bbox.append((box[0], box[1], box[2], box[3]))
                    cv2.rectangle(img0, pt1, pt2, (0, 255, 0), 1)
                boxes = self.box_to_center_scale(det[:, :4])
                outputs = self.predict_poses(boxes, img0)

                if 'simdr' in self.model_name:
                    coords = get_simdr_final_preds(*outputs, boxes, self.patch_size)
                else:
                    coords = get_final_preds(outputs, boxes)
                body_list, bboxes = self.humans_with_body_label(self.body_labels, coords, bbox)

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                tracker.update(xywhs.cpu(), confs.cpu(), clss.cpu(), img0, body_list)

        return img0

    @torch.no_grad()
    def predict(self, image, model):
        img = self.preprocess(image)
        pred = model(img)[0]
        frame = self.postprocess(pred, img, image)
        return frame


camera_prop = getConfigParam("camera_list", config_file)
url = getConfigParam("url", camera_prop[0])
cap = cv2.VideoCapture(url)
setup_cudnn()
pose = Pose(
    getConfigParam("path_to_pt", config_file),
    getConfigParam("pose_model", config_file),
    640,
    0.4,
    0.5
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tracker = StrongSORT(
    getConfigParam("tracker_model", config_file),
    device,
    max_dist=0.2,
    max_iou_distance=0.7,
    max_age=30,
    n_init=3,
    nn_budget=100,
    mc_lambda=0.995,
    ema_alpha=0.9,

)

trajectory = {}


while (cap.isOpened()):
    ret, frame = cap.read()
    frame_copy = np.copy(frame)
    if ret is False:
        break

    frame = pose.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), pose.det_model)

    for track in tracker.tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue

        track_id = track.track_id
        class_id = track.class_id
        conf = track.conf

        box = track.to_tlwh()
        x1, y1, x2, y2 = tracker._tlwh_to_xyxy(box)
        bboxes = x1, y1, x2, y2

        center = ((int(bboxes[0]) + int(bboxes[2])) // 2, (int(bboxes[1]) + int(bboxes[3])) // 2)
        action = ''

        if track_id not in trajectory:
            trajectory[track_id] = []
        trajectory[track_id].append(center)
        for i1 in range(1, len(trajectory[track_id])):
            if trajectory[track_id][i1 - 1] is None or trajectory[track_id][i1] is None:
                continue
            if abs(trajectory[track_id][i1-1][0] - trajectory[track_id][i1][0]) > 0:
                action = "walking"

        dst_l = (pose.angle_calc(track.pose_dict["left_hip"], track.pose_dict["left_knee"], track.pose_dict["left_ankle"]))
        dst_r = (pose.angle_calc(track.pose_dict["right_hip"], track.pose_dict["right_knee"], track.pose_dict["right_ankle"]))

        if dst_l < 120 and dst_r < 120:
            action = "sitting"
        elif action == "walking":
            pass
        else:
            action = "standing"
        track.status.append(action)
        if len(track.status) >= 5:
            track.action = max(track.status,key=track.status.count)
            track.status = []

        if track.action is not None:
            cv2.putText(frame, f"person : {track_id} - {track.action}", (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('frame', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()