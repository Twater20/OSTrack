import math

from lib.models.timostrack import build_timostrack
from lib.test.tracker.basetracker import BaseTracker
import torch
import numpy as np

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os

from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond
from filterpy.kalman import KalmanFilter



class TIMOSTrack(BaseTracker):
    def __init__(self, params, dataset_name):
        super(TIMOSTrack, self).__init__(params)
        network = build_timostrack(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu',weights_only=False)['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        self.kf = KalmanFilter(dim_x=4, dim_z=2)  # state: [x, y, dx, dy]
        self.kf.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        self.kf.R *= 1.15  # measurement noise
        self.kf.P[2:, 2:] *= 1.1  # state uncertainty
        self.kf.P *= 5.
        self.kf.Q[2:, 2:] *= 4.00  # process noise


        # for debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                # self.add_hook()
                self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}
        self.next = None
        
        # 历史跟踪序列缓存 - 保存前50帧的跟踪结果
        self.history_sequence = []
        self.max_history_length = self.cfg.TIMING.seq_len  # 保持50帧历史

    def initialize(self, image, info: dict):
        # forward the template once
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            self.z_dict1 = template

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.tensors.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox)

        # Initialize Kalman Filter state
        self.kf.x = np.array([info['init_bbox'][0], info['init_bbox'][1], 0, 0]).reshape(-1, 1)  # [x, y, dx, dy]

        # save states
        self.state = info['init_bbox']
        
        # 初始化历史序列 - 用初始边界框填充50帧
        initial_bbox = info['init_bbox']  # [x, y, w, h]
        # H, W, _ = image.shape
        # initial_bbox = [initial_bbox[0] / W, initial_bbox[1] / H, initial_bbox[2] / W, initial_bbox[3] / H]
        self.history_sequence = [initial_bbox] * self.max_history_length
        
        # 转换为TimesNet需要的格式 [seq_len, 4] -> [1, seq_len, 4]
        self.gt_seq = torch.tensor(self.history_sequence, dtype=torch.float32).unsqueeze(0).cuda()
        
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.next, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        with torch.no_grad():
            x_dict = search
            # merge the template and the search
            # run the transformer
            out_dict = self.network.forward(
                template=self.z_dict1.tensors, search=x_dict.tensors, gt_sequence_anno_backward=self.gt_seq,ce_template_mask=self.box_mask_z)

        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        detected_box = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        # self.seq = [self.state[0] / W, self.state[1] / H, self.state[2] / W, self.state[3] / H]

        # get the final box result

        # First: Kalman Filter prediction and update
        self.kf.predict()
        measurement = np.array([detected_box[0], detected_box[1]]).reshape(-1, 1)
        self.kf.update(measurement)

        # Second: Use detection result as final output
        self.state = detected_box
        # Finally: Use Kalman filtered state for next frame search
        kalman_state = self.kf.x[:2].flatten()  # [x, y]
        self.next = [kalman_state[0], kalman_state[1], detected_box[2],
                     detected_box[3]]  # Use Kalman filtered position with detected size

        # 更新历史序列 - 滑动窗口机制
        # self.history_sequence.append(self.seq)
        self.history_sequence.append(self.state)
        if len(self.history_sequence) > self.max_history_length:
            self.history_sequence.pop(0)  # 移除最旧的帧
        
        # 更新gt_seq为TimesNet格式
        self.gt_seq = torch.tensor(self.history_sequence, dtype=torch.float32).unsqueeze(0).cuda()

        self.update_threshold = 1.2
        self.update_intervals = 200  # 290

        if self.frame_id % self.update_intervals == 0 or self.frame_id <= 5:
            # 触发模板采集
            self.collecting_templates = True
            self.templates_collected = 0
            self.template_bank = []

        # 如果处于模板采集状态，则采集当前帧模板
        if self.collecting_templates:
            self.collect_current_frame_template(image, initial_score=pred_score_map.max().item(),
                                                hann_score=response.max().item())
            # 检查是否完成采集
            if self.templates_collected >= 20 or (self.templates_collected >= 5 and self.frame_id < 20):
                self.select_and_update_template(initial_score_threshold=0.866, hann_score_threshold=0.851)
                self.collecting_templates = False
        #
        # # Save detection and Kalman results
        # self.save_dir = "debug"
        # if not os.path.exists(self.save_dir):
        #     os.makedirs(self.save_dir)
        # gt=info['gt_bbox'].tolist()
        # x0=gt[0]
        # y0=gt[1]
        # w0=gt[2]
        # h0=gt[3]
        # x1, y1, w, h = self.state
        # result_path = os.path.join(self.save_dir, "results.txt")
        # with open(result_path, 'a') as f:
        #     f.write(f"{self.frame_id} {x0} {y0} {w0} {h0} ; {x1:.1f} {y1:.1f} {w:.1f} {h:.1f}; \n")

        # for debug
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:
                self.visdom.register((image, info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')
                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region')
                self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template')
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann')

                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}

    def _is_near_border(self, box, H, W, threshold):
        """判断框是否靠近图像边界"""
        x, y, w, h = box
        return (x < threshold or y < threshold or
                (x+W/4 ) > (W - threshold) or (y+H/4 ) > (H - threshold))

    def collect_current_frame_template(self, image, initial_score=0.0, hann_score=0.0):
        """采集当前帧的模板"""
        # 提取模板

        z_patch_arr, _, z_amask_arr = sample_target(image, self.state, self.params.template_factor,
                                                    output_sz=self.params.template_size)
        # 存储模板和得分
        self.template_bank.append((self.frame_id,z_patch_arr, initial_score, hann_score,z_amask_arr))
        self.templates_collected += 1
    def select_and_update_template(self,initial_score_threshold, hann_score_threshold):
        """筛选并选择最优模板"""
        # 筛选合格模板
        valid_templates = [t for t in self.template_bank if t[2] > initial_score_threshold and t[3] > hann_score_threshold]
        # 如果合格模板数量足够，选择最优模板
        if len(valid_templates) >= 5 or (self.frame_id<=20 and len(valid_templates) >= 2):
            # 按初始得分排序
            sorted_templates = sorted(valid_templates, key=lambda x: x[2], reverse=True)
            # 按Hann得分排序
            hann_sorted = sorted(valid_templates, key=lambda x: x[3], reverse=True)

            top_initial = sorted_templates[:3]
            top_hann = hann_sorted[:3]

            # 查找同时在两个列表中的模板
            candidates = []
            for t in top_initial:
                if t in top_hann:
                    # 计算综合得分（这里使用乘积，确保两者都高）
                    combined_score = t[2] * t[3]
                    candidates.append((t[0], t[1], t[2], t[3],t[4],combined_score))
            #best_template = sorted_templates[0]
            if candidates:
                best_template = sorted(candidates, key=lambda x: x[5], reverse=True)[0]
            else:
                best_template = hann_sorted[0]
            # 更新当前模板
            self.z_patch_arr = best_template[1]  # 更新模板
            z_amask_arr = best_template[4]  # 更新模板掩码
            template = self.preprocessor.process(self.z_patch_arr, z_amask_arr)
            self.z_dict1 = template

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights


def get_tracker_class():
    return TIMOSTrack
