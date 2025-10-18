from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate


class TIMOSTrackActor(BaseActor):
    """ Actor for training OSTrack models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(out_dict, data)

        return loss, status

    def forward_pass(self, data):
        # currently only support 1 template and 1 search region
        assert len(data['template_images']) == 1
        assert len(data['search_images']) == 1

        template_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1,
                                                             *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
            # template_att_i = data['template_att'][i].view(-1, *data['template_att'].shape[2:])  # (batch, 128, 128)
            template_list.append(template_img_i)

        search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 320, 320)
        # search_att = data['search_att'][0].view(-1, *data['search_att'].shape[2:])  # (batch, 320, 320)

        gt_sequence_anno_backward=data['gt_sequence_anno_backward'].view(-1, *data['gt_sequence_anno_backward'].shape[2:])#历史帧序列的标签
        #gt_sequence_anno_forward=data['gt_sequence_anno_forward']
        
        # 获取数据集名称和实际序列长度（用于自适应TimesNet）
        dataset_name = data.get('dataset', 'default')
        if isinstance(dataset_name, (list, tuple)):
            dataset_name = dataset_name[0] if len(dataset_name) > 0 else 'default'
        if isinstance(dataset_name, str):
            dataset_name = dataset_name.lower()
        
        # 计算实际序列长度（非零部分）
        actual_seq_lens = self._compute_actual_sequence_lengths(gt_sequence_anno_backward)

        box_mask_z = None
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            box_mask_z = generate_mask_cond(self.cfg, template_list[0].shape[0], template_list[0].device,
                                            data['template_anno'][0])#[32,64]

            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH#20
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH#80
            ce_keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=ce_start_epoch,
                                                total_epochs=ce_start_epoch + ce_warm_epoch,
                                                ITERS_PER_EPOCH=1,
                                                base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])

        if len(template_list) == 1:
            template_list = template_list[0]

        out_dict = self.net(template=template_list,
                            search=search_img,
                            gt_sequence_anno_backward=gt_sequence_anno_backward,
                            ce_template_mask=box_mask_z,
                            ce_keep_rate=ce_keep_rate,
                            return_last_attn=False,
                            dataset_name=dataset_name,
                            actual_seq_lens=actual_seq_lens)

        return out_dict

    def _compute_actual_sequence_lengths(self, gt_sequence_anno_backward):
        """
        计算每个样本的实际序列长度（非零部分）
        Args:
            gt_sequence_anno_backward: [B, seq_len, 4] 
        Returns:
            actual_seq_lens: [B] 每个样本的实际序列长度
        """
        B, seq_len, _ = gt_sequence_anno_backward.shape
        actual_seq_lens = []
        
        for i in range(B):
            # 找到非零bbox的数量
            bbox_norms = torch.norm(gt_sequence_anno_backward[i], dim=1)  # [seq_len]
            non_zero_count = (bbox_norms > 1e-6).sum().item()
            
            # 确保至少有最小长度
            actual_len = max(non_zero_count, self.cfg.TIMING.get('min_seq_len', 10))
            # 限制最大长度
            actual_len = min(actual_len, self.cfg.TIMING.get('max_seq_len', 200))
            
            actual_seq_lens.append(actual_len)
        
        return torch.tensor(actual_seq_lens, device=gt_sequence_anno_backward.device)

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        # gt gaussian map
        gt_bbox = gt_dict['search_anno'][-1] #[32,4] # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)
        gt_sequence_anno_forward=gt_dict['gt_sequence_anno_forward'].squeeze(0)

        # Get boxes
        pred_boxes = pred_dict['pred_boxes']#[32,1,4]
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # compute location loss
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)
        
        # compute timesnet loss
        timesnet_l1_loss = torch.tensor(0.0, device=l1_loss.device)
        timesnet_giou_loss = torch.tensor(0.0, device=l1_loss.device)
        multi_head_loss = torch.tensor(0.0, device=l1_loss.device)
        confidence_loss = torch.tensor(0.0, device=l1_loss.device)
        
        if 'timesnet_pred' in pred_dict and pred_dict['timesnet_pred'] is not None:
            # TimesNet预测的bbox: [B, pred_len, 4] 格式为(x, y, w, h)
            timesnet_pred_boxes = pred_dict['timesnet_pred']  
            
            # 使用最后一个预测作为当前帧的预测
            if timesnet_pred_boxes.dim() == 3:  # [B, pred_len, 4]
                timesnet_pred_current = timesnet_pred_boxes[:, -1, :]  # [B, 4] 取最后一个时间步的预测
            else:  # [B, 4]
                timesnet_pred_current = timesnet_pred_boxes
            
           # TimesNet预测框处理：转换为xyxy格式
            timesnet_pred_xyxy = box_xywh_to_xyxy(timesnet_pred_current)
            
            # 使用gt_sequence_anno_forward作为GT框（预测序列）
            # 取最后一个时间步的GT框作为当前帧的GT
            if gt_sequence_anno_forward.dim() == 3:  # [B, seq_len, 4]
                gt_bbox_forward = gt_sequence_anno_forward[:, -1, :]  # [B, 4] 取最后一个时间步的GT
            else:  # [B, 4]
                gt_bbox_forward = gt_sequence_anno_forward

             # GT框处理：转换为xyxy格式（与OSTrack保持一致）
            gt_bbox_xyxy = box_xywh_to_xyxy(gt_bbox_forward).clamp(min=0.0, max=1.0)
            
            # 计算TimesNet的L1损失
            timesnet_l1_loss = self.objective['timesnet_l1'](timesnet_pred_xyxy, gt_bbox_xyxy)

            # 计算TimesNet的GIoU损失
            try:
                timesnet_giou_loss, _ = self.objective['timesnet_giou'](timesnet_pred_xyxy, gt_bbox_xyxy)
            except:
                timesnet_giou_loss = torch.tensor(0.0, device=l1_loss.device)
            
            # 计算多头预测损失（如果使用自适应TimesNet）
            if 'timesnet_all_predictions' in pred_dict and pred_dict['timesnet_all_predictions'] is not None:
                all_predictions = pred_dict['timesnet_all_predictions']  # [B, num_heads, pred_len, 4]
                selection_weights = pred_dict.get('timesnet_selection_weights', None)  # [B, num_heads]
                
                if all_predictions is not None and selection_weights is not None:
                    B, num_heads, pred_len, _ = all_predictions.shape
                    
                    # 对每个预测头计算损失
                    head_losses = []
                    for h in range(num_heads):
                        head_pred = all_predictions[:, h, -1, :]  # [B, 4] 取最后一个时间步
                        head_pred_xyxy = box_xywh_to_xyxy(head_pred)
                        
                        try:
                            head_l1 = self.objective['timesnet_l1'](head_pred_xyxy, gt_bbox_xyxy)
                            head_giou, _ = self.objective['timesnet_giou'](head_pred_xyxy, gt_bbox_xyxy)
                            head_loss = head_l1 + head_giou
                        except:
                            head_loss = torch.tensor(0.0, device=l1_loss.device)
                        
                        head_losses.append(head_loss)
                    
                    # 加权平均多头损失
                    if head_losses:
                        head_losses = torch.stack(head_losses, dim=0)  # [num_heads]
                        # 使用选择权重加权损失
                        weighted_head_losses = head_losses.unsqueeze(0) * selection_weights  # [B, num_heads]
                        multi_head_loss = weighted_head_losses.sum(dim=1).mean()  # 平均batch损失
            
            # 计算置信度损失（如果有置信度预测）
            if 'timesnet_confidence_scores' in pred_dict and pred_dict['timesnet_confidence_scores'] is not None:
                confidence_scores = pred_dict['timesnet_confidence_scores']  # [B, pred_len, 1]
                if confidence_scores is not None:
                    # 使用IoU作为置信度目标
                    try:
                        _, iou_scores = self.objective['timesnet_giou'](timesnet_pred_xyxy, gt_bbox_xyxy)
                        target_confidence = iou_scores.detach().unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
                        target_confidence = target_confidence.expand(-1, confidence_scores.size(1), -1)  # [B, pred_len, 1]
                        
                        # MSE损失用于置信度回归
                        confidence_loss = F.mse_loss(confidence_scores, target_confidence)
                    except:
                        confidence_loss = torch.tensor(0.0, device=l1_loss.device)
        
        # weighted sum
        loss = (self.loss_weight['giou'] * giou_loss + 
                self.loss_weight['l1'] * l1_loss + 
                self.loss_weight['focal'] * location_loss +
                self.loss_weight['timesnet_l1'] * timesnet_l1_loss +
                self.loss_weight['timesnet_giou'] * timesnet_giou_loss +
                self.loss_weight.get('multi_head', 0.1) * multi_head_loss +
                self.loss_weight.get('confidence', 0.05) * confidence_loss)
        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      "Loss/timesnet_l1": timesnet_l1_loss.item(),
                      "Loss/timesnet_giou": timesnet_giou_loss.item(),
                      "Loss/multi_head": multi_head_loss.item(),
                      "Loss/confidence": confidence_loss.item(),
                      "IoU": mean_iou.item()}
            return loss, status
        else:
            return loss
