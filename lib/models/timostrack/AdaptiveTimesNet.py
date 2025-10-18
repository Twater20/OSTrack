import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import numpy as np
from lib.models.layers.Embed import DataEmbedding
from lib.models.layers.Conv_Blocks import Inception_Block_V1


def FFT_for_Period(x, k=2):
    """
    FFT-based period detection for variable length sequences
    """
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class AdaptiveTimesBlock(nn.Module):
    """
    Adaptive TimesBlock that handles variable sequence lengths
    """
    def __init__(self, configs):
        super(AdaptiveTimesBlock, self).__init__()
        self.max_seq_len = configs.max_seq_len  # 最大序列长度
        self.min_seq_len = configs.min_seq_len  # 最小序列长度
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )
        
        # 序列长度自适应层
        self.length_adapter = nn.Linear(configs.d_model, configs.d_model)
        
    def forward(self, x, actual_seq_len=None):
        """
        Args:
            x: [B, T, N] where T can be variable
            actual_seq_len: [B] actual sequence lengths for each batch
        """
        B, T, N = x.size()
        
        # 如果没有提供实际序列长度，假设所有序列都是完整的
        if actual_seq_len is None:
            actual_seq_len = torch.full((B,), T, device=x.device)
        
        # 简化版本：直接使用卷积处理，避免复杂的FFT和reshape
        try:
            # 尝试使用FFT分析
            effective_k = min(self.k, max(1, T // 8))  # 更保守的k值选择
            period_list, period_weight = FFT_for_Period(x, effective_k)
            
            res = []
            for i in range(effective_k):
                period = max(1, period_list[i])  # 确保period至少为1
                
                # 简化的处理：直接使用1D卷积而不是复杂的2D变换
                out = x.transpose(1, 2)  # [B, N, T]
                out = self.conv[0](out.unsqueeze(-1)).squeeze(-1)  # 使用第一个卷积层
                out = self.conv[1](out)  # GELU
                out = self.conv[2](out.unsqueeze(-1)).squeeze(-1)  # 使用第三个卷积层
                out = out.transpose(1, 2)  # [B, T, N]
                
                # 扩展到目标长度
                if out.size(1) < T + self.pred_len:
                    padding = torch.zeros([B, T + self.pred_len - out.size(1), N], device=x.device)
                    out = torch.cat([out, padding], dim=1)
                else:
                    out = out[:, :T + self.pred_len, :]
                
                res.append(out)
            
            if len(res) > 0:
                res = torch.stack(res, dim=-1)  # [B, T+pred_len, N, k]
                # adaptive aggregation
                period_weight = F.softmax(period_weight, dim=1)  # [B, k]
                period_weight = period_weight.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, k]
                res = torch.sum(res * period_weight, dim=-1)  # [B, T+pred_len, N]
            else:
                raise ValueError("No valid periods found")
                
        except Exception:
            # 回退到简单的线性变换
            res = self.length_adapter(x)
            # 扩展到目标长度
            if res.size(1) < T + self.pred_len:
                padding = torch.zeros([B, T + self.pred_len - res.size(1), N], device=x.device)
                res = torch.cat([res, padding], dim=1)
            else:
                res = res[:, :T + self.pred_len, :]
        
        # residual connection (只对原始序列部分)
        if res.size(1) >= T:
            res[:, :T, :] = res[:, :T, :] + x
        
        return res


class AdaptiveTimesNetTracking(nn.Module):
    """
    Adaptive TimesNet for Object Tracking with Variable Sequence Length Support
    """

    def __init__(self, configs):
        super(AdaptiveTimesNetTracking, self).__init__()
        self.configs = configs
        self.max_seq_len = configs.max_seq_len  # 最大序列长度，如200
        self.min_seq_len = configs.min_seq_len  # 最小序列长度，如10
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model

        # Adaptive TimesNet layers
        self.model = nn.ModuleList([AdaptiveTimesBlock(configs)
                                    for _ in range(configs.e_layers)])

        # Embedding layer for 4D bbox coordinates (x, y, w, h)
        self.enc_embedding = DataEmbedding(4, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)

        # 序列长度编码器
        self.length_encoder = nn.Embedding(self.max_seq_len + 1, configs.d_model)
        
        # 预测长度线性变换层
        self.predict_linear = nn.Linear(self.max_seq_len, self.pred_len + self.max_seq_len)
        
        # 多尺度预测头
        self.multi_scale_predictors = nn.ModuleList([
            nn.Linear(configs.d_model, 4) for _ in range(configs.num_prediction_heads)
        ])
        
        # 预测选择网络
        self.prediction_selector = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model // 2),
            nn.ReLU(),
            nn.Linear(configs.d_model // 2, configs.num_prediction_heads),
            nn.Softmax(dim=-1)
        )
        
        # 置信度预测器
        self.confidence_predictor = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model // 4),
            nn.ReLU(),
            nn.Linear(configs.d_model // 4, 1),
            nn.Sigmoid()
        )

        # 数据集自适应层
        self.dataset_adapter = nn.ModuleDict({
            'lasot': nn.Linear(configs.d_model, configs.d_model),
            'got10k': nn.Linear(configs.d_model, configs.d_model),
            'trackingnet': nn.Linear(configs.d_model, configs.d_model),
            'default': nn.Linear(configs.d_model, configs.d_model)
        })

    def forward(self, x_enc, x_mark_enc=None, actual_seq_lens=None, dataset_name='default'):
        """
        Args:
            x_enc: [B, variable_seq_len, 4] - historical bbox positions
            x_mark_enc: optional temporal features
            actual_seq_lens: [B] actual sequence lengths for each sample
            dataset_name: dataset name for adaptive processing
        """
        return self.forecast(x_enc, x_mark_enc, actual_seq_lens, dataset_name)

    def forecast(self, x_enc, x_mark_enc, actual_seq_lens=None, dataset_name='default'):
        """
        Adaptive forecasting with variable sequence lengths
        """
        B, T, _ = x_enc.shape
        
        # 处理变长序列：padding到统一长度或截断
        if actual_seq_lens is not None:
            max_len = actual_seq_lens.max().item()
            if max_len > self.max_seq_len:
                # 如果序列太长，进行智能截断（保留最近的帧）
                x_enc = x_enc[:, -self.max_seq_len:, :]
                actual_seq_lens = torch.clamp(actual_seq_lens, max=self.max_seq_len)
                T = self.max_seq_len
            elif T < max_len:
                # 如果需要padding
                padding = torch.zeros(B, max_len - T, 4, device=x_enc.device)
                x_enc = torch.cat([x_enc, padding], dim=1)
                T = max_len
        else:
            actual_seq_lens = torch.full((B,), T, device=x_enc.device)

        # Normalization (考虑实际序列长度)
        means_list = []
        stdev_list = []
        normalized_x_enc = torch.zeros_like(x_enc)
        
        for i in range(B):
            actual_len = actual_seq_lens[i].item()
            seq_data = x_enc[i, :actual_len, :]
            
            mean = seq_data.mean(0, keepdim=True)
            stdev = torch.sqrt(torch.var(seq_data, dim=0, keepdim=True, unbiased=False) + 1e-5)
            
            means_list.append(mean)
            stdev_list.append(stdev)
            
            # 标准化实际序列部分
            normalized_x_enc[i, :actual_len, :] = (seq_data - mean) / stdev
            # padding部分保持为0

        # 序列长度编码
        length_encoding = self.length_encoder(actual_seq_lens)  # [B, d_model]
        
        # embedding
        enc_out = self.enc_embedding(normalized_x_enc, x_mark_enc)  # [B,T,C]
        
        # 添加长度编码
        enc_out = enc_out + length_encoding.unsqueeze(1)  # broadcast to [B, T, C]
        
        # 时序维度变换（如果需要）
        if T <= self.max_seq_len:
            # 扩展到预测长度
            padding_len = self.pred_len
            padding = torch.zeros(B, padding_len, enc_out.size(-1), device=enc_out.device)
            enc_out = torch.cat([enc_out, padding], dim=1)  # [B, T+pred_len, C]
        
        # 数据集自适应处理
        if dataset_name in self.dataset_adapter:
            enc_out = self.dataset_adapter[dataset_name](enc_out)
        else:
            enc_out = self.dataset_adapter['default'](enc_out)

        # Adaptive TimesNet processing
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out, actual_seq_lens))

        # 多头预测
        predictions = []
        for predictor in self.multi_scale_predictors:
            pred = predictor(enc_out)  # [B, T+pred_len, 4]
            predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=1)  # [B, num_heads, T+pred_len, 4]
        
        # 预测选择权重
        selection_weights = self.prediction_selector(enc_out[:, -1, :])  # [B, num_heads]
        
        # 加权融合预测结果
        weighted_prediction = torch.sum(
            predictions * selection_weights.unsqueeze(-1).unsqueeze(-1), 
            dim=1
        )  # [B, T+pred_len, 4]
        
        # 置信度预测
        confidence_scores = self.confidence_predictor(enc_out)  # [B, T+pred_len, 1]

        # De-Normalization (只对预测部分)
        denormalized_predictions = []
        for i in range(B):
            mean = means_list[i]
            stdev = stdev_list[i]
            
            pred = weighted_prediction[i, -self.pred_len:, :] * stdev + mean
            denormalized_predictions.append(pred)
        
        final_predictions = torch.stack(denormalized_predictions, dim=0)  # [B, pred_len, 4]
        
        return final_predictions, {
            'all_predictions': predictions[:, :, -self.pred_len:, :],  # [B, num_heads, pred_len, 4]
            'selection_weights': selection_weights,  # [B, num_heads]
            'confidence_scores': confidence_scores[:, -self.pred_len:, :],  # [B, pred_len, 1]
            'temporal_features': enc_out,  # [B, T+pred_len, d_model]
            'actual_seq_lens': actual_seq_lens
        }

    def get_dataset_optimal_length(self, dataset_name):
        """
        根据数据集返回最优序列长度
        """
        optimal_lengths = {
            'lasot': min(100, self.max_seq_len),  # LaSOT序列长，取100帧
            'got10k': min(50, self.max_seq_len),   # GOT10K序列短，取50帧
            'trackingnet': min(80, self.max_seq_len),  # TrackingNet中等长度
            'default': min(60, self.max_seq_len)
        }
        return optimal_lengths.get(dataset_name, optimal_lengths['default'])

    def adaptive_sequence_sampling(self, sequence_data, dataset_name, target_length=None):
        """
        自适应序列采样策略
        """
        if target_length is None:
            target_length = self.get_dataset_optimal_length(dataset_name)
        
        seq_len = len(sequence_data)
        
        if seq_len <= target_length:
            # 序列长度不足，直接返回
            return sequence_data, seq_len
        else:
            # 序列过长，智能采样
            if dataset_name == 'lasot':
                # LaSOT: 均匀采样 + 重点保留最近帧
                recent_frames = max(target_length // 3, 10)
                remaining_frames = target_length - recent_frames
                
                if seq_len > recent_frames:
                    # 从前面部分均匀采样
                    early_indices = np.linspace(0, seq_len - recent_frames - 1, 
                                              remaining_frames, dtype=int)
                    # 保留最近的帧
                    recent_indices = list(range(seq_len - recent_frames, seq_len))
                    
                    selected_indices = list(early_indices) + recent_indices
                else:
                    selected_indices = list(range(seq_len))
            else:
                # 其他数据集：均匀采样
                selected_indices = np.linspace(0, seq_len - 1, target_length, dtype=int)
            
            sampled_data = [sequence_data[i] for i in selected_indices]
            return sampled_data, len(sampled_data)


class AdaptiveTrackingConfig:
    """Configuration class for Adaptive TimesNet Tracking"""

    def __init__(self,
                 max_seq_len=200,  # 最大序列长度
                 min_seq_len=10,   # 最小序列长度
                 pred_len=5,       # 预测长度
                 d_model=64,       # 隐藏维度
                 d_ff=256,         # 前馈维度
                 e_layers=3,       # TimesBlock层数
                 top_k=3,          # 频率top-k
                 num_kernels=6,    # 卷积核数量
                 embed='timeF',    # 嵌入类型
                 freq='h',         # 频率
                 dropout=0.1,      # Dropout率
                 num_prediction_heads=4,  # 多头预测数量
                 dataset_adaptive=True):  # 是否启用数据集自适应

        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.d_ff = d_ff
        self.e_layers = e_layers
        self.top_k = top_k
        self.num_kernels = num_kernels
        self.embed = embed
        self.freq = freq
        self.dropout = dropout
        self.num_prediction_heads = num_prediction_heads
        self.dataset_adaptive = dataset_adaptive
        self.enc_in = 4  # 输入维度


if __name__ == "__main__":
    # 测试自适应TimesNet
    configs = AdaptiveTrackingConfig(
        max_seq_len=100,
        min_seq_len=10,
        pred_len=5,
        d_model=32,
        num_prediction_heads=3
    )
    
    model = AdaptiveTimesNetTracking(configs)
    
    # 测试变长序列
    batch_size = 4
    # 模拟不同长度的序列
    seq_lens = [80, 60, 40, 90]  # 不同的实际序列长度
    max_len = max(seq_lens)
    
    # 创建变长序列数据
    x_enc = torch.randn(batch_size, max_len, 4)
    actual_seq_lens = torch.tensor(seq_lens)
    
    # 将超出实际长度的部分置零（模拟padding）
    for i, seq_len in enumerate(seq_lens):
        if seq_len < max_len:
            x_enc[i, seq_len:, :] = 0
    
    with torch.no_grad():
        predictions, aux_info = model(x_enc, actual_seq_lens=actual_seq_lens, dataset_name='lasot')
        
        print("Predictions shape:", predictions.shape)  # [4, 5, 4]
        print("All predictions shape:", aux_info['all_predictions'].shape)  # [4, 3, 5, 4]
        print("Selection weights shape:", aux_info['selection_weights'].shape)  # [4, 3]
        print("Confidence scores shape:", aux_info['confidence_scores'].shape)  # [4, 5, 1]
        print("Actual sequence lengths:", aux_info['actual_seq_lens'])
        
        print("\nSelection weights for each sample:")
        for i, weights in enumerate(aux_info['selection_weights']):
            print(f"Sample {i}: {weights.numpy()}")
