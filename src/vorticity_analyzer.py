"""
Navier-Stokes 涡量场正则性分析器
Vorticity-based Regularity Predictor using Deep Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

# ============================================================
# 神经网络架构定义
# ============================================================

class VorticityEncoder(nn.Module):
    """3D 涡量场编码器"""
    
    def __init__(self, in_channels=3, hidden_dims=[32, 64, 128, 256]):
        super().__init__()
        
        layers = []
        for i, out_ch in enumerate(hidden_dims):
            layers.append(nn.Conv3d(
                in_channels if i == 0 else hidden_dims[i-1],
                out_ch,
                kernel_size=3,
                stride=2 if i > 0 else 1,
                padding=1
            ))
            layers.append(nn.BatchNorm3d(out_ch))
            layers.append(nn.ReLU(inplace=True))
        
        self.encoder = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]
    
    def forward(self, x):
        return self.encoder(x)


class PhysicsAttention(nn.Module):
    """简化版物理信息注意力机制"""
    
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        
        # 确保能被整除
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
        # 物理先验: 涡量梯度权重
        self.vorticity_weight = nn.Parameter(torch.ones(1))
    
    def forward(self, x, vorticity_features=None):
        B, C, D, H, W = x.shape
        
        # 全局池化
        x_pooled = F.adaptive_avg_pool3d(x, 1).squeeze(-1).squeeze(-1).squeeze(-1)  # (B, C)
        
        # QKV 投影
        qkv = self.qkv(x_pooled).reshape(B, 3, self.num_heads, self.head_dim)  # (B, 3, heads, head_dim)
        
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # 各 (B, heads, head_dim)
        
        # 简化的注意力
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)  # (B, heads, head_dim)
        out = out.reshape(B, -1)  # (B, dim)
        out = self.proj(out)  # (B, dim)
        
        return out


class RegularityPredictor(nn.Module):
    """正则性预测器: 输出正则性评分 [0, 1]"""
    
    def __init__(self, feature_dim, hidden_dims=[256, 128, 64]):
        super().__init__()
        
        layers = []
        in_dim = feature_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x).squeeze(-1)


class VorticityAnalyzer(nn.Module):
    """完整的涡量分析模型"""
    
    def __init__(
        self,
        in_channels=3,
        encoder_dims=[32, 64, 128, 256],
        attention_heads=8,
        classifier_dims=[256, 128, 64]
    ):
        super().__init__()
        
        self.encoder = VorticityEncoder(in_channels, encoder_dims)
        
        # 特征维度计算 (简化版: 直接池化)
        self.feat_dim = encoder_dims[-1]
        
        self.attention = PhysicsAttention(self.feat_dim, attention_heads)
        self.predictor = RegularityPredictor(self.feat_dim, classifier_dims)
    
    def forward(self, vorticity_field):
        """
        Args:
            vorticity_field: (B, C, D, H, W) 涡量场张量
        Returns:
            regularity_score: (B,) 正则性评分 [0,1], 1=正则, 0=可能奇异
        """
        features = self.encoder(vorticity_field)
        attended = self.attention(features)
        score = self.predictor(attended)
        return score
    
    def extract_features(self, vorticity_field):
        """提取特征用于可视化"""
        features = self.encoder(vorticity_field)
        return features


# ============================================================
# 涡量特征提取
# ============================================================

def compute_vorticity(velocity_field: torch.Tensor) -> torch.Tensor:
    """
    从速度场计算涡量场
    Args:
        velocity_field: (B, 3, D, H, W) 速度场
    Returns:
        vorticity: (B, 3, D, H, W) 涡量场 ω = ∇ × u
    """
    u_x, u_y, u_z = torch.unbind(velocity_field, dim=1)
    
    # 偏导数计算 (简单有限差分)
    # ∂u_y/∂x - ∂u_x/∂y = ω_z
    dx = torch.gradient(u_y, dim=3)[0] - torch.gradient(u_x, dim=2)[0]
    dy = torch.gradient(u_x, dim=3)[0] - torch.gradient(u_z, dim=1)[0]
    dz = torch.gradient(u_z, dim=2)[0] - torch.gradient(u_y, dim=1)[0]
    
    vorticity = torch.stack([dx, dy, dz], dim=1)
    return vorticity


def compute_Q_criterion(vorticity: torch.Tensor) -> torch.Tensor:
    """
    计算 Q-criterion: Q = 0.5 * (||Ω||^2 - ||S||^2)
    其中 Ω 是旋转率张量, S 是应变率张量
    
    Args:
        vorticity: (B, 3, D, H, W) 涡量场
    Returns:
        Q: (B, D, H, W) Q 判据
    """
    # 简化: Q ≈ 0.5 * ||ω||^2 作为涡量强度的代理
    # 完整的 Q 需要从速度梯度张量计算 strain rate
    # 这里使用涡量范数作为近似（满足 Q>0 区域对应涡旋核心）
    omega_norm = torch.norm(vorticity, dim=1)  # (B, D, H, W)
    Q = 0.5 * (omega_norm ** 2)
    return Q


def compute_vorticity_stats(vorticity: torch.Tensor) -> dict:
    """计算涡量场统计量"""
    omega_norm = torch.norm(vorticity, dim=1)
    
    stats = {
        'max': omega_norm.max().item(),
        'min': omega_norm.min().item(),
        'mean': omega_norm.mean().item(),
        'std': omega_norm.std().item(),
        'p99': torch.quantile(omega_norm, 0.99).item(),
    }
    
    return stats


# ============================================================
# 训练工具
# ============================================================

def create_synthetic_vorticity(batch_size=8, size=64):
    """生成合成涡量场用于测试"""
    # 随机噪声基础
    omega = torch.randn(batch_size, 3, size, size, size)
    
    # 添加涡旋结构
    x = torch.linspace(-1, 1, size)
    y = torch.linspace(-1, 1, size)
    z = torch.linspace(-1, 1, size)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    
    # Taylor-Green 涡旋
    taylor_green = torch.stack([
        torch.sin(X) * torch.cos(Y) * torch.cos(Z),
        -torch.cos(X) * torch.sin(Y) * torch.cos(Z),
        torch.zeros_like(X)
    ], dim=0).unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
    
    # 混合
    alpha = torch.rand(batch_size, 1, 1, 1, 1)
    omega = alpha * omega + (1 - alpha) * taylor_green * 10
    
    return omega


class EnhancedPhysicsLoss(nn.Module):
    """
    增强版物理一致性损失
    
    编码以下 NS 方程物理约束:
    1. 不可压条件: ∇·u = 0
    2. BKM 准则: ||ω||_{L^∞} 控制
    3. 涡量传输方程: Dω/Dt = (ω·∇)u + νΔω
    4. 能量耗散: 能量随时间递减
    5. Enstrophy 演化: 涡量场的平滑性
    6. [新增] Prodi-Serrin 条件: L^p(L^q) 可积性约束
    
    论文参考:
    - Beale-Kato-Majda (1984): ||ω||_{L^∞} 可积性 → 正则性保证
    - Prodi-Serrin (1959): L^p(L^q) 正则性条件
    - Escauriaza-Seregin-Sverak (2003): L_{3,∞} 正则性
    - Raissi et al. (2019): PINN 物理损失函数框架
    - Kochkov et al. (2024): 机器学习加速 CFD
    """
    
    def __init__(
        self,
        viscosity=0.01,
        lambda_div=1.0,       # 不可压条件权重
        lambda_bkm=0.5,       # BKM 准则权重
        lambda_smooth=0.1,    # 平滑性权重
        lambda_enstrophy=0.1, # Enstrophy 权重
        lambda_ps=0.3,        # [新增] Prodi-Serrin 权重
        bkm_threshold=5.0,    # [更新 2026-05-17] 优化阈值，从 1e4 降至 5.0
    ):
        super().__init__()
        self.viscosity = viscosity
        self.lambda_div = lambda_div
        self.lambda_bkm = lambda_bkm
        self.lambda_smooth = lambda_smooth
        self.lambda_enstrophy = lambda_enstrophy
        self.lambda_ps = lambda_ps
        self.bkm_threshold = bkm_threshold
    
    def compute_divergence(self, velocity):
        """计算速度场散度 ∇·u，验证不可压条件 (向量化版本)"""
        B, C, D, H, W = velocity.shape
        # 向量化: 直接对所有批次同时计算
        # velocity[b, 0] = u1, [1]=u2, [2]=u3
        # ∂u1/∂x (dim=3), ∂u2/∂y (dim=2), ∂u3/∂z (dim=1)
        u1 = velocity[:, 0]  # (B, D, H, W)
        u2 = velocity[:, 1]
        u3 = velocity[:, 2]
        # 有限差分散度: ∇·u = ∂u1/∂x + ∂u2/∂y + ∂u3/∂z
        div_u = (
            torch.gradient(u1, dim=3)[0] +
            torch.gradient(u2, dim=2)[0] +
            torch.gradient(u3, dim=1)[0]
        )  # (B, D, H, W)
        return div_u
    
    def compute_bkm_term(self, vorticity):
        """
        BKM 准则项: ||ω||_{L^∞} 相关的正则化
        
        BKM 准则: 如果 ∫_0^T ||ω(t)||_{L^∞} dt < ∞
        则解可以延展到时间 T
        
        损失项: 鼓励 ||ω||_{L^∞} 保持有界
        对于正则解: ω_max 应较小且平稳
        对于危险解: ω_max 应被检测并惩罚
        """
        omega_norm = torch.norm(vorticity, dim=1)  # (B, D, H, W)
        omega_max = omega_norm.flatten(1).max(dim=1)[0]  # (B,)
        
        # BKM 风险: 当 ω_max 接近阈值时增加惩罚
        # 使用 softplus 确保正值
        bkm_risk = torch.log1p(omega_max / self.bkm_threshold)
        
        return bkm_risk.mean()
    
    def compute_smoothness_term(self, vorticity):
        """
        平滑性项: ∇²ω 的范数 (向量化版本)
        
        对应正则性理论中涡量梯度的控制
        根据 BKM 准则和 Prodi-Serrin 条件，
        涡量场的梯度控制对于正则性至关重要
        """
        B, C = vorticity.shape[:2]
        # 向量化: 对所有通道和批次同时计算二阶梯度
        # laplacian = ∂²ω/∂x² + ∂²ω/∂y² + ∂²ω/∂z²
        # 计算二阶导数: gradient(gradient(field, dim), dim)
        dx2 = torch.gradient(torch.gradient(vorticity, dim=3)[0], dim=3)[0]  # ∂²/∂x²
        dy2 = torch.gradient(torch.gradient(vorticity, dim=2)[0], dim=2)[0]  # ∂²/∂y²
        dz2 = torch.gradient(torch.gradient(vorticity, dim=1)[0], dim=1)[0]  # ∂²/∂z²
        laplacian = dx2 + dy2 + dz2  # (B, C, D, H, W)
        smoothness = torch.mean(laplacian ** 2)
        return smoothness
    
    def compute_enstrophy_term(self, vorticity):
        """
        Enstrophy 项: 1/2 ∫ ||ω||² dx
        
        Enstrophy 的演化与能量耗散直接相关:
        d/dt (1/2||ω||²) = -ν||∇ω||² + (涡量拉伸项)
        
        正则解的 Enstrophy 应有合理的上界
        """
        omega_norm_sq = torch.norm(vorticity, dim=1) ** 2  # (B, D, H, W)
        enstrophy = 0.5 * torch.mean(omega_norm_sq)
        
        # 添加 Enstrophy 增长率估计
        # 这里简化为鼓励 Enstrophy 在合理范围
        enstrophy_penalty = enstrophy  # 鼓励 Enstrophy 较小
        return enstrophy_penalty
    
    def compute_prodi_serrin_term(self, vorticity):
        """
        [新增] Prodi-Serrin 条件损失项
        
        Prodi-Serrin 条件: 若 u ∈ L^p(0,T; L^q(ℝ³)) 且 2/p + 3/q ≤ 1, q > 3，
        则 NS 方程的弱解是光滑的。
        
        关键洞察: 3/q > 1 (即 q < 3) 是违反条件的区域。
        由于我们只有涡量场 ω 而非速度场 u，使用涡量推断速度的 L^q 范数。
        
        从 Helmholtz 分解，速度场的 L^q 范数与涡量场的空间变化相关：
        - u 的梯度与 ω 成正比
        - ||∇u||_{L^q} 可作为 u 的 L^q 可积性的代理
        
        损失项设计:
        - 计算 ∇ω 的 L^q 范数 (q=2, 作为标准情况)
        - 惩罚 ∇ω 的 L^q 范数过大（违反正则性条件）
        - 物理直觉: 高度不光滑的涡量场暗示违反 Prodi-Serrin 条件
        
        数学联系:
        - Escauriaza-Seregin-Šverák (2003) 证明了 L_{3,∞} 条件保证正则性
        - 当涡量场梯度很大时，速度场很可能不满足 L^p(L^q) 条件
        """
        # 计算涡量梯度 ∇ω
        omega_norm = torch.norm(vorticity, dim=1, keepdim=True)  # (B,1,D,H,W)
        
        # 方法: 使用涡量范数的梯度作为 ∇ω 的代理
        # ∂ω/∂x, ∂ω/∂y, ∂ω/∂z (对每个分量分别计算)
        d_omega = [
            torch.gradient(vorticity, dim=d)[0] for d in [1, 2, 3]
        ]  # 3个 (B, 3, D, H, W)
        
        # 合并梯度: ∇ω 的 Frobenius 范数
        grad_omega_norm = torch.sqrt(
            d_omega[0]**2 + d_omega[1]**2 + d_omega[2]**2 + 1e-8
        )  # (B, 3, D, H, W) 每个分量通道的梯度范数
        
        # 取沿通道平均: ||∇ω|| ≈ mean_c ||∂c_ω||
        # 这与能量耗散率中的 ||∇ω||² 直接相关
        grad_omega_mean = grad_omega_norm.mean(dim=1)  # (B, D, H, W)
        
        # 计算 L^q 代理 (q=2, 标准情况)
        # ||∇ω||_{L^2} 的空间均值
        grad_L2 = torch.mean(grad_omega_mean ** 2)  # scalar
        
        # Prodi-Serrin 惩罚:
        # 正则解应该使 ||∇ω|| 较小
        # 危险解中 ||∇ω|| 较大，对应 2/p + 3/q > 1 的区域
        # 使用 log(1 + ||∇ω||) 确保正值且增长适度
        ps_risk = torch.log1p(grad_L2)
        
        return ps_risk
    
    def compute_vorticity_stretching(self, velocity, vorticity):
        """
        涡量拉伸项: (ω·∇)u (向量化版本)
        
        这是 NS 方程中涡量传输方程的核心非线性项:
        Dω/Dt = (ω·∇)u + νΔω
        
        涡量拉伸是导致涡量增强的机制，
        与正则性问题直接相关
        """
        B, C, D, H, W = vorticity.shape
        # ω 的各分量
        omega = vorticity  # (B, 3, D, H, W)
        
        # 计算 ∇u 的各分量 (使用速度场梯度)
        # du_i/∂x_j: 对 velocity 的每个分量在每个方向上求导
        if velocity is not None:
            # ∂u1/∂x, ∂u1/∂y, ∂u1/∂z
            du1 = torch.stack([
                torch.gradient(velocity[:, 0], dim=d)[0] for d in [1, 2, 3]
            ], dim=1)  # (B, 3, D, H, W)
            # ∂u2/∂x, ∂u2/∂y, ∂u2/∂z
            du2 = torch.stack([
                torch.gradient(velocity[:, 1], dim=d)[0] for d in [1, 2, 3]
            ], dim=1)
            # ∂u3/∂x, ∂u3/∂y, ∂u3/∂z
            du3 = torch.stack([
                torch.gradient(velocity[:, 2], dim=d)[0] for d in [1, 2, 3]
            ], dim=1)
            # (ω·∇)u = ω1*∂u/∂x + ω2*∂u/∂y + ω3*∂u/∂z
            # 对于 u1 分量: ω1*∂u1/∂x + ω2*∂u1/∂y + ω3*∂u1/∂z
            omega_x = omega[:, 0:1]  # (B,1,D,H,W)
            omega_y = omega[:, 1:2]
            omega_z = omega[:, 2:3]
            
            stretch_u1 = (omega_x * du1[:, 0:1] + omega_y * du1[:, 1:2] + omega_z * du1[:, 2:3])
            stretch_u2 = (omega_x * du2[:, 0:1] + omega_y * du2[:, 1:2] + omega_z * du2[:, 2:3])
            stretch_u3 = (omega_x * du3[:, 0:1] + omega_y * du3[:, 1:2] + omega_z * du3[:, 2:3])
            stretching_vec = torch.sqrt(stretch_u1**2 + stretch_u2**2 + stretch_u3**2 + 1e-8)
        else:
            # 无速度场时，使用涡量梯度代理
            # 涡量拉伸与 ||∇ω|| 相关
            d_omega = torch.stack([
                torch.gradient(vorticity, dim=d)[0] for d in [1, 2, 3]
            ], dim=1)
            stretching_vec = torch.norm(d_omega, dim=1, keepdim=True)  # (B,1,D,H,W)
        
        # 拉伸强度: ||(ω·∇)u|| 的空间均值
        stretching = torch.mean(stretching_vec)
        return stretching
    
    def forward(self, velocity_pred, vorticity, vorticity_next=None):
        """
        综合物理损失计算
        
        Args:
            velocity_pred: (B, 3, D, H, W) 预测的速度场
            vorticity: (B, 3, D, H, W) 当前涡量场
            vorticity_next: (B, 3, D, H, W) 下一时刻涡量场（可选）
        
        Returns:
            total_loss: 标量总损失
            loss_dict: 各分量的字典
        """
        # 1. 不可压条件损失
        div_u = self.compute_divergence(velocity_pred)
        loss_div = torch.mean(div_u ** 2)
        
        # 2. BKM 准则损失
        loss_bkm = self.compute_bkm_term(vorticity)
        
        # 3. 平滑性损失
        loss_smooth = self.compute_smoothness_term(vorticity)
        
        # 4. Enstrophy 损失
        loss_enstrophy = self.compute_enstrophy_term(vorticity)
        
        # 5. 涡量拉伸损失
        loss_stretch = self.compute_vorticity_stretching(velocity_pred, vorticity)
        
        # 6. [新增] Prodi-Serrin 条件损失
        loss_ps = self.compute_prodi_serrin_term(vorticity)
        
        # 综合损失
        total_loss = (
            self.lambda_div * loss_div +
            self.lambda_bkm * loss_bkm +
            self.lambda_smooth * loss_smooth +
            self.lambda_enstrophy * loss_enstrophy +
            0.1 * loss_stretch +
            self.lambda_ps * loss_ps
        )
        
        return total_loss, {
            'divergence': loss_div.item(),
            'bkm': loss_bkm.item(),
            'smoothness': loss_smooth.item(),
            'enstrophy': loss_enstrophy.item(),
            'stretching': loss_stretch.item(),
            'prodi_serrin': loss_ps.item(),
            'total': total_loss.item()
        }


# ============================================================
# 训练数据集生成
# ============================================================

def generate_taylor_green_vortex(batch_size=4, size=64, re=100.0):
    """
    生成 Taylor-Green 涡旋 (已知的光滑 NS 解)
    
    Taylor-Green 涡旋是 NS 方程的精确解之一:
    u = (sin(x)cos(y)cos(z), -cos(x)sin(y)cos(z), 0) * A * exp(-2νkt)
    
    这是一个正则的流动，作为训练的正样本
    """
    x = torch.linspace(0, 2*torch.pi, size)
    y = torch.linspace(0, 2*torch.pi, size)
    z = torch.linspace(0, 2*torch.pi, size)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    
    amplitude = 1.0
    k = 1.0  # 波数
    
    u1 = amplitude * torch.sin(k*X) * torch.cos(k*Y) * torch.cos(k*Z)
    u2 = -amplitude * torch.cos(k*X) * torch.sin(k*Y) * torch.cos(k*Z)
    u3 = torch.zeros_like(X)
    
    velocity = torch.stack([u1, u2, u3], dim=0).unsqueeze(0)
    velocity = velocity.expand(batch_size, -1, -1, -1, -1)
    
    # 从速度场计算涡量
    vorticity = compute_vorticity(velocity)
    
    return velocity, vorticity


def generate_directional_flow(batch_size=4, size=64):
    """
    生成定向剪切流 (简单的正则流动)
    
    产生一个已知的正则速度场，作为训练数据
    """
    x = torch.linspace(-1, 1, size)
    y = torch.linspace(-1, 1, size)
    z = torch.linspace(-1, 1, size)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    
    # 简单的 Couette 流变体
    u1 = torch.sin(X) * torch.cos(Y) * torch.cos(Z) * 0.5
    u2 = -torch.cos(X) * torch.sin(Y) * torch.cos(Z) * 0.5
    u3 = 0.1 * torch.sin(Z)
    
    velocity = torch.stack([u1, u2, u3], dim=0).unsqueeze(0)
    velocity = velocity.expand(batch_size, -1, -1, -1, -1)
    
    vorticity = compute_vorticity(velocity)
    
    return velocity, vorticity


def generate_intensifying_vortex(batch_size=4, size=64, intensity=1.0):
    """
    生成增强涡旋 (潜在的接近奇异流动)
    
    关键修复 (2026-05-17): 构造满足 ∇·u = 0 的速度场。
    
    使用纯旋涡 (pure swirl) 构造:
    - 在柱坐标系中: u_θ = f(r), u_r = 0, u_z = 0
    - 这自动满足 ∇·u = 0 (因为 ∂(r·u_r)/∂r / r = 0 且 ∂u_z/∂z = 0)
    
    对于 3D 笛卡尔坐标，构造如下:
    - u(x,y,z) = (-y, x, 0) * g(r) * Gaussian envelope
    - 其中 g(r) 是径向剖面函数
    
    这与真实旋涡 (Lamb-Oseen) 的物理行为一致:
    - 速度切向于涡旋核
    - 无径向流动 (u_r = 0)
    - 自动不可压
    
    涡量方向: 仅在 z 方向
    ω_z = ∂v/∂x - ∂u/∂y = 2g(r) + O(r·g'(r))
    """
    x = torch.linspace(-2, 2, size)
    y = torch.linspace(-2, 2, size)
    z = torch.linspace(-2, 2, size)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    
    r = torch.sqrt(X**2 + Y**2 + 1e-6)
    
    # === 关键修复: 纯旋涡构造 (divergence-free) ===
    # 径向剖面: 使用 Gaussian 衰减
    # g(r) = intensity * exp(-intensity * r²)
    # 在 r=0 附近有最大切向速度
    g_r = intensity * torch.exp(-intensity * r**2)
    
    # 旋涡速度场 (切向, 自动满足 ∇·u = 0)
    # u(x,y) = (-y·g(r), x·g(r), 0)
    u1 = -Y * g_r          # 纯旋涡, ∇·u = 0
    u2 = X * g_r
    u3 = torch.zeros_like(X)
    
    # 添加轻微的 z 方向扰动 (模拟 3D 效应, 保持 ∇·u ≈ 0)
    # 使用 sin/cos 组合，不破坏散度条件
    u3_perturb = 0.02 * intensity * torch.sin(X) * torch.cos(Y) * torch.exp(-intensity * r**2 / 4)
    u3 = u3 + u3_perturb
    
    velocity = torch.stack([u1, u2, u3], dim=0).unsqueeze(0)
    velocity = velocity.expand(batch_size, -1, -1, -1, -1)
    
    vorticity = compute_vorticity(velocity)
    
    return velocity, vorticity


def generate_taylor_green_timesequence(batch_size=4, size=32, n_steps=10, nu=0.01, t_max=5.0):
    """
    [新增] 生成 Taylor-Green 涡旋时间序列
    
    Taylor-Green 涡旋的 NS 方程精确解:
    u(x,t) = u₀(x) · exp(-2νk²t)
    其中 k=1 是波数，u₀ 是初始速度场。
    
    涡量场的解析形式:
    ω(x,t) = ω₀(x) · exp(-2νk²t)
    其中 ω₀(x) = (-k cos(kx) sin(ky) sin(kz), k sin(kx) cos(ky) sin(kz), 0)
    
    参数:
        batch_size: 批量大小
        size: 空间网格分辨率
        n_steps: 时间步数
        nu: 运动粘度系数
        t_max: 最大时间
    
    返回:
        timesteps: (n_steps,) 时间点
        velocity_seq: (n_steps, B, 3, D, H, W) 速度序列
        vorticity_seq: (n_steps, B, 3, D, H, W) 涡量序列
        regularity_labels: (n_steps, B) 正则性标签 (随时间递减)
    
    用途:
        - 时间序列预测任务
        - 正则性丧失的动态演化分析
        - 作为训练 FNO/LSTM 时间预测模块的基础数据
    """
    x = torch.linspace(0, 2*torch.pi, size)
    y = torch.linspace(0, 2*torch.pi, size)
    z = torch.linspace(0, 2*torch.pi, size)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    
    k = 1.0  # 波数
    amplitude = 1.0
    
    # 初始速度场
    u1_0 = amplitude * torch.sin(k*X) * torch.cos(k*Y) * torch.cos(k*Z)
    u2_0 = -amplitude * torch.cos(k*X) * torch.sin(k*Y) * torch.cos(k*Z)
    u3_0 = torch.zeros_like(X)
    
    # 初始涡量场 (精确解析形式)
    omega1_0 = -k * torch.cos(k*X) * torch.sin(k*Y) * torch.sin(k*Z)
    omega2_0 = k * torch.sin(k*X) * torch.cos(k*Y) * torch.sin(k*Z)
    omega3_0 = torch.zeros_like(X)
    
    # 时间序列
    timesteps = torch.linspace(0, t_max, n_steps)
    
    velocity_seq = []
    vorticity_seq = []
    regularity_labels = []
    
    for t in timesteps:
        # 衰减因子
        decay = torch.exp(-2 * nu * k**2 * t)
        
        # 速度场 (batch_size 份副本)
        u1 = (u1_0 * decay).unsqueeze(0).expand(batch_size, -1, -1, -1)
        u2 = (u2_0 * decay).unsqueeze(0).expand(batch_size, -1, -1, -1)
        u3 = (u3_0 * decay).unsqueeze(0).expand(batch_size, -1, -1, -1)
        velocity = torch.stack([u1, u2, u3], dim=1)  # (B, 3, D, H, W)
        
        # 涡量场 (解析形式，保持精确)
        w1 = (omega1_0 * decay).unsqueeze(0).expand(batch_size, -1, -1, -1)
        w2 = (omega2_0 * decay).unsqueeze(0).expand(batch_size, -1, -1, -1)
        w3 = (omega3_0 * decay).unsqueeze(0).expand(batch_size, -1, -1, -1)
        vorticity = torch.stack([w1, w2, w3], dim=1)  # (B, 3, D, H, W)
        
        velocity_seq.append(velocity)
        vorticity_seq.append(vorticity)
        
        # 正则性标签: Taylor-Green 涡旋始终是正则的 (衰减解)
        # 但我们可以给出一个与时间相关的标签，表示"保持正则的可能性"
        # 衰减的涡量场始终是光滑的，所以 label 接近 1.0
        regularity_labels.append(torch.full((batch_size,), 0.98, dtype=torch.float32))
    
    velocity_seq = torch.stack(velocity_seq, dim=0)  # (n_steps, B, 3, D, H, W)
    vorticity_seq = torch.stack(vorticity_seq, dim=0)  # (n_steps, B, 3, D, H, W)
    regularity_labels = torch.stack(regularity_labels, dim=0)  # (n_steps, B)
    
    return timesteps, velocity_seq, vorticity_seq, regularity_labels


def generate_dangerous_timesequence(batch_size=4, size=32, n_steps=10, 
                                     intensity_growth=5.0, nu=0.01):
    """
    [更新 2026-05-17] 生成"危险"时间序列 (强度递增的涡旋)
    
    关键修复: intensity_growth 从默认值 2.0 提高到 5.0，
    使晚期 ω_max 可达到 1.0 以上，与静态危险样本有明显区分。
    
    模拟涡量强度随时间增长的流动，接近正则性丧失边界。
    初始为 Taylor-Green 型流动，但强度逐渐增强，
    模拟可能导致正则性问题的流动演化。
    
    强度增长曲线: I(t) = intensity_growth * t * exp(-νt/2)
    - t=1 时达到峰值: I_max ≈ intensity_growth * 0.61
    - 对纯旋涡: ω_max ≈ 2 * g(0) = 2 * intensity
    
    物理直觉: 
    - 纯旋涡 ω_z = ∂v/∂x - ∂u/∂y = 2g(r) + 2r·g'(r)
    - 在 r=0: ω_z ≈ 2·intensity
    - 当 intensity=5.0, ω_max ≈ 10.0 (远超边界阈值)
    
    参数:
        batch_size: 批量大小
        size: 空间网格分辨率
        n_steps: 时间步数
        intensity_growth: 强度增长率 (默认 5.0，使晚期 ω_max > 1.0)
        nu: 运动粘度系数
    
    返回:
        timesteps: (n_steps,) 时间点
        velocity_seq: (n_steps, B, 3, D, H, W) 速度序列 (divergence-free)
        vorticity_seq: (n_steps, B, 3, D, H, W) 涡量序列
        regularity_labels: (n_steps, B) 正则性标签 (随时间递减)
    
    正则性标签设计:
        - 早期 (t<0.3t_max): label ≈ 1.0 (正则)
        - 中期 (0.3t_max < t < 0.7t_max): label ≈ 0.5 (边界)
        - 晚期 (t > 0.7t_max): label ≈ 0.0 (危险)
    """
    x = torch.linspace(-2, 2, size)
    y = torch.linspace(-2, 2, size)
    z = torch.linspace(-2, 2, size)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    
    n_steps_actual = max(3, n_steps)
    timesteps = torch.linspace(0, 1, n_steps_actual)
    
    velocity_seq = []
    vorticity_seq = []
    regularity_labels = []
    
    for i, t in enumerate(timesteps):
        # 强度随时间增长 (但受粘度衰减抑制)
        intensity = intensity_growth * t * torch.exp(-nu * t * 0.5)
        
        # 双涡旋结构: Taylor-Green + 增强核心 (divergence-free)
        r = torch.sqrt(X**2 + Y**2 + 1e-6)
        
        # Taylor-Green 型背景流动 (精确满足 ∇·u = 0)
        u1_tg = torch.sin(X) * torch.cos(Y) * torch.cos(Z)
        u2_tg = -torch.cos(X) * torch.sin(Y) * torch.cos(Z)
        u3_tg = torch.zeros_like(X)
        
        # 增强涡核 (纯旋涡, 自动满足 ∇·u = 0)
        # 关键修复: 使用固定衰减率 (alpha=1.0), 不随 intensity 增大
        # 原公式 exp(-intensity*r²/4) 在 intensity=5 时会导致涡核过窄、幅度过小
        # 新公式 exp(-r²/4): 涡核宽度恒定，强度由 intensity 线性控制
        g_r = intensity * torch.exp(-r**2 / 4)
        u1_enh = -Y * g_r
        u2_enh = X * g_r
        u3_enh = 0.02 * intensity * torch.sin(X) * torch.cos(Y) * torch.exp(-r**2 / 16)
        
        # 混合
        u1 = u1_tg + u1_enh
        u2 = u2_tg + u2_enh
        u3 = u3_tg + u3_enh
        
        u1 = u1.unsqueeze(0).expand(batch_size, -1, -1, -1)
        u2 = u2.unsqueeze(0).expand(batch_size, -1, -1, -1)
        u3 = u3.unsqueeze(0).expand(batch_size, -1, -1, -1)
        velocity = torch.stack([u1, u2, u3], dim=1)  # (B, 3, D, H, W)
        
        vorticity = compute_vorticity(velocity)
        
        velocity_seq.append(velocity)
        vorticity_seq.append(vorticity)
        
        # 正则性标签: 随时间递减
        t_ratio = i / (n_steps_actual - 1)
        if t_ratio < 0.3:
            label = 1.0
        elif t_ratio < 0.7:
            label = 0.5
        else:
            label = 0.0
        regularity_labels.append(torch.full((batch_size,), label, dtype=torch.float32))
    
    velocity_seq = torch.stack(velocity_seq, dim=0)
    vorticity_seq = torch.stack(vorticity_seq, dim=0)
    regularity_labels = torch.stack(regularity_labels, dim=0)
    
    return timesteps, velocity_seq, vorticity_seq, regularity_labels


def create_time_series_dataset(n_sequences=50, size=32, n_steps=10):
    """
    [新增] 创建时间序列数据集
    
    用于训练时间感知的正则性预测模型。
    包含:
    - 正则序列: Taylor-Green 涡旋时间演化 (始终正则)
    - 危险序列: 强度递增涡旋 (正则性逐渐丧失)
    
    返回:
        vorticity_sequences: (N, n_steps, 3, D, H, W)
        regularity_labels: (N, n_steps)
    """
    all_sequences = []
    all_labels = []
    
    # 正则序列
    for _ in range(n_sequences // 2):
        _, _, vort_seq, labels = generate_taylor_green_timesequence(
            batch_size=2, size=size, n_steps=n_steps
        )
        for b in range(vort_seq.shape[1]):
            all_sequences.append(vort_seq[:, b])  # (n_steps, 3, D, H, W)
            all_labels.append(labels[:, b])  # (n_steps,)
    
    # 危险序列
    for _ in range(n_sequences // 2):
        _, _, vort_seq, labels = generate_dangerous_timesequence(
            batch_size=2, size=size, n_steps=n_steps,
            intensity_growth=1.5 + torch.rand(1).item()
        )
        for b in range(vort_seq.shape[1]):
            all_sequences.append(vort_seq[:, b])
            all_labels.append(labels[:, b])
    
    # 合并并打乱
    sequences = torch.stack(all_sequences)  # (N, n_steps, 3, D, H, W)
    labels = torch.stack(all_labels)  # (N, n_steps)
    
    indices = torch.randperm(len(sequences))
    sequences = sequences[indices]
    labels = labels[indices]
    
    return sequences, labels


def create_mixed_dataset(n_samples=200, size=32):
    """
    创建混合训练数据集
    
    包含:
    - Taylor-Green 涡旋 (正则样本, label=1)
    - 定向剪切流 (正则样本, label=1)  
    - 增强涡旋 (接近奇异, label=0)
    
    标签设计:
    - 1 = 正则解 (低涡量/光滑)
    - 0 = 潜在奇异 (高涡量/危险)
    """
    regular_samples = []
    regular_labels = []
    dangerous_samples = []
    dangerous_labels = []
    
    # 正则样本: Taylor-Green 涡旋
    for _ in range(n_samples // 4):
        _, omega = generate_taylor_green_vortex(batch_size=4, size=size)
        for i in range(omega.shape[0]):
            regular_samples.append(omega[i])
            regular_labels.append(1.0)  # 正则标签
    
    # 正则样本: 定向剪切流
    for _ in range(n_samples // 4):
        _, omega = generate_directional_flow(batch_size=4, size=size)
        for i in range(omega.shape[0]):
            regular_samples.append(omega[i])
            regular_labels.append(1.0)
    
    # 危险样本: 低强度增强涡旋 (接近边界)
    for _ in range(n_samples // 8):
        _, omega = generate_intensifying_vortex(batch_size=4, size=size, intensity=5.0)
        for i in range(omega.shape[0]):
            dangerous_samples.append(omega[i])
            dangerous_labels.append(0.5)  # 边界标签
    
    # 危险样本: 高强度增强涡旋 (危险)
    for _ in range(n_samples // 8):
        _, omega = generate_intensifying_vortex(batch_size=4, size=size, intensity=15.0)
        for i in range(omega.shape[0]):
            dangerous_samples.append(omega[i])
            dangerous_labels.append(0.0)  # 危险标签
    
    # 合并
    all_samples = regular_samples + dangerous_samples
    all_labels = regular_labels + dangerous_labels
    
    # 打乱
    indices = torch.randperm(len(all_samples))
    samples = torch.stack([all_samples[i] for i in indices])
    labels = torch.tensor([all_labels[i] for i in indices], dtype=torch.float32)
    
    return samples, labels


# ============================================================
# 训练循环
# ============================================================

def train_model(
    model,
    dataset=None,
    n_epochs=50,
    lr=1e-3,
    batch_size=16,
    device='cpu',
    lambda_phys=0.1,
    lambda_class=1.0,
):
    """
    训练涡量分析模型
    
    综合损失函数:
    L = λ_class * L_class + λ_phys * L_physics
    
    其中 L_class 是分类损失，L_physics 是物理一致性损失
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    physics_loss_fn = EnhancedPhysicsLoss()
    
    if dataset is None:
        X, y = create_mixed_dataset(n_samples=200, size=32)
        X, y = X.to(device), y.to(device)
    else:
        X, y = dataset
    
    model.train()
    history = {'epoch': [], 'class_loss': [], 'phys_loss': [], 'total_loss': [], 'acc': []}
    
    n_samples = X.shape[0]
    n_batches = max(1, n_samples // batch_size)
    
    print(f"开始训练: {n_epochs} epochs, {n_batches} batches/epoch")
    print("-" * 60)
    
    for epoch in range(n_epochs):
        epoch_class_loss = 0.0
        epoch_phys_loss = 0.0
        epoch_total_loss = 0.0
        correct = 0
        total = 0
        
        # 打乱数据
        perm = torch.randperm(n_samples)
        X_shuffled = X[perm]
        y_shuffled = y[perm]
        
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_samples)
            
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]
            
            # 分类损失
            pred = model(X_batch)  # (B,)
            class_loss = F.mse_loss(pred, y_batch)
            
            # 物理损失
            _, phys_losses = physics_loss_fn(
                velocity_pred=X_batch,  # 近似用涡量场代替
                vorticity=X_batch
            )
            phys_loss = torch.tensor(phys_losses['total'], device=device)
            
            # 总损失
            total_loss = lambda_class * class_loss + lambda_phys * phys_loss
            
            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 统计
            epoch_class_loss += class_loss.item()
            epoch_phys_loss += phys_loss.item()
            epoch_total_loss += total_loss.item()
            
            # 准确率 (二分类阈值 0.5)
            predicted = (pred > 0.5).float()
            correct += (predicted == y_batch).sum().item()
            total += y_batch.shape[0]
        
        # Epoch 统计
        n_batches_actual = max(1, n_batches)
        avg_class_loss = epoch_class_loss / n_batches_actual
        avg_phys_loss = epoch_phys_loss / n_batches_actual
        avg_total_loss = epoch_total_loss / n_batches_actual
        acc = correct / total * 100
        
        scheduler.step(avg_total_loss)
        
        history['epoch'].append(epoch + 1)
        history['class_loss'].append(avg_class_loss)
        history['phys_loss'].append(avg_phys_loss)
        history['total_loss'].append(avg_total_loss)
        history['acc'].append(acc)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            lr_current = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:3d}/{n_epochs} | "
                  f"Loss: {avg_total_loss:.4f} (cls={avg_class_loss:.4f}, phys={avg_phys_loss:.4f}) | "
                  f"Acc: {acc:.1f}% | LR: {lr_current:.6f}")
    
    print("-" * 60)
    print("训练完成!")
    
    return history


def evaluate_model(model, X_test, y_test, device='cpu'):
    """评估模型性能"""
    model.eval()
    with torch.no_grad():
        X_test = X_test.to(device)
        y_test = y_test.to(device)
        
        predictions = model(X_test)
        predictions_binary = (predictions > 0.5).float()
        
        accuracy = (predictions_binary == y_test).float().mean()
        
        # 计算各损失项
        physics_loss_fn = EnhancedPhysicsLoss()
        _, phys_losses = physics_loss_fn(X_test, X_test)
        
    print("\n" + "=" * 50)
    print("模型评估结果")
    print("=" * 50)
    print(f"准确率: {accuracy.item()*100:.2f}%")
    print(f"\n物理损失分项:")
    for k, v in phys_losses.items():
        print(f"  {k}: {v:.6f}")
    print("=" * 50)
    
    return accuracy.item(), phys_losses


# ============================================================
# 3D Fourier Neural Operator (FNO) - 涡量场预测
# ============================================================

class FourierLayer3D(nn.Module):
    """
    单层 3D Fourier 变换 + 频域乘法
    
    核心思想: 在频域中通过学习到的复数权重 R 进行参数化映射，
    保持 NS 方程的自然对称性 (平移不变性、周期性)。
    
    参考: Li et al. (2021) - Fourier Neural Operator
    """
    
    def __init__(self, in_channels, out_channels, modes=12):
        super().__init__()
        self.modes = modes
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 复数权重: [in_channels, out_channels, modes, modes, modes, 2]
        # 2 表示实部和虚部
        self.weight = nn.Parameter(
            torch.randn(in_channels, out_channels, modes, modes, modes, 2) * 0.02
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, in_channels, D, H, W)
        Returns:
            out: (B, out_channels, D, H, W)
        """
        B, C_in, D, H, W = x.shape
        
        # 3D 实时傅里叶变换 (real FFT)
        # 输出: (B, C, D, H, W//2+1) 复数
        x_ft = torch.fft.rfftn(x, dim=[2, 3, 4])
        
        D_ft = x_ft.shape[2]
        H_ft = x_ft.shape[3]
        W_ft = x_ft.shape[4]
        
        # 频域乘法
        # 截断高频模式: 保留 modes × modes × modes 的低频成分
        d_m = min(self.modes, D_ft)
        h_m = min(self.modes, H_ft)
        w_m = min(self.modes, W_ft)
        
        # 提取低频部分
        x_ft_mid = x_ft[:, :, :d_m, :h_m, :w_m]
        
        # 权重: (C_in, C_out, modes, modes, modes, 2)
        w_real = self.weight[:, :, :d_m, :h_m, :w_m, 0]  # (C_in, C_out, d_m, h_m, w_m)
        w_imag = self.weight[:, :, :d_m, :h_m, :w_m, 1]
        
        # 频域乘法: 对每个输入通道和输出通道组合
        # out_ft[c_out] = sum_c_in x_ft[c_in] * W[c_in, c_out]
        out_ft_mid = torch.zeros(
            B, self.out_channels, d_m, h_m, w_m,
            dtype=torch.complex64, device=x.device
        )
        
        for c_out in range(self.out_channels):
            for c_in in range(self.in_channels):
                # 复数乘法: z * w = (a+bi) * (c+di) = (ac-bd) + (ad+bc)i
                # 其中 z = x_ft_mid[:, c_in] (复数), w = w_real + i*w_imag (实数)
                x_c = x_ft_mid[:, c_in]  # ComplexFloat
                w_r = w_real[c_in, c_out]  # Float
                w_i = w_imag[c_in, c_out]  # Float
                # 实部和虚部
                real_part = x_c.real * w_r - x_c.imag * w_i
                imag_part = x_c.real * w_i + x_c.imag * w_r
                # 合并为复数张量
                combined = torch.stack([real_part, imag_part], dim=-1)  # (... , 2)
                combined_complex = torch.view_as_complex(combined)
                out_ft_mid[:, c_out] = out_ft_mid[:, c_out] + combined_complex
        
        # 填充高频区域 (置零)
        out_ft = torch.zeros(
            B, self.out_channels, D_ft, H_ft, W_ft,
            dtype=torch.complex64, device=x.device
        )
        out_ft[:, :, :d_m, :h_m, :w_m] = out_ft_mid
        
        # 逆 FFT
        out = torch.fft.irfftn(out_ft, s=[D, H, W], dim=[2, 3, 4])
        
        return out


class VorticityFNO3D(nn.Module):
    """
    3D Fourier Neural Operator for 涡量场演化预测
    
    输入: 单帧涡量场 ω(x,t) → [B, 3, D, H, W]
    输出: 下一时刻涡量场预测 ω(x,t+Δt) → [B, 3, D, H, W]
    
    架构:
    1. Linear lift: 3 → width 通道
    2. 4 层 Fourier 层 (保持周期性边界)
    3. 非线性激活 (GeLU)
    4. Linear project: width → 3 通道
    
    物理意义:
    - 频域权重编码了 NS 方程的线性算子 (扩散、传输)
    - 残差连接编码非线性 (涡量拉伸、对流)
    """
    
    def __init__(self, width=32, modes=8, n_layers=4):
        super().__init__()
        self.modes = modes
        self.width = width
        
        # 提升维度: 3 → width
        self.lift = nn.Conv3d(3, width, kernel_size=1)
        
        # Fourier 层堆叠
        self.fourier_layers = nn.ModuleList([
            FourierLayer3D(width, width, modes=modes)
            for _ in range(n_layers)
        ])
        
        # 批归一化
        self.bn = nn.ModuleList([
            nn.BatchNorm3d(width)
            for _ in range(n_layers)
        ])
        
        # 激活
        self.act = nn.GELU()
        
        # 降维: width → 3
        self.project = nn.Conv3d(width, 3, kernel_size=1)
        
        # 残差连接缩放因子
        self.alpha = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, omega_t):
        """
        Args:
            omega_t: (B, 3, D, H, W) 当前涡量场
        Returns:
            omega_pred: (B, 3, D, H, W) 下一时刻预测
        """
        x = self.lift(omega_t)  # (B, width, D, H, W)
        x = self.act(x)
        
        for fourier_layer, bn in zip(self.fourier_layers, self.bn):
            x_residual = x
            x = fourier_layer(x)
            x = bn(x)
            x = self.act(x)
            x = x + self.alpha * x_residual  # 残差连接
        
        delta_omega = self.project(x)  # (B, 3, D, H, W)
        omega_pred = omega_t + delta_omega  # 增量预测
        
        return omega_pred


# ============================================================
# 时序涡量分析器 (TemporalVorticityAnalyzer)
# ============================================================

class TemporalFeatureExtractor(nn.Module):
    """
    时序特征提取器: 对单帧涡量场提取空间特征
    
    输入: (B, 3, D, H, W) 涡量场
    输出: (B, feature_dim) 特征向量
    """
    
    def __init__(self, in_channels=3, hidden_dims=[16, 32, 64], feature_dim=64):
        super().__init__()
        
        layers = []
        for i, out_ch in enumerate(hidden_dims):
            layers.append(nn.Conv3d(
                in_channels if i == 0 else hidden_dims[i-1],
                out_ch,
                kernel_size=3,
                stride=2 if i > 0 else 1,
                padding=1
            ))
            layers.append(nn.BatchNorm3d(out_ch))
            layers.append(nn.ReLU(inplace=True))
        
        self.cnn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool3d(1)
        
        # 投影到统一维度
        self.proj = nn.Linear(hidden_dims[-1], feature_dim)
    
    def forward(self, omega):
        """
        Args:
            omega: (B, 3, D, H, W)
        Returns:
            features: (B, feature_dim)
        """
        x = self.cnn(omega)  # (B, C', D', H', W')
        x = self.pool(x).squeeze(-1).squeeze(-1).squeeze(-1)  # (B, C')
        x = self.proj(x)  # (B, feature_dim)
        return x


class TemporalLSTM(nn.Module):
    """
    LSTM 时序建模: 分析涡量场的时间演化模式
    
    输入: (B, T, feature_dim) 时间序列特征
    输出: (B, hidden_dim*2) 双向 LSTM 输出
    """
    
    def __init__(self, feature_dim=64, hidden_dim=32, num_layers=2):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        self.hidden_dim = hidden_dim
    
    def forward(self, features_seq):
        """
        Args:
            features_seq: (B, T, feature_dim)
        Returns:
            lstm_out: (B, T, hidden_dim*2)
            last_hidden: (B, hidden_dim*2) 最后一层隐藏状态
        """
        lstm_out, (h_n, c_n) = self.lstm(features_seq)
        
        # 双向最后隐藏状态拼接
        last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # (B, hidden_dim*2)
        
        return lstm_out, last_hidden


class TemporalAttentionPooling(nn.Module):
    """
    时序注意力池化: 加权聚合时间序列特征
    
    物理意义: 识别哪些时刻对正则性预测最重要
    - 正则样本: 注意力可能分散在所有时刻
    - 危险样本: 注意力可能集中在晚期 (t ≈ T)
    """
    
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, lstm_out):
        """
        Args:
            lstm_out: (B, T, hidden_dim)
        Returns:
            attended: (B, hidden_dim) 加权平均特征
            weights: (B, T) 注意力权重
        """
        # 计算注意力分数
        attn_scores = self.attention(lstm_out)  # (B, T, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (B, T, 1)
        
        # 加权平均
        attended = (lstm_out * attn_weights).sum(dim=1)  # (B, hidden_dim)
        
        return attended, attn_weights.squeeze(-1)


class TrendAnalyzer(nn.Module):
    """
    趋势分析器: 分析涡量场物理量的时间演化趋势
    
    计算:
    1. ω_max 的时间导数 (dω_max/dt)
    2. Q_max 的时间导数 (dQ_max/dt)  
    3. 趋势得分 = dω_max/dt / (ω_max + ε)
    
    物理意义:
    - 正趋势 (>0): 涡量正在增长，可能接近正则性边界
    - 负趋势 (<0): 涡量正在衰减，解保持正则
    - 大趋势 (>阈值): 可能发生正则性丧失
    """
    
    def __init__(self, hidden_dim=64):
        super().__init__()
        
        # 从序列特征预测趋势得分
        self.trend_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()  # 输出 [-1, 1]，-1=衰减, +1=增长
        )
        
        # 从物理量计算趋势
        self.growth_threshold = nn.Parameter(torch.tensor(0.1))
    
    def compute_physical_trend(self, omega_seq):
        """
        从涡量场序列计算物理趋势
        
        Args:
            omega_seq: (B, T, 3, D, H, W)
        Returns:
            trends: (B,) 平均趋势得分
        """
        B, T = omega_seq.shape[:2]
        
        omega_norms = torch.norm(omega_seq, dim=2)  # (B, T, D, H, W)
        omega_max_seq = omega_norms.flatten(2).max(dim=2)[0]  # (B, T)
        
        # 计算时间导数 (简单差分)
        omega_trend = omega_max_seq[:, 1:] - omega_max_seq[:, :-1]  # (B, T-1)
        
        # 归一化趋势
        omega_mean = omega_max_seq.mean(dim=1, keepdim=True) + 1e-6
        normalized_trend = (omega_trend / omega_mean).mean(dim=1)  # (B,)
        
        return normalized_trend
    
    def forward(self, lstm_out, omega_seq=None):
        """
        Args:
            lstm_out: (B, T, hidden_dim) LSTM 输出
            omega_seq: (B, T, 3, D, H, W) 可选的原始涡量序列
        Returns:
            trend_score: (B,) 趋势得分 [-1, 1]
        """
        # 从 LSTM 特征预测趋势
        feature_trend = self.trend_net(lstm_out.mean(dim=1)).squeeze(-1)  # (B,)
        
        # 如果有原始序列，结合物理趋势
        if omega_seq is not None:
            physical_trend = self.compute_physical_trend(omega_seq)  # (B,)
            # 融合: 60% 特征预测 + 40% 物理计算
            trend_score = 0.6 * feature_trend + 0.4 * torch.tanh(physical_trend)
        else:
            trend_score = feature_trend
        
        return trend_score


class RegularityPredictionHead(nn.Module):
    """
    正则性预测头: 结合静态特征、时序特征和趋势分析
    
    输入:
    - static_features: (B, feature_dim) 当前时刻静态特征
    - temporal_features: (B, hidden_dim*2) 时序 LSTM 特征
    - trend_score: (B,) 趋势得分 [-1, 1]
    
    输出:
    - regularity_score: (B,) 正则性评分 [0, 1]
    """
    
    def __init__(self, static_dim=64, temporal_dim=64, output_dim=1):
        super().__init__()
        
        total_dim = static_dim + temporal_dim + 1  # +1 for trend
        
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, static_features, temporal_features, trend_score):
        """
        Args:
            static_features: (B, static_dim)
            temporal_features: (B, temporal_dim)
            trend_score: (B,)
        Returns:
            regularity_score: (B,)
        """
        # 扩展 trend_score 到维度匹配
        trend_expanded = trend_score.unsqueeze(-1)  # (B, 1)
        
        combined = torch.cat([static_features, temporal_features, trend_expanded], dim=-1)
        return self.fusion(combined).squeeze(-1)


class TemporalVorticityAnalyzer(nn.Module):
    """
    时序涡量分析器: 端到端正则性预测系统
    
    架构:
    1. 空间特征提取 (VorticityEncoder)
    2. 时序建模 (TemporalLSTM)
    3. 趋势分析 (TrendAnalyzer)
    4. 正则性预测 (RegularityPredictionHead)
    
    功能:
    - 静态正则性预测: 基于当前涡量场
    - 动态正则性预测: 结合历史时间序列
    - 趋势预警: 检测涡量场增长趋势
    """
    
    def __init__(self, 
                 encoder_dims=[32, 64, 128, 256],
                 attention_heads=8,
                 temporal_feature_dim=64,
                 lstm_hidden_dim=32,
                 lstm_num_layers=2):
        super().__init__()
        
        # 空间特征提取 (复用 VorticityEncoder)
        self.encoder = VorticityEncoder(in_channels=3, hidden_dims=encoder_dims)
        self.feat_dim = encoder_dims[-1]
        
        # 注意力 (可选,用于特征增强)
        self.attention = PhysicsAttention(self.feat_dim, attention_heads)
        
        # 时序特征提取器 (用于为 LSTM 准备特征)
        self.temporal_feature_extractor = TemporalFeatureExtractor(
            in_channels=3,
            hidden_dims=[16, 32, 64],
            feature_dim=temporal_feature_dim
        )
        
        # LSTM 时序建模
        self.lstm = TemporalLSTM(
            feature_dim=temporal_feature_dim,
            hidden_dim=lstm_hidden_dim,
            num_layers=lstm_num_layers
        )
        
        # 趋势分析
        self.trend_analyzer = TrendAnalyzer(hidden_dim=lstm_hidden_dim*2)
        
        # 正则性预测头
        self.regularity_head = RegularityPredictionHead(
            static_dim=self.feat_dim,  # VorticityEncoder 输出维度
            temporal_dim=lstm_hidden_dim*2,  # 双向 LSTM
            output_dim=1
        )
        
        self.temporal_feature_dim = temporal_feature_dim
        self.lstm_hidden_dim = lstm_hidden_dim
    
    def forward(self, vorticity_sequence):
        """
        前向传播
        
        Args:
            vorticity_sequence: (B, T, 3, D, H, W) 涡量场时间序列
            或者单帧: (B, 3, D, H, W)
        
        Returns:
            regularity_score: (B,) 正则性评分 [0, 1]
            temporal_features: (B, hidden_dim*2) 时序特征 (可选)
            trend_score: (B,) 趋势得分 (可选)
        """
        # 处理单帧输入
        if vorticity_sequence.dim() == 4:
            vorticity_sequence = vorticity_sequence.unsqueeze(1)  # (B, 1, 3, D, H, W)
        
        B, T = vorticity_sequence.shape[:2]
        
        # 1. 空间特征提取
        spatial_features = []
        for t in range(T):
            omega_t = vorticity_sequence[:, t]  # (B, 3, D, H, W)
            
            # 使用 VorticityEncoder
            feat_t = self.encoder(omega_t)  # (B, feat_dim, D', H', W')
            feat_t = feat_t.mean(dim=[2, 3, 4])  # (B, feat_dim)
            spatial_features.append(feat_t)
        
        spatial_features = torch.stack(spatial_features, dim=1)  # (B, T, feat_dim)
        
        # 当前时刻的静态特征
        current_features = spatial_features[:, -1]  # (B, feat_dim)
        
        # 2. 时序特征提取 (从原始涡量场)
        temporal_features_seq = []
        for t in range(T):
            omega_t = vorticity_sequence[:, t]
            feat_t = self.temporal_feature_extractor(omega_t)  # (B, temporal_feature_dim)
            temporal_features_seq.append(feat_t)
        
        temporal_features_seq = torch.stack(temporal_features_seq, dim=1)  # (B, T, temporal_feature_dim)
        
        # 3. LSTM 时序建模
        lstm_out, last_hidden = self.lstm(temporal_features_seq)  # (B, T, hidden*2), (B, hidden*2)
        
        # 注意力池化
        attention_pooled, attn_weights = TemporalAttentionPooling(
            hidden_dim=self.lstm_hidden_dim*2
        )(lstm_out)
        
        # 4. 趋势分析
        trend_score = self.trend_analyzer(lstm_out, vorticity_sequence)  # (B,)
        
        # 5. 正则性预测
        regularity_score = self.regularity_head(
            current_features,  # 静态特征
            attention_pooled,   # 时序特征
            trend_score         # 趋势得分
        )
        
        return regularity_score, last_hidden, trend_score
    
    def predict_with_uncertainty(self, vorticity_sequence, n_samples=10):
        """
        带不确定性的预测 (Monte Carlo Dropout)
        
        Args:
            vorticity_sequence: (B, T, 3, D, H, W)
            n_samples: MC 采样次数
        
        Returns:
            mean_score: (B,) 平均预测
            std_score: (B,) 预测标准差 (不确定性估计)
        """
        self.train()  # 开启 dropout
        
        scores = []
        for _ in range(n_samples):
            score, _, _ = self.forward(vorticity_sequence)
            scores.append(score)
        
        scores = torch.stack(scores, dim=0)  # (n_samples, B)
        
        self.eval()
        
        mean_score = scores.mean(dim=0)
        std_score = scores.std(dim=0)
        
        return mean_score, std_score


def compute_temporal_metrics(omega_seq, predictions, labels):
    """
    计算时序预测的额外指标
    
    Args:
        omega_seq: (T, B, 3, D, H, W) 涡量序列
        predictions: (T, B) 预测值
        labels: (T, B) 标签
    
    Returns:
        metrics: dict 时序指标
    """
    T, B = predictions.shape
    
    # 1. 时间一致性: 预测是否随时间单调
    consistency_scores = []
    for b in range(B):
        pred_b = predictions[:, b]
        # 计算相邻预测的差异
        diff = pred_b[1:] - pred_b[:-1]
        # 正则样本应该递减 (从 1→低值 或 保持高值)
        # 危险样本应该递减 (从高值→0)
        # 检查是否单调
        is_monotonic = (diff.abs().sum() < 1e-6) or (
            (diff > 0).sum() < T//4 or (diff < 0).sum() < T//4
        )
        consistency_scores.append(is_monotonic)
    
    # 2. 预警准确性: 危险样本是否在正确时间点发出预警
    early_warnings = []
    for b in range(B):
        label_b = labels[:, b]
        pred_b = predictions[:, b]
        
        # 找到第一个预测为"危险"的时间点
        danger_threshold = 0.3
        first_danger_pred = None
        first_danger_label = None
        
        for t in range(T):
            if pred_b[t] < danger_threshold:
                first_danger_pred = t
                break
        
        for t in range(T):
            if label_b[t] == 0.0:
                first_danger_label = t
                break
        
        if first_danger_pred is not None and first_danger_label is not None:
            # 计算预警提前量 (负数=延迟预警, 正数=提前预警)
            warning_lead = first_danger_label - first_danger_pred
            early_warnings.append(warning_lead)
        else:
            early_warnings.append(0)
    
    metrics = {
        'temporal_consistency': sum(consistency_scores) / B,
        'avg_warning_lead': sum(early_warnings) / B if early_warnings else 0,
        'early_warning_rate': sum(1 for w in early_warnings if w >= 0) / B if early_warnings else 0,
    }
    
    return metrics


# ============================================================
# 训练工具: 时序模型
# ============================================================

def train_temporal_model(
    model,
    train_sequences,
    train_labels,
    n_epochs=30,
    lr=1e-3,
    batch_size=8,
    device='cpu',
    lambda_trend=0.1,
    val_sequences=None,
    val_labels=None
):
    """
    训练时序涡量分析器
    
    综合损失函数:
    L = L_class + λ_trend * L_trend + λ_consistency * L_consistency
    
    其中:
    - L_class: 分类损失 (MSE)
    - L_trend: 趋势损失 (鼓励趋势与实际物理一致)
    - L_consistency: 一致性损失 (鼓励预测随时间平滑变化)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    criterion = nn.MSELoss()
    
    model = model.to(device)
    model.train()
    
    n_sequences = len(train_sequences)
    n_batches = max(1, n_sequences // batch_size)
    
    history = {
        'epoch': [], 'loss': [], 'class_loss': [], 'trend_loss': [],
        'acc': [], 'val_acc': []
    }
    
    print(f"开始时序训练: {n_sequences} 序列, {n_epochs} epochs")
    print("-" * 60)
    
    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0.0
        epoch_cls_loss = 0.0
        epoch_trend_loss = 0.0
        correct = 0
        total = 0
        
        # 打乱数据
        indices = torch.randperm(n_sequences)
        
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_sequences)
            batch_indices = indices[start:end]
            
            # 准备批次数据
            batch_seq = torch.stack([train_sequences[i] for i in batch_indices]).to(device)
            batch_labels = torch.stack([train_labels[i] for i in batch_indices]).float().to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            predictions, _, trend_scores = model(batch_seq)  # (B,), (B, hidden*2), (B,)
            
            # 分类损失
            # 使用最后一个时间步的标签
            target = batch_labels[:, -1]  # (B,)
            class_loss = criterion(predictions, target)
            
            # 趋势损失: 鼓励趋势与 ω_max 变化一致
            # 计算实际的涡量趋势
            omega_norms = torch.norm(batch_seq, dim=2)  # (B, T, D, H, W)
            omega_max_seq = omega_norms.flatten(2).max(dim=2)[0]  # (B, T)
            actual_trend = (omega_max_seq[:, -1] - omega_max_seq[:, 0]) / (omega_max_seq.mean(dim=1) + 1e-6)
            actual_trend = torch.tanh(actual_trend)  # 归一化到 [-1, 1]
            
            # 预测趋势应该与实际趋势相关
            trend_loss = criterion(trend_scores, actual_trend)
            
            # 一致性损失: 鼓励相邻时间步预测平滑变化
            pred_diff = predictions[1:] - predictions[:-1]
            consistency_loss = (pred_diff ** 2).mean()
            
            # 总损失
            total_loss = class_loss + lambda_trend * trend_loss + 0.05 * consistency_loss
            
            # 反向传播
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += total_loss.item()
            epoch_cls_loss += class_loss.item()
            epoch_trend_loss += trend_loss.item()
            
            # 统计
            pred_binary = (predictions > 0.5).float()
            correct += (pred_binary == target).sum().item()
            total += target.shape[0]
        
        scheduler.step(epoch_loss / n_batches)
        
        # 评估
        acc = correct / total * 100
        
        # 验证集 (如果有)
        val_acc = None
        if val_sequences is not None:
            model.eval()
            with torch.no_grad():
                val_seq = torch.stack(val_sequences).to(device)
                val_label = torch.stack(val_labels).float().to(device)
                val_pred, _, _ = model(val_seq)
                val_target = val_label[:, -1]
                val_binary = (val_pred > 0.5).float()
                val_acc = (val_binary == val_target).float().mean().item() * 100
            model.train()
        
        history['epoch'].append(epoch)
        history['loss'].append(epoch_loss / n_batches)
        history['class_loss'].append(epoch_cls_loss / n_batches)
        history['trend_loss'].append(epoch_trend_loss / n_batches)
        history['acc'].append(acc)
        history['val_acc'].append(val_acc if val_acc else 0)
        
        if epoch % 5 == 0 or epoch == 1:
            lr_current = optimizer.param_groups[0]['lr']
            val_str = f", val_acc={val_acc:.1f}%" if val_acc else ""
            print(f"Epoch {epoch:2d}: loss={epoch_loss/n_batches:.4f}, "
                  f"cls={epoch_cls_loss/n_batches:.4f}, acc={acc:.1f}%{val_str}, "
                  f"lr={lr_current:.2e}")
    
    print("-" * 60)
    print("时序训练完成!")
    
    return history


def create_time_series_training_data(n_sequences=100, n_steps=8, size=24):
    """
    创建时序训练数据集
    
    包含:
    - Taylor-Green 时间序列 (正则)
    - 危险时间序列 (强度递增)
    
    Returns:
        sequences: list of (T, 3, D, H, W) 涡量序列
        labels: list of (T,) 时间序列标签
    """
    sequences = []
    labels = []
    
    # 正则序列: Taylor-Green (衰减)
    for _ in range(n_sequences // 2):
        timesteps, _, vort_seq, regularity = generate_taylor_green_timesequence(
            batch_size=1, size=size, n_steps=n_steps, t_max=5.0
        )
        sequences.append(vort_seq.squeeze(1))  # (T, 3, D, H, W)
        labels.append(regularity.squeeze(1))   # (T,)
    
    # 危险序列: 强度递增
    for _ in range(n_sequences // 2):
        timesteps, _, vort_seq, regularity = generate_dangerous_timesequence(
            batch_size=1, size=size, n_steps=n_steps,
            intensity_growth=3.0 + np.random.rand() * 2  # 随机强度 3-5
        )
        sequences.append(vort_seq.squeeze(1))
        labels.append(regularity.squeeze(1))
    
    # 打乱
    indices = torch.randperm(len(sequences))
    sequences = [sequences[i] for i in indices]
    labels = [labels[i] for i in indices]
    
    return sequences, labels


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 70)
    print("Navier-Stokes 涡量分析器 - 测试")
    print("=" * 70)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    # 测试 1: 基础 VorticityAnalyzer
    print("\n" + "=" * 70)
    print("测试 1: VorticityAnalyzer (基础模型)")
    print("=" * 70)
    
    model = VorticityAnalyzer()
    model = model.to(device)
    
    batch_size = 4
    omega = create_synthetic_vorticity(batch_size, size=32).to(device)
    
    print(f"输入涡量场形状: {omega.shape}")
    
    with torch.no_grad():
        regularity_score = model(omega)
    
    print(f"正则性评分: {regularity_score}")
    print(f"预测: {['可能奇异' if s < 0.5 else '正则' for s in regularity_score]}")
    
    # 统计量
    stats = compute_vorticity_stats(omega)
    print(f"\n涡量场统计:")
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}")
    
    # 测试增强物理损失
    print("\n" + "-" * 60)
    print("测试 EnhancedPhysicsLoss:")
    physics_loss_fn = EnhancedPhysicsLoss()
    _, loss_dict = physics_loss_fn(omega, omega)
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.6f}")
    
    # 测试数据集生成
    print("\n" + "-" * 60)
    print("测试数据集生成:")
    
    # Taylor-Green 涡旋
    u_tg, omega_tg = generate_taylor_green_vortex(batch_size=2, size=16)
    print(f"  Taylor-Green 涡旋: u {u_tg.shape}, ω {omega_tg.shape}")
    stats_tg = compute_vorticity_stats(omega_tg)
    print(f"    ω_max: {stats_tg['max']:.4f}, ω_mean: {stats_tg['mean']:.4f}")
    
    # 增强涡旋
    u_intense, omega_intense = generate_intensifying_vortex(batch_size=2, size=16, intensity=15.0)
    stats_intense = compute_vorticity_stats(omega_intense)
    print(f"  增强涡旋 (intensity=15): ω_max: {stats_intense['max']:.4f}, ω_mean: {stats_intense['mean']:.4f}")
    
    # 参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nVorticityAnalyzer 参数总量: {total_params:,}")
    
    # 测试 FNO
    print("\n" + "=" * 70)
    print("测试 2: VorticityFNO3D (涡量场预测)")
    print("=" * 70)
    
    fno = VorticityFNO3D(width=32, modes=8, n_layers=3)
    fno_params = sum(p.numel() for p in fno.parameters())
    print(f"  FNO 参数量: {fno_params:,}")
    
    with torch.no_grad():
        omega_fno = fno(omega[:, :, :16, :16, :16])
    print(f"  FNO 输出形状: {omega_fno.shape}")
    
    # 测试时间序列
    print("\n" + "=" * 70)
    print("测试 3: 时间序列数据生成")
    print("=" * 70)
    
    # Taylor-Green 时间序列
    timesteps, _, vort_seq_tg, labels_tg = generate_taylor_green_timesequence(
        batch_size=1, size=16, n_steps=6, t_max=3.0
    )
    print(f"  Taylor-Green 时间序列: {vort_seq_tg.shape}")
    print(f"  涡量衰减: {vort_seq_tg.max(dim=1)[0].max(dim=1)[0].max(dim=1)[0]}")
    
    # 危险时间序列
    timesteps, _, vort_seq_danger, labels_danger = generate_dangerous_timesequence(
        batch_size=1, size=16, n_steps=6, intensity_growth=4.0
    )
    print(f"  危险时间序列: {vort_seq_danger.shape}")
    print(f"  涡量增长: {vort_seq_danger.max(dim=1)[0].max(dim=1)[0].max(dim=1)[0]}")
    
    # 测试 TemporalVorticityAnalyzer
    print("\n" + "=" * 70)
    print("测试 4: TemporalVorticityAnalyzer (时序模型)")
    print("=" * 70)
    
    temporal_model = TemporalVorticityAnalyzer(
        encoder_dims=[32, 64, 128],
        attention_heads=4,
        temporal_feature_dim=32,
        lstm_hidden_dim=24,
        lstm_num_layers=2
    )
    temporal_params = sum(p.numel() for p in temporal_model.parameters())
    print(f"  时序模型参数量: {temporal_params:,}")
    
    # 测试时序前向传播
    # 创建时间序列输入: (B, T, 3, D, H, W)
    seq_len = 5
    time_series_input = torch.randn(2, seq_len, 3, 16, 16, 16)
    
    with torch.no_grad():
        regularity, temporal_feat, trend = temporal_model(time_series_input)
    
    print(f"  输入形状: {time_series_input.shape}")
    print(f"  正则性评分: {regularity}")
    print(f"  时序特征形状: {temporal_feat.shape}")
    print(f"  趋势得分: {trend}")
    
    # 测试单帧输入
    single_frame = torch.randn(2, 3, 16, 16, 16)
    with torch.no_grad():
        reg_single, _, trend_single = temporal_model(single_frame)
    print(f"\n  单帧输入测试:")
    print(f"    输入形状: {single_frame.shape}")
    print(f"    正则性评分: {reg_single}")
    
    # 测试时序数据集创建
    print("\n" + "-" * 60)
    print("测试时序训练数据生成:")
    seq_list, label_list = create_time_series_training_data(
        n_sequences=20, n_steps=6, size=16
    )
    print(f"  生成 {len(seq_list)} 个时间序列")
    print(f"  第一个序列形状: {seq_list[0].shape}")
    print(f"  第一个标签形状: {label_list[0].shape}")
    
    # 模型对比
    print("\n" + "=" * 70)
    print("模型参数量对比")
    print("=" * 70)
    print(f"  VorticityAnalyzer:      {total_params:>12,} 参数")
    print(f"  VorticityFNO3D:         {fno_params:>12,} 参数")
    print(f"  TemporalVorticityAnalyzer: {temporal_params:>12,} 参数")
    
    print("\n" + "=" * 70)
    print("测试完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
