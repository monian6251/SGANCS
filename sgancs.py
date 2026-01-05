"""
SGANCS: Structure-Guided Anisotropic Non-Convex Smoothing

基于 Wei Liu 等人论文的改进算法实现
"A generalized framework for edge-preserving and structure-preserving image smoothing"
IEEE TPAMI, 2021
"""

import numpy as np
import cv2
from scipy.fft import fft2, ifft2


class StructureAnalyzer:
    """
    第一阶段：分析图像结构，生成自适应参数和方向场
    
    通过结构张量分析图像局部方向信息：
    - v1: 梯度方向（跨边缘）
    - v2: 切线方向（沿边缘）
    """
    
    def __init__(self, image):
        """
        Args:
            image: 输入图像，BGR 格式，uint8 或 float32
        """
        # 确保输入是 float32, 范围 [0, 1]
        self.img = image.astype(np.float32) / 255.0 if image.max() > 1.0 else image.astype(np.float32)
        if len(self.img.shape) == 3:
            # 结构张量基于灰度图计算，但应用于所有通道
            self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        else:
            self.gray = self.img
        
        self.rows, self.cols = self.gray.shape
        self.v1 = None  # 梯度方向 (跨边缘)
        self.v2 = None  # 切线方向 (沿边缘)
        self.mu1 = None  # 大特征值
        self.mu2 = None  # 小特征值

    def compute_structure_tensor(self, sigma_grad=1.0, sigma_tensor=2.0):
        """
        计算结构张量及其特征分解
        S = Gaussian * (grad * grad.T)
        
        Args:
            sigma_grad: 梯度计算的高斯尺度（未使用，保留接口）
            sigma_tensor: 张量平滑的积分尺度，越大方向估计越鲁棒
            
        Returns:
            (mu1, mu2): 特征值，mu1 >= mu2
        """
        # 1. 计算梯度
        dx = cv2.Sobel(self.gray, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(self.gray, cv2.CV_32F, 0, 1, ksize=3)

        # 2. 构建张量积
        Ixx = dx * dx
        Ixy = dx * dy
        Iyy = dy * dy

        # 3. 高斯平滑张量分量 (积分尺度)
        ksize = int(2 * round(3 * sigma_tensor) + 1)
        Sxx = cv2.GaussianBlur(Ixx, (ksize, ksize), sigma_tensor)
        Sxy = cv2.GaussianBlur(Ixy, (ksize, ksize), sigma_tensor)
        Syy = cv2.GaussianBlur(Iyy, (ksize, ksize), sigma_tensor)

        # 4. 特征分解 (2x2 矩阵解析解)
        trace = Sxx + Syy
        det = Sxx * Syy - Sxy**2
        
        # Eigenvalues: mu1 >= mu2
        delta = np.sqrt(np.maximum(0, (trace/2)**2 - det))
        self.mu1 = trace/2 + delta
        self.mu2 = trace/2 - delta

        # Eigenvectors
        # v1 对应 mu1 (主要梯度方向/跨边缘)
        # v2 对应 mu2 (切线方向/沿边缘)
        vx = Sxx - self.mu2
        vy = Sxy
        
        # 归一化，处理零向量情况
        mag = np.sqrt(vx**2 + vy**2) + 1e-6
        vx /= mag
        vy /= mag
        
        # 保存方向场 (H, W, 2)
        self.v1 = np.stack((vx, vy), axis=-1)
        self.v2 = np.stack((-vy, vx), axis=-1)  # v2 与 v1 垂直

        return self.mu1, self.mu2

    def generate_adaptive_weights(self, base_lambda=0.01, texture_suppression=True):
        """
        基于特征值生成自适应权重图
        
        Args:
            base_lambda: 基础平滑强度
            texture_suppression: 是否在纹理区加强平滑
            
        Returns:
            (lambda1, lambda2): 两个方向的自适应权重图
        """
        # 归一化特征值
        max_val = self.mu1.max() + 1e-6
        mu1_norm = self.mu1 / max_val
        mu2_norm = self.mu2 / max_val
        
        # Coherence (相干性): 描述是否是线状结构
        coherence = mu1_norm - mu2_norm
        
        # --- 权重策略 ---
        
        # 1. 跨边缘方向: 强边缘保留，平坦区平滑
        weight_map_1 = 1.0 / (1.0 + 10.0 * coherence) 
        lambda_1 = base_lambda * weight_map_1
        
        # 2. 沿边缘方向: 通常都允许平滑以连接断裂
        weight_map_2 = np.ones_like(coherence)
        if texture_suppression:
            # 在纹理区加大平滑力度
            weight_map_2 = 1.0 + 5.0 * mu2_norm 
        
        lambda_2 = base_lambda * 2.0 * weight_map_2
        
        return lambda_1, lambda_2


class SGANCS_Solver:
    """
    第二阶段：ADMM 求解器
    
    使用 ADMM 框架求解各向异性非凸优化问题，
    通过 FFT 快速求解 u-子问题，保持 O(N log N) 复杂度。
    """
    
    def __init__(self, image, v1, v2, lambda1_map, lambda2_map, rho=0.1):
        """
        Args:
            image: 输入图像
            v1: 梯度方向场 (H, W, 2)
            v2: 切线方向场 (H, W, 2)
            lambda1_map: 跨边缘方向权重图
            lambda2_map: 沿边缘方向权重图
            rho: ADMM 惩罚参数
        """
        self.f = image.astype(np.float32) / 255.0 if image.max() > 1.0 else image.astype(np.float32)
        self.v1 = v1
        self.v2 = v2
        self.L1 = lambda1_map
        self.L2 = lambda2_map
        self.rho = rho
        
        self.rows, self.cols = self.f.shape[:2]
        self.channels = 1 if len(self.f.shape) == 2 else self.f.shape[2]
        
        # 预计算 FFT 算子 (u-subproblem)
        self.otf_denominator = self._precompute_fft_operator()

    def _psf2otf(self, psf, shape):
        """Matlab psf2otf 的 Python 实现"""
        in_size = psf.shape
        out_size = shape
        pad_size = (out_size[0] - in_size[0], out_size[1] - in_size[1])
        psf_padded = np.pad(psf, ((0, pad_size[0]), (0, pad_size[1])), 'constant')
        
        # 循环移位，使中心在 (0,0)
        for i in range(2):
            psf_padded = np.roll(psf_padded, -int(in_size[i] / 2), axis=i)
            
        return fft2(psf_padded)

    def _precompute_fft_operator(self):
        """预计算拉普拉斯算子的 FFT"""
        laplacian_kernel = np.array([[0, -1, 0], 
                                     [-1, 4, -1], 
                                     [0, -1, 0]], dtype=np.float32)
        
        otf_lap = self._psf2otf(laplacian_kernel, (self.rows, self.cols))
        return 1.0 + self.rho * otf_lap

    def _compute_gradient(self, img):
        """周期性边界条件的前向差分"""
        grad_x = np.roll(img, -1, axis=1) - img
        grad_y = np.roll(img, -1, axis=0) - img
        return grad_x, grad_y

    def _compute_divergence(self, vx, vy):
        """后向差分计算散度"""
        div_x = vx - np.roll(vx, 1, axis=1)
        div_y = vy - np.roll(vy, 1, axis=0)
        return div_x + div_y

    def _solve_prox_L0(self, q, threshold):
        """
        L0 范数的 Proximal Operator (Hard Thresholding)
        z = q if |q|^2 > 2*lambda/rho else 0
        """
        mask = (q**2) > (2 * threshold)
        return q * mask

    def _solve_prox_L2(self, q, threshold):
        """
        L2 范数的 Proximal Operator (Tikhonov)
        min lambda*z^2 + rho/2(z-q)^2 -> z = q / (1 + 2*lambda/rho)
        """
        return q / (1.0 + 2.0 * threshold)

    def solve(self, iterations=20, verbose=True):
        """
        执行 ADMM 优化
        
        Args:
            iterations: 迭代次数
            verbose: 是否打印收敛信息
            
        Returns:
            优化后的图像 (uint8)
        """
        u = self.f.copy()
        
        # 处理多通道
        if self.channels > 1:
            u_out = np.zeros_like(u)
            for c in range(self.channels):
                u_out[:,:,c] = self._solve_single_channel(u[:,:,c], self.f[:,:,c], iterations, verbose)
            return u_out
        else:
            return self._solve_single_channel(u, self.f, iterations, verbose)

    def _solve_single_channel(self, u_init, f_channel, iterations, verbose):
        """单通道 ADMM 求解"""
        u = u_init
        
        # 辅助变量 z1, z2
        z1 = np.zeros((self.rows, self.cols), dtype=np.float32)
        z2 = np.zeros((self.rows, self.cols), dtype=np.float32)
        
        # 对偶变量 eta1, eta2
        eta1 = np.zeros_like(z1)
        eta2 = np.zeros_like(z2)
        
        for i in range(iterations):
            # --- 1. Z-update (Non-convex Projection) ---
            gx, gy = self._compute_gradient(u)
            
            # 投影到结构方向
            grad_v1 = gx * self.v1[...,0] + gy * self.v1[...,1]
            grad_v2 = gx * self.v2[...,0] + gy * self.v2[...,1]
            
            # 构造 Prox 输入
            q1 = grad_v1 + eta1 / self.rho
            q2 = grad_v2 + eta2 / self.rho
            
            # 方向1 (跨边缘): L0 保持锐利
            z1 = self._solve_prox_L0(q1, self.L1 / self.rho)
            # 方向2 (沿边缘): L2 强力平滑
            z2 = self._solve_prox_L2(q2, self.L2 / self.rho)
            
            # --- 2. U-update (FFT Linear System) ---
            z1_hat = z1 - eta1 / self.rho
            z2_hat = z2 - eta2 / self.rho
            
            # 将标量投影回 x, y 坐标系
            Vx = z1_hat * self.v1[...,0] + z2_hat * self.v2[...,0]
            Vy = z1_hat * self.v1[...,1] + z2_hat * self.v2[...,1]
            
            # RHS = f - rho * div(V)
            rhs_spatial = f_channel - self.rho * self._compute_divergence(Vx, Vy)
            rhs_fft = fft2(rhs_spatial)
            
            # FFT 求解
            u_fft = rhs_fft / self.otf_denominator
            u = np.real(ifft2(u_fft))
            
            # --- 3. Eta-update (Dual Ascent) ---
            gx_new, gy_new = self._compute_gradient(u)
            grad_v1_new = gx_new * self.v1[...,0] + gy_new * self.v1[...,1]
            grad_v2_new = gx_new * self.v2[...,0] + gy_new * self.v2[...,1]
            
            eta1 = eta1 + self.rho * (grad_v1_new - z1)
            eta2 = eta2 + self.rho * (grad_v2_new - z2)
            
            if verbose and i % 5 == 0:
                residue = np.mean(np.abs(grad_v1_new - z1))
                print(f"Iter {i}: Residue = {residue:.5f}")

        return np.clip(u * 255.0, 0, 255).astype(np.uint8)


def smooth(image, base_lambda=0.02, rho=0.5, sigma_tensor=2.0, 
           iterations=30, texture_suppression=True, verbose=True):
    """
    SGANCS 图像平滑的便捷接口
    
    Args:
        image: 输入图像 (BGR, uint8)
        base_lambda: 基础平滑强度，越大越平滑
        rho: ADMM 惩罚参数，通常为 lambda 的 10-50 倍
        sigma_tensor: 结构张量积分尺度，噪点多时调大
        iterations: ADMM 迭代次数
        texture_suppression: 是否在纹理区加强平滑
        verbose: 是否打印收敛信息
        
    Returns:
        平滑后的图像 (BGR, uint8)
    """
    # 分析阶段
    analyzer = StructureAnalyzer(image)
    analyzer.compute_structure_tensor(sigma_tensor=sigma_tensor)
    lambda1_map, lambda2_map = analyzer.generate_adaptive_weights(
        base_lambda=base_lambda, 
        texture_suppression=texture_suppression
    )
    
    # 求解阶段
    solver = SGANCS_Solver(
        image, analyzer.v1, analyzer.v2, 
        lambda1_map, lambda2_map, rho=rho
    )
    result = solver.solve(iterations=iterations, verbose=verbose)
    
    return result

