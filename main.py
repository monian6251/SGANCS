import numpy as np
import cv2
from scipy.fft import fft2, ifft2

class StructureAnalyzer:
    """
    第一阶段：分析图像结构，生成自适应参数和方向场
    """
    def __init__(self, image):
        # 确保输入是 float32, 范围 [0, 1]
        self.img = image.astype(np.float32) / 255.0 if image.max() > 1.0 else image.astype(np.float32)
        if len(self.img.shape) == 3:
            # 简单起见，结构张量基于灰度图计算，但应用于所有通道
            self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        else:
            self.gray = self.img
        
        self.rows, self.cols = self.gray.shape
        self.v1 = None # 梯度方向 (跨边缘)
        self.v2 = None # 切线方向 (沿边缘)
        self.mu1 = None # 大特征值
        self.mu2 = None # 小特征值

    def compute_structure_tensor(self, sigma_grad=1.0, sigma_tensor=2.0):
        """
        计算结构张量及其特征分解
        S = Gaussian * (grad * grad.T)
        """
        # 1. 计算梯度
        dx = cv2.Sobel(self.gray, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(self.gray, cv2.CV_32F, 0, 1, ksize=3)

        # 2. 构建张量积
        Ixx = dx * dx
        Ixy = dx * dy
        Iyy = dy * dy

        # 3. 高斯平滑张量分量 (积分尺度)
        # 这一步决定了方向估计的鲁棒性
        ksize = int(2 * round(3 * sigma_tensor) + 1)
        Sxx = cv2.GaussianBlur(Ixx, (ksize, ksize), sigma_tensor)
        Sxy = cv2.GaussianBlur(Ixy, (ksize, ksize), sigma_tensor)
        Syy = cv2.GaussianBlur(Iyy, (ksize, ksize), sigma_tensor)

        # 4. 特征分解 (Analytical solution for 2x2 matrix)
        # Trace and Determinant
        trace = Sxx + Syy
        det = Sxx * Syy - Sxy**2
        
        # Eigenvalues: mu1 >= mu2
        delta = np.sqrt(np.maximum(0, (trace/2)**2 - det))
        self.mu1 = trace/2 + delta
        self.mu2 = trace/2 - delta

        # Eigenvectors
        # v1 对应 mu1 (主要梯度方向/跨边缘)
        # v2 对应 mu2 (切线方向/沿边缘)
        
        # 计算 v1 = [vx, vy]
        # 矩阵 M - mu2*I 的列向量即为 v1
        vx = Sxx - self.mu2
        vy = Sxy
        
        # 归一化，处理零向量情况
        mag = np.sqrt(vx**2 + vy**2) + 1e-6
        vx /= mag
        vy /= mag
        
        # 保存方向场 (H, W, 2)
        self.v1 = np.stack((vx, vy), axis=-1)
        self.v2 = np.stack((-vy, vx), axis=-1) # v2 与 v1 垂直

        return self.mu1, self.mu2

    def generate_adaptive_weights(self, base_lambda=0.01, texture_suppression=True):
        """
        基于特征值生成自适应权重图
        Mapping mu1, mu2 -> lambda1(x), lambda2(x)
        """
        # 归一化特征值以进行加权
        max_val = self.mu1.max() + 1e-6
        mu1_norm = self.mu1 / max_val
        mu2_norm = self.mu2 / max_val
        
        # Coherence (相干性): 描述是否是线状结构
        # C = (mu1 - mu2) / (mu1 + mu2)^2
        # 简化版：C = (mu1 - mu2)
        coherence = mu1_norm - mu2_norm
        
        # --- 设计权重策略 ---
        
        # 1. 跨边缘方向 (Direction 1): 
        # 如果是强边缘 (coherence 高)，我们希望保留 -> lambda 小 (或者让L0起作用)
        # 如果是平坦区，我们希望平滑 -> lambda 大
        weight_map_1 = 1.0 / (1.0 + 10.0 * coherence) 
        lambda_1 = base_lambda * weight_map_1
        
        # 2. 沿边缘方向 (Direction 2):
        # 无论是边缘还是平坦，我们通常都希望沿切向平滑以连接断裂
        # 但如果有纹理抑制需求，在复杂纹理区可能需要更大的力度
        weight_map_2 = np.ones_like(coherence)
        if texture_suppression:
             # 在纹理区加大平滑力度
             weight_map_2 = 1.0 + 5.0 * mu2_norm 
        
        lambda_2 = base_lambda * 2.0 * weight_map_2 # 沿边缘通常允许更强的平滑
        
        return lambda_1, lambda_2

class SGANCS_Solver:
    """
    第二阶段：ADMM 求解器
    """
    def __init__(self, image, v1, v2, lambda1_map, lambda2_map, rho=0.1):
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
        """
        Matlab psf2otf 的 Python 实现，用于 FFT 卷积
        """
        in_size = psf.shape
        out_size = shape
        pad_size = (out_size[0] - in_size[0], out_size[1] - in_size[1])
        psf_padded = np.pad(psf, ((0, pad_size[0]), (0, pad_size[1])), 'constant')
        
        # 循环移位，使中心在 (0,0)
        for i in range(2):
            psf_padded = np.roll(psf_padded, -int(in_size[i] / 2), axis=i)
            
        return fft2(psf_padded)

    def _precompute_fft_operator(self):
        # 离散拉普拉斯算子
        laplacian_kernel = np.array([[0, -1, 0], 
                                     [-1, 4, -1], 
                                     [0, -1, 0]], dtype=np.float32)
        
        otf_lap = self._psf2otf(laplacian_kernel, (self.rows, self.cols))
        
        # LHS: I + rho * Laplacian
        # 注意：这里的 Laplacian 对应 -div(grad)，所以是 + rho * otf
        # 严格推导：||z - Du||^2 -> (I + rho D^T D) -> D^T D 是卷积 -Laplacian
        # 也就是 I - rho * Delta (如果是连续形式)
        # 在离散FFT中，psf2otf([0 -1 0; -1 4 -1]) 已经是特征值了
        return 1.0 + self.rho * otf_lap

    def _compute_gradient(self, img):
        # 使用周期性边界条件的差分
        grad_x = np.roll(img, -1, axis=1) - img
        grad_y = np.roll(img, -1, axis=0) - img
        return grad_x, grad_y

    def _compute_divergence(self, vx, vy):
        # div = dx_bwd + dy_bwd
        # Backward difference using roll
        div_x = vx - np.roll(vx, 1, axis=1)
        div_y = vy - np.roll(vy, 1, axis=0)
        return div_x + div_y

    def _solve_prox_L0(self, q, threshold):
        """
        Hard Thresholding for L0 norm
        z = q if |q|^2 > 2*lambda/rho else 0
        """
        # threshold here passed is lambda/rho
        # L0 prox condition: |x| > sqrt(2*lambda/rho)
        mask = (q**2) > (2 * threshold)
        return q * mask

    def _solve_prox_L2(self, q, threshold):
        """
        Scaling for L2 norm (Tikhonov)
        min lambda*z^2 + rho/2(z-q)^2 -> z = (rho * q) / (2*lambda + rho)
        """
        # threshold passed is lambda/rho
        return q / (1.0 + 2.0 * threshold)

    def solve(self, iterations=20):
        # 初始化
        u = self.f.copy()
        
        # 处理多通道
        if self.channels > 1:
            u_out = np.zeros_like(u)
            for c in range(self.channels):
                u_out[:,:,c] = self._solve_single_channel(u[:,:,c], iterations)
            return u_out
        else:
            return self._solve_single_channel(u, iterations)

    def _solve_single_channel(self, u_init, iterations):
        u = u_init
        
        # 辅助变量 z1, z2 (标量场)
        z1 = np.zeros((self.rows, self.cols), dtype=np.float32)
        z2 = np.zeros((self.rows, self.cols), dtype=np.float32)
        
        # 对偶变量 eta1, eta2
        eta1 = np.zeros_like(z1)
        eta2 = np.zeros_like(z2)
        
        for i in range(iterations):
            # --- 1. Z-update (Non-convex Projection) ---
            
            # 计算当前梯度
            gx, gy = self._compute_gradient(u)
            
            # 投影到结构方向
            # v1: (rows, cols, 2) -> v1[...,0] is x-comp, v1[...,1] is y-comp
            grad_v1 = gx * self.v1[...,0] + gy * self.v1[...,1]
            grad_v2 = gx * self.v2[...,0] + gy * self.v2[...,1]
            
            # 构造 Prox 输入
            q1 = grad_v1 + eta1 / self.rho
            q2 = grad_v2 + eta2 / self.rho
            
            # 求解 Prox
            # 方向1 (跨边缘): 使用 L0 保持锐利
            z1 = self._solve_prox_L0(q1, self.L1 / self.rho)
            
            # 方向2 (沿边缘): 使用 L2 强力平滑 (或用 L1)
            # 这里演示各向异性：不同方向用不同范数
            z2 = self._solve_prox_L2(q2, self.L2 / self.rho)
            
            # --- 2. U-update (FFT Linear System) ---
            
            # 我们需要重构向量场 V = (z1_hat * v1 + z2_hat * v2)
            # 其中 z_hat = z - eta/rho
            z1_hat = z1 - eta1 / self.rho
            z2_hat = z2 - eta2 / self.rho
            
            # 将标量投影回 x, y 坐标系
            # V_x = z1_hat * v1_x + z2_hat * v2_x
            Vx = z1_hat * self.v1[...,0] + z2_hat * self.v2[...,0]
            Vy = z1_hat * self.v1[...,1] + z2_hat * self.v2[...,1]
            
            # 计算 RHS = f + rho * div(V)
            # 注意: 之前的推导是 (I - rho*Delta)u = f - rho*div(Z - eta/rho)
            # 符号取决于定义的拉普拉斯算子方向，标准形式：
            # (u-f) + rho * div_adj(grad u - z_hat) = 0
            # u + rho * div_adj(grad u) = f + rho * div_adj(z_hat)
            # div_adj = -div.
            # u - rho * div(grad u) = f - rho * div(V)
            
            rhs_spatial = self.f - self.rho * self._compute_divergence(Vx, Vy)
            rhs_fft = fft2(rhs_spatial)
            
            # 求解
            u_fft = rhs_fft / self.otf_denominator
            u = np.real(ifft2(u_fft))
            
            # --- 3. Eta-update (Dual Ascent) ---
            
            # 重新计算新 u 的投影梯度
            gx_new, gy_new = self._compute_gradient(u)
            grad_v1_new = gx_new * self.v1[...,0] + gy_new * self.v1[...,1]
            grad_v2_new = gx_new * self.v2[...,0] + gy_new * self.v2[...,1]
            
            eta1 = eta1 + self.rho * (grad_v1_new - z1)
            eta2 = eta2 + self.rho * (grad_v2_new - z2)
            
            if i % 5 == 0:
                # 简单监控收敛性
                residue = np.mean(np.abs(grad_v1_new - z1))
                print(f"Iter {i}: Residue = {residue:.5f}")

        return np.clip(u * 255.0, 0, 255).astype(np.uint8)

# --- 使用示例 ---

def run_demo(image_path):
    # 1. 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found.")
        return

    # 2. 分析阶段
    print("Analyzing Structure Tensor...")
    analyzer = StructureAnalyzer(img)
    mu1, mu2 = analyzer.compute_structure_tensor(sigma_tensor=2.0)
    
    # 3. 生成自适应权重
    # lambda_base 越大，整体越平滑
    lambda1_map, lambda2_map = analyzer.generate_adaptive_weights(base_lambda=0.02)
    
    # 可视化权重图 (调试用)
    cv2.imshow("Lambda1 (Cross-Edge) - Dark means Keep Edge", (lambda1_map / lambda1_map.max()))
    
    # 4. 求解阶段
    print("Optimizing...")
    solver = SGANCS_Solver(img, analyzer.v1, analyzer.v2, lambda1_map, lambda2_map, rho=0.5)
    result = solver.solve(iterations=30)
    
    # 5. 显示结果
    cv2.imshow("Original", img)
    cv2.imshow("SGANCS Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 取消注释以运行 (需要本地图片)
run_demo('./ref/generalized_smoothing_framework/imgs/clip_art_compression_removal/01.jpg')