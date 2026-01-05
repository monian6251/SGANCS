#!/usr/bin/env python3
"""
SGANCS Demo - 图像平滑测试脚本

用法:
    python demo.py                           # 使用默认图片
    python demo.py path/to/image.jpg         # 指定图片路径
    python demo.py image.jpg --lambda 0.05   # 自定义参数
"""

import argparse
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sgancs import StructureAnalyzer, SGANCS_Solver, smooth


def parse_args():
    parser = argparse.ArgumentParser(
        description='SGANCS: Structure-Guided Anisotropic Non-Convex Smoothing'
    )
    parser.add_argument(
        'image', 
        nargs='?',
        default='images/red-back-5-6.jpg',
        help='输入图片路径 (默认: images/red-back-5-6.jpg)'
    )
    parser.add_argument(
        '--lambda', '-l',
        dest='base_lambda',
        type=float, 
        default=0.02,
        help='基础平滑强度 (默认: 0.02)'
    )
    parser.add_argument(
        '--rho', '-r',
        type=float, 
        default=0.5,
        help='ADMM 惩罚参数 (默认: 0.5)'
    )
    parser.add_argument(
        '--sigma', '-s',
        dest='sigma_tensor',
        type=float, 
        default=2.0,
        help='结构张量积分尺度 (默认: 2.0)'
    )
    parser.add_argument(
        '--iter', '-i',
        dest='iterations',
        type=int, 
        default=30,
        help='ADMM 迭代次数 (默认: 30)'
    )
    parser.add_argument(
        '--no-texture',
        dest='texture_suppression',
        action='store_false',
        help='禁用纹理抑制'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='output',
        help='输出目录 (默认: output)'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='静默模式，不打印迭代信息'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='显示调试信息（权重图等）'
    )
    
    return parser.parse_args()


def create_comprehensive_visualization(original, result, analyzer, residue_history, args):
    """创建综合可视化图像"""
    # 设置matplotlib中文字体和样式
    plt.rcParams['font.size'] = 10
    plt.style.use('dark_background')
    
    # 创建子图
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f'SGANCS Analysis - {os.path.basename(args.image)}', fontsize=16, color='white')
    
    # 转换BGR到RGB用于matplotlib显示
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # 确保result是正确的uint8格式和数值范围
    if result.dtype != np.uint8:
        if result.max() <= 1.0:
            result = (result * 255).astype(np.uint8)
        else:
            result = np.clip(result, 0, 255).astype(np.uint8)
    
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    
    # 第一行
    # 原图
    axes[0,0].imshow(original_rgb)
    axes[0,0].set_title('Original Image', color='white')
    axes[0,0].axis('off')
    
    # 结果图
    axes[0,1].imshow(result_rgb)
    axes[0,1].set_title('SGANCS Result', color='white')
    axes[0,1].axis('off')
    
    # 特征值图
    mu1_norm = analyzer.mu1 / (analyzer.mu1.max() + 1e-6)
    mu2_norm = analyzer.mu2 / (analyzer.mu2.max() + 1e-6)
    coherence = mu1_norm - mu2_norm
    
    im1 = axes[0,2].imshow(coherence, cmap='hot')
    axes[0,2].set_title('Coherence (Edge Strength)', color='white')
    axes[0,2].axis('off')
    plt.colorbar(im1, ax=axes[0,2], fraction=0.046)
    
    # 收敛曲线
    if residue_history:
        axes[0,3].plot(residue_history, 'cyan', linewidth=2)
        axes[0,3].set_title('ADMM Convergence', color='white')
        axes[0,3].set_xlabel('Iteration', color='white')
        axes[0,3].set_ylabel('Residue', color='white')
        axes[0,3].grid(True, alpha=0.3)
        axes[0,3].tick_params(colors='white')
    else:
        axes[0,3].text(0.5, 0.5, 'No Convergence\nData Available', 
                      ha='center', va='center', color='white', fontsize=12)
        axes[0,3].axis('off')
    
    # 第二行
    # Lambda1 权重图 (跨边缘)
    lambda1_map, lambda2_map = analyzer.generate_adaptive_weights(base_lambda=args.base_lambda)
    im2 = axes[1,0].imshow(lambda1_map, cmap='viridis')
    axes[1,0].set_title('Lambda1 (Cross-Edge)\nDark=Preserve', color='white')
    axes[1,0].axis('off')
    plt.colorbar(im2, ax=axes[1,0], fraction=0.046)
    
    # Lambda2 权重图 (沿边缘)
    im3 = axes[1,1].imshow(lambda2_map, cmap='plasma')
    axes[1,1].set_title('Lambda2 (Along-Edge)\nBright=Smooth', color='white')
    axes[1,1].axis('off')
    plt.colorbar(im3, ax=axes[1,1], fraction=0.046)
    
    # 方向场可视化
    v1_vis = np.zeros_like(original)
    # 将方向场映射到颜色
    v1_angle = np.arctan2(analyzer.v1[...,1], analyzer.v1[...,0])
    v1_mag = np.sqrt(analyzer.v1[...,0]**2 + analyzer.v1[...,1]**2)
    
    # HSV颜色空间：角度->色相，幅度->饱和度
    hsv = np.zeros((original.shape[0], original.shape[1], 3), dtype=np.uint8)
    hsv[...,0] = ((v1_angle + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
    hsv[...,1] = (v1_mag * 255).astype(np.uint8)
    hsv[...,2] = 255
    v1_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    axes[1,2].imshow(v1_vis)
    axes[1,2].set_title('Direction Field v1\n(Gradient Direction)', color='white')
    axes[1,2].axis('off')
    
    # 参数信息
    param_text = f"""Parameters:
lambda: {args.base_lambda}
rho: {args.rho}
sigma: {args.sigma_tensor}
iterations: {args.iterations}

Image Size: {original.shape[1]}×{original.shape[0]}
Channels: {original.shape[2] if len(original.shape)==3 else 1}

Algorithm: SGANCS
Structure-Guided Anisotropic 
Non-Convex Smoothing"""
    
    axes[1,3].text(0.05, 0.95, param_text, transform=axes[1,3].transAxes,
                  verticalalignment='top', color='white', fontsize=9,
                  bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
    axes[1,3].axis('off')
    
    plt.tight_layout()
    return fig


def run_demo(args):
    """运行 SGANCS 演示"""
    
    # 1. 读取图像
    if not os.path.exists(args.image):
        print(f"错误: 图片不存在 - {args.image}")
        return
    
    img = cv2.imread(args.image)
    if img is None:
        print(f"错误: 无法读取图片 - {args.image}")
        return
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    print(f"输入图片: {args.image}")
    print(f"图片尺寸: {img.shape[1]}x{img.shape[0]}")
    print(f"输出目录: {args.output}")
    print(f"参数: lambda={args.base_lambda}, rho={args.rho}, "
          f"sigma={args.sigma_tensor}, iterations={args.iterations}")
    print("-" * 50)
    
    # 2. 分析阶段
    print("分析图像结构...")
    analyzer = StructureAnalyzer(img)
    mu1, mu2 = analyzer.compute_structure_tensor(sigma_tensor=args.sigma_tensor)
    
    # 3. 生成自适应权重
    lambda1_map, lambda2_map = analyzer.generate_adaptive_weights(
        base_lambda=args.base_lambda,
        texture_suppression=args.texture_suppression
    )
    
    # 获取权重图用于可视化
    lambda1_map, lambda2_map = analyzer.generate_adaptive_weights(base_lambda=args.base_lambda)
    
    # 4. 求解阶段
    print("ADMM 优化中...")
    solver = SGANCS_Solver(
        img, analyzer.v1, analyzer.v2, 
        lambda1_map, lambda2_map, 
        rho=args.rho
    )
    
    # 使用修改后的求解器收集收敛历史
    residue_history = []
    
    class SGANCSWithHistory(SGANCS_Solver):
        def _solve_single_channel(self, u_init, f_channel, iterations, verbose):
            u = u_init
            z1 = np.zeros((self.rows, self.cols), dtype=np.float32)
            z2 = np.zeros((self.rows, self.cols), dtype=np.float32)
            eta1 = np.zeros_like(z1)
            eta2 = np.zeros_like(z2)
            
            for i in range(iterations):
                # Z-update
                gx, gy = self._compute_gradient(u)
                grad_v1 = gx * self.v1[...,0] + gy * self.v1[...,1]
                grad_v2 = gx * self.v2[...,0] + gy * self.v2[...,1]
                
                q1 = grad_v1 + eta1 / self.rho
                q2 = grad_v2 + eta2 / self.rho
                
                z1 = self._solve_prox_L0(q1, self.L1 / self.rho)
                z2 = self._solve_prox_L2(q2, self.L2 / self.rho)
                
                # U-update
                z1_hat = z1 - eta1 / self.rho
                z2_hat = z2 - eta2 / self.rho
                
                Vx = z1_hat * self.v1[...,0] + z2_hat * self.v2[...,0]
                Vy = z1_hat * self.v1[...,1] + z2_hat * self.v2[...,1]
                
                rhs_spatial = f_channel - self.rho * self._compute_divergence(Vx, Vy)
                from scipy.fft import fft2, ifft2
                rhs_fft = fft2(rhs_spatial)
                u_fft = rhs_fft / self.otf_denominator
                u = np.real(ifft2(u_fft))
                
                # Eta-update
                gx_new, gy_new = self._compute_gradient(u)
                grad_v1_new = gx_new * self.v1[...,0] + gy_new * self.v1[...,1]
                grad_v2_new = gx_new * self.v2[...,0] + gy_new * self.v2[...,1]
                
                eta1 = eta1 + self.rho * (grad_v1_new - z1)
                eta2 = eta2 + self.rho * (grad_v2_new - z2)
                
                # 收集收敛信息（只在第一个通道收集）
                residue = np.mean(np.abs(grad_v1_new - z1))
                if len(residue_history) < iterations:  # 只在第一个通道时收集
                    residue_history.append(residue)
                
                if verbose and i % 5 == 0:
                    print(f"Iter {i}: Residue = {residue:.5f}")

            return np.clip(u * 255.0, 0, 255).astype(np.uint8)
    
    # 使用增强版求解器
    solver_with_history = SGANCSWithHistory(
        img, analyzer.v1, analyzer.v2, 
        lambda1_map, lambda2_map, 
        rho=args.rho
    )
    result = solver_with_history.solve(iterations=args.iterations, verbose=not args.quiet)
    
    print("-" * 50)
    print("完成!")
    
    # 5. 生成输出文件名
    base_name = os.path.splitext(os.path.basename(args.image))[0]
    result_path = os.path.join(args.output, f"{base_name}_sgancs.jpg")
    analysis_path = os.path.join(args.output, f"{base_name}_analysis.png")
    
    # 6. 保存结果
    cv2.imwrite(result_path, result)
    print(f"平滑结果已保存: {result_path}")
    
    # 7. 创建综合分析图并保存
    print(f"Result image stats: dtype={result.dtype}, shape={result.shape}, "
          f"min={result.min():.2f}, max={result.max():.2f}")
    fig = create_comprehensive_visualization(img, result, analyzer, residue_history, args)
    fig.savefig(analysis_path, dpi=150, bbox_inches='tight', 
                facecolor='black', edgecolor='none')
    plt.close(fig)
    print(f"算法分析图已保存: {analysis_path}")
    
    print(f"\n所有输出文件已保存到: {args.output}/")
    print("- 平滑结果:", os.path.basename(result_path))
    print("- 算法分析:", os.path.basename(analysis_path))


def main():
    args = parse_args()
    run_demo(args)


if __name__ == '__main__':
    main()

