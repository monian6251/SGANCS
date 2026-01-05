# SGANCS

Structure-Guided Anisotropic Non-Convex Smoothing

基于 Wei Liu 等人论文 *"A generalized framework for edge-preserving and structure-preserving image smoothing"* (IEEE TPAMI, 2021) 的改进算法。

## 安装

```bash
pip install -r requirements.txt
```

## 使用方法

### 命令行

```bash
# 使用默认图片
python demo.py

# 指定图片
python demo.py path/to/image.jpg

# 自定义参数
python demo.py image.jpg --lambda 0.05 --rho 0.3 --iter 50

# 指定输出目录
python demo.py image.jpg -o my_output

# 调试模式（显示权重图）
python demo.py image.jpg --debug

# 静默模式
python demo.py image.jpg --quiet

# 查看所有参数
python demo.py --help
```

### 参数说明

| 参数 | 缩写 | 默认值 | 说明 |
|------|------|--------|------|
| `--lambda` | `-l` | 0.02 | 基础平滑强度，越大越平滑 |
| `--rho` | `-r` | 0.5 | ADMM 惩罚参数，通常为 lambda 的 10-50 倍 |
| `--sigma` | `-s` | 2.0 | 结构张量积分尺度，噪点多时调大 |
| `--iter` | `-i` | 30 | ADMM 迭代次数 |
| `--output` | `-o` | output | 输出目录，自动保存结果和对比图 |
| `--no-texture` | - | - | 禁用纹理抑制 |
| `--debug` | - | - | 显示调试信息（权重图等） |
| `--quiet` | `-q` | - | 静默模式，不打印迭代信息 |

### Python API

```python
import cv2
from sgancs import smooth

# 简单调用
img = cv2.imread('image.jpg')
result = smooth(img, base_lambda=0.02, rho=0.5)
cv2.imwrite('result.jpg', result)

# 完整参数
result = smooth(
    img,
    base_lambda=0.02,      # 基础平滑强度
    rho=0.5,               # ADMM 惩罚参数
    sigma_tensor=2.0,      # 结构张量积分尺度
    iterations=30,         # 迭代次数
    texture_suppression=True,  # 纹理抑制
    verbose=True           # 打印收敛信息
)
```

### 高级用法（分步控制）

```python
import cv2
from sgancs import StructureAnalyzer, SGANCS_Solver

img = cv2.imread('image.jpg')

# 1. 分析图像结构
analyzer = StructureAnalyzer(img)
mu1, mu2 = analyzer.compute_structure_tensor(sigma_tensor=2.0)

# 2. 生成自适应权重
lambda1_map, lambda2_map = analyzer.generate_adaptive_weights(base_lambda=0.02)

# 3. ADMM 求解
solver = SGANCS_Solver(img, analyzer.v1, analyzer.v2, lambda1_map, lambda2_map, rho=0.5)
result = solver.solve(iterations=30)

cv2.imwrite('result.jpg', result)
```

## 算法设计

详见 [DESIGN.md](DESIGN.md)
