# SGANCS: Structure-Guided Anisotropic Non-Convex Smoothing

基于 Wei Liu 等人论文 *"A generalized framework for edge-preserving and structure-preserving image smoothing"* (IEEE TPAMI, 2021) 的改进算法设计文档。

本算法将原来的"标量场非凸优化"提升为 **"张量场导向的非凸优化"**，是一个扎实的理论创新点。

---

## 1. 核心痛点与创新逻辑

### 1.1 原算法的局限性（Isotropic）

Wei Liu 的框架中，正则项是 $\phi(|\nabla u|)$。

**几何含义**：它在每个像素点画一个"圆"——不管梯度的方向如何，只要幅度大就惩罚。

**后果**：它会把边缘附近的噪声和边缘本身混在一起处理，容易导致：
- 边缘断裂
- 锯齿伪影

### 1.2 本算法的创新（Anisotropic）

我们希望正则项能"感知方向"。

**几何含义**：我们在每个像素点画一个"椭圆"，实现方向敏感的惩罚：

| 方向 | 操作 | 效果 |
|------|------|------|
| **沿着边缘**（切向） | 大力平滑 | 消除锯齿和断裂 |
| **跨越边缘**（法向） | 严格保护 | 保持锐度 |

### 1.3 创新的杀伤力

在一个统一的框架里，同时实现了：
- **边缘的锐化**（非凸惩罚）
- **边缘的连续性**（各向异性扩散）

---

## 2. 数学建模：从标量到张量

### 2.1 构建结构张量（Structure Tensor）

对于输入图像 $f$，计算结构张量：

$$S_\rho = G_\rho * (\nabla f \nabla f^T) = \begin{pmatrix} s_{11} & s_{12} \\ s_{12} & s_{22} \end{pmatrix}$$

对 $S_\rho$ 进行特征分解，得到特征向量 $v_1, v_2$（对应特征值 $\mu_1 \ge \mu_2$）：

- $v_1$：梯度方向（跨越边缘的方向，变化最剧烈）
- $v_2$：切线方向（沿着边缘的方向，变化最小）

### 2.2 定义扩散张量（Diffusion Tensor / Guidance Tensor）

构建张量 $D(x)$ 来扭曲梯度的度量方式：

$$D(x) = \frac{1}{\sqrt{\mu_1+\mu_2}} (c_1 v_1 v_1^T + c_2 v_2 v_2^T)$$

**关键设计**（自适应参数的用武之地）：

| 参数 | 控制方向 | 设计原则 | 对应效果 |
|------|----------|----------|----------|
| $c_1$ | 跨越边缘 | 应极小，保护边缘 | $L_0$ 效果 |
| $c_2$ | 沿着边缘 | 应较大，连接断裂 | $L_2$ 或弱 $L_1$ 效果 |

### 2.3 新的目标函数（核心创新公式）

**张量范数形式**：

$$\min_u \frac{1}{2} \|u - f\|^2 + \lambda \sum_{x} \phi \left( \sqrt{\nabla u(x)^T \mathbf{D}(x) \nabla u(x)} \right)$$

**方向导数解耦形式**（便于嵌入 ADMM 框架）：

$$\min_u \frac{1}{2} \|u - f\|^2 + \sum_{x} \left( \alpha(x) \phi(|\nabla u^T v_1|) + \beta(x) \psi(|\nabla u^T v_2|) \right)$$

两项的设计：

| 项 | 惩罚对象 | 惩罚函数 | 权重 |
|----|----------|----------|------|
| 第一项 | 梯度方向的变化 | $\phi$：非凸惩罚（如 $L_0$） | $\alpha(x)$ 自适应 |
| 第二项 | 切向方向的变化 | $\psi$：可为 $L_2$（凸函数） | $\beta(x)$ 较大，强制顺滑 |

---

## 3. 符号定义与模型构建

假设图像域为 $\Omega$，对于每个像素 $x \in \Omega$：

### 3.1 结构张量特征向量

- $v_1(x)$: 梯度方向（跨边缘）
- $v_2(x)$: 切线方向（沿边缘）

**性质**：$v_1, v_2$ 构成 $\mathbb{R}^2$ 空间的一组标准正交基：

$$v_1^T v_1 = 1, \quad v_1^T v_2 = 0, \quad v_1 v_1^T + v_2 v_2^T = \mathbf{I}$$

### 3.2 方向导数

$$\nabla_1 u = v_1(x)^T \nabla u(x), \quad \nabla_2 u = v_2(x)^T \nabla u(x)$$

### 3.3 自适应惩罚函数

- $\lambda_1(x)$: 控制跨边缘的平滑（应保护边缘）
- $\lambda_2(x)$: 控制沿边缘的平滑（应加强平滑）

### 3.4 目标函数

$$\min_u E(u) = \frac{1}{2} \|u - f\|_2^2 + \sum_{x \in \Omega} \left( \lambda_1(x) \phi(|\nabla_1 u(x)|) + \lambda_2(x) \psi(|\nabla_2 u(x)|) \right)$$

其中 $\phi$ 是非凸惩罚项（如 $L_0$, Welsch），$\psi$ 可以是凸项或非凸项。

---

## 4. ADMM 求解框架推导

引入辅助变量 $z_1, z_2$ 分别对应两个方向的导数：

**约束**：$z_1(x) = \nabla_1 u(x)$, $z_2(x) = \nabla_2 u(x)$

### 4.1 增广拉格朗日函数

$$\mathcal{L}(u, z_1, z_2, \eta_1, \eta_2) = \frac{1}{2}\|u-f\|^2 + \sum_{i=1,2} \left( \sum_x \lambda_i \phi_i(|z_i|) + \frac{\rho}{2} \|z_i - \nabla_i u + \frac{\eta_i}{\rho}\|^2 \right)$$

其中 $\eta_1, \eta_2$ 是对偶变量，$\rho$ 是惩罚参数（全局常数，保持算法快速的关键）。

### 4.2 子问题分解

#### (1) $z$-子问题：局部非凸投影

$$\min_{z_i} \sum_x \lambda_i(x) \phi_i(|z_i(x)|) + \frac{\rho}{2} (z_i(x) - \text{target}_i(x))^2$$

其中 $\text{target}_i(x) = \nabla_i u(x) - \eta_i(x)/\rho$。

**关键分析**：虽然 $\lambda_i(x)$ 是空间变化的，但该问题在像素之间是 **完全解耦的**（Pixel-wise decoupled）。对于任意像素 $x$，本质上是标量 Proximal Operator：

$$z_i^*(x) = \text{Prox}_{\lambda_i(x)/\rho, \phi_i} (\text{target}_i(x))$$

**结论**：可直接复用 Wei Liu 论文中的闭式解，引入各向异性和自适应 $\lambda$ 没有增加计算复杂度。✅

#### (2) $u$-子问题：线性方程组求解

$$\min_u \frac{1}{2}\|u-f\|^2 + \frac{\rho}{2} \sum_{i=1,2} \| z_i - \nabla_i u + \frac{\eta_i}{\rho} \|^2$$

对 $u$ 求导并令其为 0（Euler-Lagrange Equation）：

$$(u - f) + \rho \sum_{i=1,2} \nabla_i^T (\nabla_i u - z_i - \frac{\eta_i}{\rho}) = 0$$

整理得到线性方程组 $( \mathbf{I} + \rho \mathbf{L} ) u = \text{RHS}$，其中：

$$\mathbf{L} = \nabla_1^T \nabla_1 + \nabla_2^T \nabla_2$$

**正交性技巧（The Orthogonality Trick）**：

展开算子 $\mathbf{L}$：

$$\nabla_i = v_i^T \nabla \implies \nabla_i^T \nabla_i = \nabla^T (v_i v_i^T) \nabla$$

$$\mathbf{L} = \nabla^T (v_1 v_1^T + v_2 v_2^T) \nabla$$

由完整性关系（Completeness Relation）：

$$v_1(x) v_1(x)^T + v_2(x) v_2(x)^T = \mathbf{I}$$

代入得：

$$\mathbf{L} = \nabla^T \mathbf{I} \nabla = \nabla^T \nabla = \Delta \quad (\text{Laplacian Operator})$$

**惊人结论**：只要保持 $\rho$ 为全局常数，$u$-子问题的左端矩阵依然是标准拉普拉斯矩阵！

$$(\mathbf{I} - \rho \Delta) u = \text{RHS}$$

该方程可使用 **FFT 快速求解**。✅

**RHS 计算**：

$$\text{RHS} = f + \rho [ \nabla_1^T(z_1 + \eta_1/\rho) + \nabla_2^T(z_2 + \eta_2/\rho) ]$$

具体操作：
1. 计算 $A = z_1 + \eta_1/\rho$，$B = z_2 + \eta_2/\rho$
2. 构造向量场 $\vec{V} = A \cdot \vec{v}_1 + B \cdot \vec{v}_2$
3. 计算 $\text{div}(\vec{V})$

复杂度 $O(N)$。

#### (3) $\eta$-更新（Dual Update）

$$\eta_i^{k+1} = \eta_i^k + \rho (\nabla_i u^{k+1} - z_i^{k+1})$$

---

## 5. 算法复杂度分析

假设图像像素数为 $N$。

### 预处理（分析阶段）
- 计算结构张量、特征值分解：$O(N)$
- 计算自适应权重 $\lambda_1(x), \lambda_2(x)$：$O(N)$

### ADMM 迭代（优化阶段）
- $z$-update：Prox 映射闭式解，$O(N)$
- $u$-update：
  - RHS 组装（向量旋转和散度）：$O(N)$
  - FFT 求解：$O(N \log N)$
- $\eta$-update：$O(N)$

**总复杂度**：$O(K \cdot N \log N)$，其中 $K$ 是迭代次数。

与 Wei Liu 原论文算法复杂度一致，在不牺牲速度的前提下引入了各向异性和自适应性。

---

## 6. 实现细节与注意事项

### 6.1 离散梯度的旋转不变性

在离散域中，使用标准 Laplacian 卷积核（如 `[0 -1 0; -1 4 -1; 0 -1 0]`）对应的 FFT，而非显式构造 $\nabla_1^T \nabla_1$，以保证数值稳定性。

### 6.2 边界条件

FFT 求解隐含周期性边界条件（Periodic Boundary Condition）。可扩展图像或使用 DCT（Neumann Boundary）处理。

### 6.3 U-Update 的巧妙设计

通过 `V = z1_hat * v1 + z2_hat * v2` 把经过各向异性处理的梯度场重新合成为向量场，然后计算散度 `div(V)`，最后交给标准 FFT 求解。巧妙绕过了"矩阵求逆难"的问题。

---

## 7. 调试指南

### 7.1 关键参数

| 参数 | 位置 | 说明 |
|------|------|------|
| `base_lambda` | `run_demo()` | 全局平滑力度基准。无变化→调大(0.05~0.1)；全糊→调小 |
| `rho` | `SGANCS_Solver` | ADMM 收敛速度控制。通常设为 lambda 的 10 倍左右 |
| `sigma_tensor` | `compute_structure_tensor()` | 方向感知范围。噪点多→调大(3.0~4.0) |

### 7.2 rho 参数建议

对于非凸问题（$L_0$），`rho` 通常设得比 `lambda` 大一个数量级：
- 例如：`lambda=0.01`, `rho=0.1~0.5`
- `rho` 太小：结果充满噪点
- `rho` 太大：收敛极慢

### 7.3 各向异性策略实验

当前实现：
- 方向1（跨边缘）：`_solve_prox_L0` → 保持锐利边缘
- 方向2（沿边缘）：`_solve_prox_L2` → 强力平滑

可尝试的变体：
- 方向2 改用 L0：线条更生硬（矢量图风格）
- 保持 L2：边缘连贯性更好（油画风格）

---

## 8. 理论贡献总结

本算法的核心亮点（可写入论文）：

1. **Implicit Anisotropy**：通过投影后的非凸约束（$z$-step）引入各向异性，同时保持 $u$-step 的各向同性结构（Laplacian）

2. **Computational Efficiency**：证明了引入张量场后，算法依然可以通过 FFT 在 $O(N \log N)$ 内求解，打破了"引入各向异性必然导致计算变慢"的刻板印象

3. **Adaptive Regularization**：证明了空间变化的 $\lambda(x)$ 在 ADMM 框架下不会破坏子问题的闭式解

