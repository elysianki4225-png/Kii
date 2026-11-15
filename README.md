# Kii Stress

Soft UI Evolution 亮色风格的轻量化 **CPU / GPU / 内存 (RAM) 极限压力测试**静态网页，零构建依赖，可直接部署到 **GitHub Pages (`main` 分支根目录)**。

## 核心特性

- **克制的 Soft UI Evolution 亮色界面**：柔和冷瓷灰基底、细腻双层高光/内凹阴影、高对比度字体与等宽数字读数，支持中/英双语切换，不堆砌多余装饰。
- **CPU 极限压力测试（WebAssembly SIMD128 + 多 Worker）**：
  - 内置纯代码动态编译的 **WebAssembly SIMD128 (`v128`) 矢量核爆内核**，每线程并行展开 8 路 `f32x4` + 4 路 `f64x2` + 2 路 `i32x4` + 4 MB 缓存行读写，直接触发 CPU 底层 SSE/AVX/NEON 矢量执行单元满载。
  - 同时提供 **极限混合（FPU + ALU + 缓存）**、**浮点矩阵（FP64 GEMM）**、**整数哈希（SHA-256）** 内核；支持动态增减并发线程数至 `256`，100% 满载采用 `MessageChannel` 零延迟调度，另支持 75% / 50% 占空比。
- **GPU 极限压力测试（WebGPU 双管线 + WebGL2 4-MRT）**：
  - **WebGPU 后端**：最高 128 MB 乒乓显存缓冲 + 8 路 `vec4<f32>` 双发射计算着色器 + 周期性跨步显存读写 + 同步并发的离屏 3D 光线步进光栅管线（Compute + Render 全管线同时压满），4–6 层深队列持续提交。
  - **WebGL2 后端**：**4-MRT 四重渲染目标离屏 FBO**（最高 `4096×4096 × 4` 目标）、256² 程序化噪声纹理 TMU 采样、最高 `1024` 轮 8 路 `vec4` ALU 满载着色器与 `fenceSync` 非阻塞深队列。
  - 支持 **自适应满载 (`Auto-Max`)** 与 **高压 / 极限 / 狂暴毁灭** 档位、`1×–32×` 负载倍率、解除 VSync 帧率上限、纯算力模式与 TDR 驱动超时保护。
- **内存 (RAM) 压力与稳定性测试（多 Worker 物理页提交）**：
  - 跨多 Worker 以 `32 MB` 粒度分块分配并逐页写入（4 KB 页提交），绕开单线程堆上限（支持 `256 MB` 至 `16 GB`），并内置 OOM 安全保护；停止时立即终止 Worker 瞬间归还全部物理内存。
  - 提供 **极限带宽**（原生 SIMD `memcpy` + 64B 缓存行读写，跑满多通道 DDR4/DDR5/LPDDR5X 带宽）、**稳定性校验**（MemTest86 风格交替位型与地址相关 PRNG 逐字回读纠错）、**随机延迟击穿**（打穿硬件预取器与 L3 缓存）三种模式。
- **灵活的时长与安全控制**：支持 `1 / 5 / 15 / 30 / 60 分钟`、自定义分秒倒计时，以及 **∞ 持续运行**；内置屏幕防休眠（Wake Lock API）、`ESC` 一键紧急停止与三烤摘要统计。

## GitHub Pages 部署方式

1. 将代码合并到仓库的 `main` 分支。
2. 进入 GitHub 仓库的 **Settings → Pages**。
3. 在 **Build and deployment → Source** 中选择 **Deploy from a branch**。
4. Branch 选择 **`main`**，目录选择 **`/ (root)`**，点击 **Save** 即可访问。
