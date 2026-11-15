# Kii Stress

Soft UI Evolution 亮色风格的轻量化 **CPU / GPU 极限压力测试**静态网页，零构建依赖，可直接部署到 **GitHub Pages (`main` 分支根目录)**。

## 核心特性

- **克制的 Soft UI Evolution 亮色界面**：柔和冷瓷灰基底、细腻双层高光/内凹阴影、高对比度字体与等宽数字读数，不堆砌多余装饰。
- **CPU 极限压力测试（Web Workers）**：
  - 自动识别 `navigator.hardwareConcurrency`，并支持自定义并发线程数最高至 `128` 线程（应对浏览器核心数限制或超多线程处理器）。
  - 零 GC 预分配内存 + 多路展开流水线，提供 **极限混合（FPU + ALU + 缓存）**、**浮点矩阵（FP64 GEMM）**、**整数哈希（SHA-256）** 三种内核。
  - 使用 `MessageChannel` 零延迟调度避免 `setTimeout` 4ms 限频，实现接近 100% 核心饱和度，同时主线程保持流畅响应。
- **GPU 极限压力测试（WebGPU + WebGL2 双后端）**：
  - 优先使用 **WebGPU Compute Shader**（双缓冲显存读写 + 高密度 `vec4` FMA/超越函数计算 + 异步多批次队列流水线），自动回退至 **WebGL2 高分辨率离屏 FBO 多通道 3D Mandelbulb 光线步进 + ALU 满载着色器**（配合 `fenceSync` 非阻塞队列压满）。
  - 支持 **自适应满载 (Auto-Max)** 动态校准：自动压满从集成显卡到旗舰独立显卡的算力，同时内置驱动 TDR 安全保护，防止单批次超时。
  - 支持解除 VSync 帧率上限、负载倍率调节（1×–16×）及「纯算力模式」（跳过预览合成）。
- **灵活的时长控制**：支持 `1 / 5 / 15 / 30 / 60 分钟`、自定义分秒倒计时，以及 **∞ 持续运行**；内置屏幕防休眠（Wake Lock API）与 `ESC` 一键紧急停止。

## GitHub Pages 部署方式

1. 将代码合并到仓库的 `main` 分支。
2. 进入 GitHub 仓库的 **Settings → Pages**。
3. 在 **Build and deployment → Source** 中选择 **Deploy from a branch**。
4. Branch 选择 **`main`**，目录选择 **`/ (root)`**，点击 **Save**。
5. 等待约 1 分钟即可通过 GitHub Pages 链接直接访问。
