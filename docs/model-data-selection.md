# 公开模型与数据选型（核验于 2026-10-16）

## 结论

本项目继续使用专用视觉模型，而不是 LLM。公开资源能提供很好的初始化权重、教师模型和
基准数据。在本次核验的候选中，尚无一个已经证明能同时满足本项目的输出契约、LiteRT
委托兼容性、复杂场景精度以及分发要求。本版已经内置 DocAligner 四角热图与 UVDoc
展平权重，采用 ONNX Runtime Android 1.23.2 CPU 执行，并通过桌面/Android 数值回归。
实时取景以 OpenCV 为快速路径；没有可用四边形时调用一次 DocAligner 彩色扩边模型。
去阴影仍为明确标识的传统处理。

拍照后的实际交互为：先在原始照片上检测并显示边框，用户可拖动四角和四个边缘控制点；
点击“应用并预览”或“校正并保存”后才处理原图。调整控制点或处理选项后，旧预览立即
失效；保存已经生成且未修改的预览时复用结果。

App 1.3 的 search-v3 在原四角权重上增加尺度自适应复检：小目标使用局部图，贴边或
弱预测使用背景扩边图，结果始终映射回原图，并检查不同尺度的一致性。低置信或贴边
候选需在裁剪页确认；扩边不会补全照片外内容。远近、轻微出画和边缘中断通过受控
样本回归；白纸叠白纸、低对比背景加入一张用户真实失败图回归。尚未做独立真实场景
盲测，权重本身未重新训练。

建议的生产组合是：

1. 实时边框：以 DocAligner 的 LCNet 热图模型作为教师模型，用 UVDoc、Doc3D 和自采真实
   照片重新训练 `384x384 mask + 4 corner heatmaps` 双头网络。
2. 拍照精修：使用相同数据训练 `1024 px` 精修分割/角点网络，不直接复用只有四角输出的
   DocAligner 权重。
3. 褶皱展平：本版保留 UVDoc 原生 `45x31` 网格；后续可蒸馏 `33x33` 学生模型。
4. 去阴影：使用 DocShadow-SD7K 配对数据，并让 FSENet 只充当离线教师；端侧学生模型只预测
   低频光照图，避免重绘文字。

精确下载地址、固定版本、哈希和机器可读决策在
`training/public_resources.json`。

## 权重核验

| 资源 | 已核验的仓库许可 | 现成权重 | 用途与限制 |
| --- | --- | --- | --- |
| [DocAligner](https://github.com/DocsaidLab/DocAligner) | Apache-2.0 | Release 1.1.1，LCNet/FastViT ONNX | 最适合作为四角检测教师和转换基线 |
| [UVDoc](https://github.com/tanguymagne/UVDoc) | MIT | 仓库内 `best_model.pkl`，约 32 MB | 最适合展平初始化/蒸馏 |
| [DocShadow-SD7K / FSENet](https://github.com/CXH-Research/DocShadow-SD7K) | MIT；外部数据/权重条款需另核对 | Release `Weights`，SD7K 权重约 118 MB | 去阴影教师候选；Android 性能未测 |
| [DewarpNet](https://github.com/cvlab-stonybrook/DewarpNet) | MIT；外部权重条款需另核对 | README 链接到 Google Drive，未完整下载 | 几何模型对照候选；Android 性能未测 |
| [DocTr](https://github.com/fh2019ustc/DocTr) | 禁止未授权商用 | 提供预训练权重 | 排除；取得书面商业授权前不得使用或分发 |

实际下载并检查了 DocAligner 1.1.1 的两个小模型：

- `lcnet100_h_e_bifpn_256_fp32.onnx`：4,767,987 字节，输入
  `[1,3,256,256]`，输出四通道热图；SHA-256 为
  `f4117b786e3a18470f3865c93f3c2bd69d9b998edd60f385574a5c665e79594e`。
- `lcnet050_p_multi_decoder_l3_d64_256_fp32.onnx`：4,911,217 字节，输出
  `[1,8]` 四角坐标和 `[1,1]` 目标置信度；SHA-256 为
  `32d186080ce16442674d4c0eaaaaac878eea289b56a8d1284f05fff1ff42e220`。

下载入口：[DocAligner 1.1.1 官方权重](https://github.com/DocsaidLab/DocAligner/releases/tag/1.1.1)。
这两个权重只有四角相关输出，不提供文档掩码。它们可以作为拍照后裁剪框的检测候选，
不能直接替代计划中的双头分割模型。README 将网络描述为 RGB，但官方示例先把 RGB 转为
BGR，推理源码只做缩放、NCHW 转置和 `/255`。本版已采用真实彩色 BGR 路径，
并以无损 PNG 输入在桌面与 Android 上对齐全部 65,536 个热图元素。

热图模型包含 20 个 `Einsum`，点回归模型包含大量动态 `Shape/Reshape/Gather/MatMul`
节点。它们是 NCHW ONNX，本版新增专用 ONNX 适配器，没有复用旧 NHWC LiteRT 接口。
热图模型随 APK 安装并实际执行；点回归模型仅作为外部比较候选，没有加入 App。
若后续迁移 LiteRT，仍需完成转换数值一致性、算子检查和 GPU/NPU 委托测试。

UVDoc 的固定版本权重也已下载：

- [best_model.pkl，固定 commit](https://raw.githubusercontent.com/tanguymagne/UVDoc/026782ddc8488679522c16cdb38c223b4fc3934e/model/best_model.pkl)：32,158,393 字节。
- SHA-256：`7e90861b8a516eb4bc51f84bd889cb77275743d2d1d3ca8091951ec9f2b7da23`。
- 输入 `[1,3,712,488]`，输出二维反向网格 `[1,2,45,31]` 和三维网格；
  官方用二维网格、`align_corners=True` 完成采样。

本版用 `torch.load(weights_only=True)` 安全加载 checkpoint，只导出二维网格 head 为
FP32 ONNX opset 17；与 PyTorch 的最大绝对误差为 `1.4305115e-6`。
Android 模拟器与桌面 ONNX Runtime 的网格最大差异为 `2.3841858e-7`。
导出文件 SHA-256：`e748175c085bab34c19ff5c6af47f074086ed4099f6f3ff17373aeb662718f16`。

展平默认关闭，用户可开启并调整强度。输入是按已确认边框生成的低分辨率裁剪图；输出
网格经过有限值、范围与折叠检查，再和原图几何合成，最终只对原始像素重采样一次。
手动边缘调整采用连续位移场，替代旧四分块近似。模型失败不会静默输出近似结果。
这验证了工程接入和数值一致性；裁剪后输入分布、展平强度与复杂场景效果仍需盲测。

[FSENet 的 SD7K.pth](https://github.com/CXH-Research/DocShadow-SD7K/releases/download/Weights/SD7K.pth)
已核验发布信息，尚未完整下载。它输出的是去阴影后的 RGB 图片，并非低频光照图；因此
更适合作为训练教师候选。当前无法据此保证旗舰手机一秒内完成。

## 数据集核验

| 数据集 | 场景与标注 | 许可结论 | 用法 |
| --- | --- | --- | --- |
| [UVDoc Dataset](https://github.com/tanguymagne/UVDoc-Dataset) | 真实纸张形变采集、合成纹理、精确 2D/3D 网格 | 仓库 MIT；文档纹理来源仍需逐项做商业来源复核 | 边界、角点和网格训练主数据 |
| [Doc3D](https://huggingface.co/datasets/StonyBrook-CVLab/doc3D-dataset) | 当前官方介绍为 100k 合成图，带反向映射、深度等；65k 是旧 v0.9 | 仓库与数据卡为 MIT；纹理来源需复核；下载为 gated auto | 合成形变训练，按 mesh ID 切分 |
| [DocShadow-SD7K](https://github.com/CXH-Research/DocShadow-SD7K) | 7k 高分辨率阴影/干净配对，30+ 遮挡物、350+ 文档 | 仓库 MIT；外部数据集条款尚未单独核验 | 配对监督去阴影/光照估计；未确认提供显式阴影掩码 |
| [MIDV-500](https://arxiv.org/abs/1807.05786) | 500 个移动拍摄视频、证件四边形真值 | 原始数据未找到明确商业许可 | 许可确认后再决定使用；场景以证件为主 |
| [DocUNet](https://www3.cs.stonybrook.edu/~cvl/docunet.html) | 小型文档展平基准 | 页面未清楚声明使用许可 | 许可确认后可考虑评估，不混入训练集 |

可直接使用的官方数据入口：

- UVDoc：[训练数据](https://igl.ethz.ch/projects/uvdoc/UVDoc_final.zip)、
  [独立基准](https://igl.ethz.ch/projects/uvdoc/UVDoc_benchmark.zip)。包含 `seg`、
  `grid2d`、`grid3d`、`uvmap`、纹理与划分信息。背景来自 DTD，文档纹理来自多种论文、
  书籍及生成数据；生产数据优先换为自有版权纹理重新合成。
- Doc3D：[Hugging Face 官方库](https://huggingface.co/datasets/StonyBrook-CVLab/doc3D-dataset)。
  需要账户并完成访问条件；旧 Google Form 下载脚本已被官方标为弃用。不下载所有模态，
  先准备 RGB、backward map 和所需几何真值，再按训练需要补齐。
- SD7K：[OpenXLab 官方链接](https://openxlab.org.cn/datasets/lkljty/ShadowDocument7K)。
  作者注明压缩版约 9 GB、原版约 100 GB，优先压缩版。本次没有批量下载全量数据集。

公共数据不能代替原计划中的自采数据。尤其是散页、白纸白桌、深色纸张、文字靠边、反光、
遮挡和手机 ISP 差异，必须由至少 20,000 张有来源记录的真实照片覆盖。训练/验证/测试按
文档内容、场景、拍摄批次和设备隔离；同一页面或背景不能跨集合。

## 落地顺序

1. 用 DocAligner 热图模型和 OpenCV 对自采图做预标注，再人工修正掩码、四角与可见边界。
2. 合并 UVDoc、Doc3D，训练 MobileNetV3-Large/DeepLabV3+ 与 BiSeNetV2；按盲测角点误差和
   IoU 选型，而不是按单一公开榜单选型。
3. 用 UVDoc 权重训练/蒸馏 `33x33` 网格学生模型；平面页通过褶皱分类直接跳过。
4. 用 SD7K 训练光照估计学生模型，并用梯度/字符边缘保持损失约束文字结构；首版 App
   不引入 OCR。
5. 本版用固定 FP32 ONNX 建立可运行基线。后续的 FP16/INT8、LiteRT 与硬件 delegate 版本，
   必须通过精度门槛和三类真机性能测试后才能替换基线。

仓库许可证、权重条款与数据中第三方素材的条款分别记录，不能由一个 MIT 标识推断全部
素材的权限。第三方权重最终随 APK 分发时，应保留其要求的版权声明和许可证文本。

本报告没有进行候选模型盲测或三平台真机计时，因此推荐表示工程适配优先级，不代表
已达到 `IoU ≥ 0.97`、自动可用率 `≥ 98%` 或 `P95 ≤ 1,000 ms`。
