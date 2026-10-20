# 公开权重的可复现接入

本目录不是已完成的自有模型训练流水线。它提供公开权重来源、UVDoc 导出和黄金图片
生成工具，用于离线 Android 基线。未采集/训练 20,000 张真实图，也未下载全量公开数据。

1. 在独立 Python 3.12 环境中安装 `requirements-public-models.txt`。
2. 按 `public_resources.json` 下载官方 DocAligner LCNet100 ONNX 与 UVDoc checkpoint。
3. 下载 DocAligner 1.1.1 的 `docs/run_test_card.jpg` 作为参考图。
4. 运行：

```sh
python verify_public_models.py \
  --weights /absolute/path/to/public-models \
  --upstream vendor/UVDoc/model.py \
  --image /absolute/path/to/run_test_card.jpg \
  --output /absolute/path/to/exported \
  --fixtures ../app/src/androidTest/assets/model-golden
```

脚本校验 checkpoint 和上游源码哈希，只允许 `weights_only=True` 读取，保留原生网格。
导出后执行 ONNX checker 和 PyTorch/ONNX Runtime 全网格对齐。脚本生成的 `.f32` 是
小端 float32，Android 仪器测试用同一无损图像输入逐元素对比。

`vendor/UVDoc/model.py` 来源：
https://github.com/tanguymagne/UVDoc/blob/026782ddc8488679522c16cdb38c223b4fc3934e/model.py
原文件未修改，许可证见同目录 LICENSE。只把导出二维 head 的 ONNX 打包进 App。

APK 的模型描述见 `app/src/main/assets/models/manifest.json`。修改权重时必须重跑数值
回归并更新哈希；GPU/NPU、量化、复杂场景精度与时延均需独立验收。
