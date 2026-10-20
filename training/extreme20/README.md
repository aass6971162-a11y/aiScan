# 固定的 20 张复杂文档边框回归

这是检测后处理回归，不是训练数据集或模型再训练脚本。使用 UVDoc benchmark
偶数编号 00000 至 00038；前 10 张开发，后 10 张保留验证，现均成为固定回归。

Python 依赖：`numpy opencv-python onnxruntime h5py`。

`fetch_extreme_samples.py` 从官方 ZIP 按条目请求，下载时校验 CRC。
`benchmark_extreme.py` 保留 search-v4 和 search-v3 后处理，供历史比较。
`search_v5.py` 是 App 1.5 当前自适应后处理的桌面复现；不使用样本编号或真值做选框。
模型路径应指向工程 `app/src/main/assets/models/lcnet100_h_e_bifpn_256_fp32.onnx`。
数据路径指向解压后的 `UVDoc_benchmark`，需要 img、grid2d、metadata_sample、metadata_geom。

```sh
SCAN_BENCH_DATA=/absolute/path/to/UVDoc_benchmark \
SCAN_BENCH_OUTPUT=/absolute/path/to/results \
SCAN_CORNER_MODEL=/absolute/path/to/lcnet100_h_e_bifpn_256_fp32.onnx \
python search_v5.py --public
```

模式：contact 生成来源清单与官方角点拼图；baseline 运行原实现；develop 只跑前10张；
final 跑20张。保留验证集已被使用，不能当作后续调参的独立盲测。

`--public` 只读取公开的 20 张 UVDoc 样本，输出 v4/v5 逐图对比。
真实叠纸照片留在本地测试资产；受控变体回归不能代替独立多层实拍盲测。

Android 测试输入仅位于 `app/src/androidTest/assets/extreme20`，不进入正式 APK。
执行 `ExtremeDocumentInstrumentedTest` 会实际运行 ONNX 并校验 Android/桌面坐标与官方角点。
报告、来源哈希、样本 ZIP 和前后对比图随交付输出保存。
