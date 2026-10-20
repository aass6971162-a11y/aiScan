"""Export UVDoc's 2D head and generate desktop numerical fixtures for Android.

Requires numpy, onnx==1.17.0, onnxruntime==1.23.2, torch==2.6.0,
opencv-python-headless==4.12.0.88. The upstream module must be the reviewed MIT
model.py from UVDoc commit 026782ddc8488679522c16cdb38c223b4fc3934e.
No unsafe checkpoint unpickling is allowed.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    args = parser.parse_args()
    assert digest(args.weights / "uvdoc_best_model.pkl") == "7e90861b8a516eb4bc51f84bd889cb77275743d2d1d3ca8091951ec9f2b7da23"
    assert digest(args.upstream) == "320f460edc54cf830dfdd9788dfbc1b96a836429b64ed6078a30582be6f59e42"
    args.output.mkdir(parents=True, exist_ok=True)
    args.fixtures.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    spec = importlib.util.spec_from_file_location("uvdoc_upstream", args.upstream)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    net = module.UVDocnet(num_filter=32, kernel_size=5).eval()
    checkpoint = torch.load(args.weights / "uvdoc_best_model.pkl", weights_only=True, map_location="cpu")
    net.load_state_dict(checkpoint["model_state"], strict=True)

    # Register net as submodule so the exporter includes parameters as initializers.
    class ExportModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = net

        def forward(self, x):
            return self.net(x)[0]

    bgr = cv2.imread(str(args.image))
    assert bgr is not None
    resized = cv2.resize(bgr, (256, 256), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(str(args.fixtures / "corners_input.png"), resized)
    inp = np.ascontiguousarray(resized.transpose(2, 0, 1)[None], dtype=np.float32) / 255
    model = args.weights / "lcnet100_h_e_bifpn_256_fp32.onnx"
    onnx.checker.check_model(onnx.load(model))
    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    heatmap = session.run(None, {session.get_inputs()[0].name: inp})[0]
    heatmap.astype("<f4").tofile(args.fixtures / "corners_expected.f32")
    print("corners", heatmap.shape, "peaks", heatmap.reshape(4, -1).max(axis=1), flush=True)
    for color in (0, 127, 255):
        blank = np.full_like(inp, color / 255)
        h = session.run(None, {session.get_inputs()[0].name: blank})[0]
        print("blank", color, "peaks", h.reshape(4, -1).max(axis=1), flush=True)

    rgb = cv2.cvtColor(cv2.resize(bgr, (488, 712)), cv2.COLOR_BGR2RGB)
    cv2.imwrite(str(args.fixtures / "uvdoc_input.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    uv_input = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255
    tensor = torch.from_numpy(uv_input)
    model_path = args.output / "uvdoc_grid_fp32.onnx"
    with torch.inference_mode():
        expected = net(tensor)[0].numpy()
        torch.onnx.export(ExportModel().eval(), tensor, str(model_path),
                          input_names=["rgb"], output_names=["grid"],
                          opset_version=17, dynamo=False, do_constant_folding=True)
    onnx.checker.check_model(onnx.load(model_path))
    uvsession = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    actual = uvsession.run(None, {"rgb": uv_input})[0]
    error = float(np.max(np.abs(actual - expected)))
    assert error < 1e-4, error
    actual.astype("<f4").tofile(args.fixtures / "uvdoc_expected.f32")
    report = {
        "upstream_commit": "026782ddc8488679522c16cdb38c223b4fc3934e",
        "upstream_module_sha256": digest(args.upstream),
        "corner_shape": list(heatmap.shape), "grid_shape": list(actual.shape),
        "torch_ort_max_abs_error": error,
        "uvdoc_onnx_sha256": digest(model_path),
        "fixture_source": "DocAligner 1.1.1 docs/run_test_card.jpg (Apache-2.0)",
    }
    (args.output / "numerical_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
