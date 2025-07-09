# ONNX to RKNN

This is script to convert PP-YOLOE model from ONNX format to RKNN format.
Must be run on hardware with Rockchip chips rk3588 rk3576.

Install dependencies:

```bash
uv sync
```

Run script:

```bash
uv run main.py --dataset {dataset.txt path} --model {onnx model path} --platform {rk3588/rk3576}
```
