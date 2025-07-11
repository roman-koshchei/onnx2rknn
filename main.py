import argparse
import os
from rknn.api import RKNN


def main():
    parser = argparse.ArgumentParser(description="Convert ONNX model to RKNN format")
    parser.add_argument("--model", type=str, help="Path to the ONNX model file")
    parser.add_argument(
        "--platform",
        type=str,
        choices=["rk3588", "rk3576"],
        default="rk3588",
        help="Target platform (default: rk3588)",
    )
    parser.add_argument(
        "--dataset", type=str, help="Path to dataset txt file (optional)"
    )
    args = parser.parse_args()

    model_path = args.model
    platform = args.platform
    dataset = args.dataset
    print(f"Convert ONNX model to RKNN format: {model_path}")

    rknn = RKNN(verbose=True)

    print("Configuring model")
    ret = rknn.config(
        mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform=platform
    )
    if ret != 0:
        print("Config failed!")
        exit(ret)
    print("Configured model successfully")

    print("Loading ONNX model")
    ret = rknn.load_onnx(
        model=model_path, inputs=["image"], input_size_list=[[1, 3, 640, 640]]
    )
    if ret != 0:
        print("Load model failed!")
        exit(ret)
    print("Loaded model successfully")

    print("Building model")
    do_quantization = dataset is not None
    if do_quantization:
        print(f"Using dataset for quantization: {dataset}")
    else:
        print("No dataset provided, disabling quantization")
    ret = rknn.build(
        do_quantization=do_quantization, dataset=dataset, rknn_batch_size=1
    )
    if ret != 0:
        print("Build model failed!")
        exit(ret)
    print("Model is built successfully")

    print("Export RKNN model")
    output_path = os.path.splitext(model_path)[0] + ".rknn"
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print("Export rknn model failed!")
        exit(ret)
    print("Done!")
    rknn.release()


if __name__ == "__main__":
    main()
