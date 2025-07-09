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
    parser.add_argument("--dataset", type=str, help="Path to dataset txt file")
    parser.add_argument(
        "--no_nms", action="store_true", help="Model exported without NMS"
    )
    args = parser.parse_args()

    model_path = args.model
    platform = args.platform
    dataset = args.dataset
    no_nms = args.no_nms
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
    if no_nms:
        print("Loading for without NMS")
        ret = rknn.load_onnx(
            model=model_path, inputs=["image"], input_size_list=[[1, 3, 640, 640]]
        )
    else:
        print("Loading for with NMS")
        ret = rknn.load_onnx(
            model=model_path,
            inputs=["image", "im_shape", "scale_factor"],
            input_size_list=[
                [1, 3, 640, 640],  # image
                [1, 2],  # im_shape
                [1, 2],  # scale_factor
            ],
        )
    if ret != 0:
        print("Load model failed!")
        exit(ret)
    print("Loaded model successfully")

    print("Building model")
    ret = rknn.build(do_quantization=True, dataset=dataset, rknn_batch_size=1)
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
