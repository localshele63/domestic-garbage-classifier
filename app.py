import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
VENDOR_DIR = ROOT_DIR / "vendor_packages"
if VENDOR_DIR.exists():
    sys.path.insert(0, str(VENDOR_DIR))

import gradio as gr
from PIL import Image

from src.categories import CLASS_EXAMPLES, display_name
from src.config import DEFAULT_CHECKPOINT, DEFAULT_CLASS_NAMES, DEFAULT_ENSEMBLE_MODELS
from src.inference import load_checkpoint, load_ensemble, predict_image, predict_image_ensemble


UNRELIABLE_CONFIDENCE_THRESHOLD = 0.60
LOW_CONFIDENCE_THRESHOLD = 0.80


def build_missing_app():
    message = (
        "还没有找到四分类模型文件。请先整理数据集并训练模型：\n"
        "1. 运行 scripts/prepare_dataset.py 整理数据集\n"
        "2. 运行 train.py 训练模型\n"
        "3. 再重新启动 app.py"
    )

    def missing_model(_image):
        return message, {}

    return gr.Interface(
        fn=missing_model,
        inputs=gr.Image(type="pil", label="上传垃圾图片"),
        outputs=[gr.Textbox(label="识别结果"), gr.Label(label="四分类置信度")],
        title="生活垃圾四分类图像识别系统",
        api_name=False,
    )


def build_app(checkpoint, class_names_path, ensemble_models=None):
    valid_ensemble = ensemble_models and all(
        Path(checkpoint_path).exists() and Path(class_path).exists()
        for checkpoint_path, class_path in ensemble_models
    )
    if not valid_ensemble and (not Path(checkpoint).exists() or not Path(class_names_path).exists()):
        return build_missing_app()

    if valid_ensemble:
        models, class_names, device = load_ensemble(ensemble_models)
        predictor_name = "集成模型"
    else:
        model, class_names, device = load_checkpoint(checkpoint, class_names_path)
        models = [model]
        predictor_name = "单模型"

    def classify(image: Image.Image):
        if image is None:
            return "请先上传一张图片。", {}

        if len(models) > 1:
            results = predict_image_ensemble(models, class_names, image, device, topk=4)
        else:
            results = predict_image(models[0], class_names, image, device, topk=4)
        best = results[0]
        if best["confidence"] < UNRELIABLE_CONFIDENCE_THRESHOLD:
            lines = [
                "预测类别：无法可靠判断，建议人工确认",
                f"最高置信度：{best['confidence'] * 100:.2f}%",
                "",
                "提示：四类置信度差距较小，模型没有足够把握。",
                "说明：当图片背景复杂、主体不清晰或物品不属于常见训练样本时，系统会提示人工确认。",
            ]
        else:
            lines = [
                f"预测类别：{display_name(best['class'])}",
                f"最高置信度：{best['confidence'] * 100:.2f}%",
            ]
        if UNRELIABLE_CONFIDENCE_THRESHOLD <= best["confidence"] < LOW_CONFIDENCE_THRESHOLD:
            lines.extend(
                [
                    "",
                    "提示：该图片置信度低于80%，建议人工确认。",
                    "说明：低置信度结果仅作为辅助参考，需要结合实际物品材质进一步判断。",
                ]
            )
        lines.extend(["", "各类别置信度："])
        label_scores = {}
        for item in results:
            label = display_name(item["class"])
            confidence = item["confidence"]
            label_scores[label] = confidence
            lines.append(f"{label}: {confidence * 100:.2f}%")
        return "\n".join(lines), label_scores

    with gr.Blocks(title="生活垃圾四分类图像识别系统") as demo:
        gr.Markdown(
            """
# 生活垃圾四分类图像识别系统
上传一张图片，系统按训练数据的四类工程标签输出预测。衣物、鞋类按原工程映射归入干垃圾；实际投放请结合地区规则和物品状态。
"""
        )
        gr.Markdown(f"当前推理方式：{predictor_name}")
        with gr.Row():
            with gr.Column(scale=5):
                image_input = gr.Image(type="pil", label="上传垃圾图片")
                run_button = gr.Button("开始识别", variant="primary")
            with gr.Column(scale=4):
                result_text = gr.Textbox(label="识别结果", lines=8)
                result_label = gr.Label(label="四分类置信度", num_top_classes=4)

        gr.Markdown(
            f"""
## 分类说明

| 类别 | 示例 |
| --- | --- |
| 可回收物 | {CLASS_EXAMPLES["recyclable"]} |
| 有害垃圾 | {CLASS_EXAMPLES["hazardous"]} |
| 湿垃圾/厨余垃圾 | {CLASS_EXAMPLES["wet"]} |
| 干垃圾/其他垃圾 | {CLASS_EXAMPLES["dry"]} |
"""
        )

        outputs = [result_text, result_label]
        run_button.click(classify, inputs=image_input, outputs=outputs, api_name=False)
        image_input.change(classify, inputs=image_input, outputs=outputs, api_name=False)

    return demo


def main():
    parser = argparse.ArgumentParser(description="启动生活垃圾四分类识别界面")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--class-names", default=DEFAULT_CLASS_NAMES)
    parser.add_argument("--single-model", action="store_true", help="只使用单个模型，不使用默认集成模型")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    args = parser.parse_args()

    ensemble_models = None if args.single_model else DEFAULT_ENSEMBLE_MODELS
    demo = build_app(args.checkpoint, args.class_names, ensemble_models=ensemble_models)
    demo.launch(server_name=args.server_name, server_port=args.server_port, show_api=False)


if __name__ == "__main__":
    main()
