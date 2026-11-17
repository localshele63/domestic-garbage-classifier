import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["timestamp_utc", "elapsed_seconds", "phase", "epoch", "gpu_name",
          "utilization_percent", "memory_used_mib", "temperature_c", "power_w", "status"]


def current_phase(path):
    if path.exists():
        for line in reversed(path.read_text(encoding="utf-8").splitlines()):
            try:
                item = json.loads(line)
                return item["phase"], item["epoch"]
            except (ValueError, KeyError):
                continue
    return "startup", 0


def gpu_sample():
    result = subprocess.run(
        ["nvidia-smi", "--id=0", "--query-gpu=name,utilization.gpu,memory.used,temperature.gpu,power.draw",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True,
        timeout=10, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    return next(csv.reader(result.stdout.splitlines(), skipinitialspace=True))


def monitor(path, phase_path, start, interval, stop):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        sample_count = 0
        while True:
            phase, epoch = current_phase(phase_path)
            row = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                   "elapsed_seconds": round(time.monotonic() - start, 3),
                   "phase": phase, "epoch": epoch, "status": "ok"}
            try:
                values = gpu_sample()
                row.update(dict(zip(FIELDS[4:9], values)))
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                row["status"] = str(error)
            writer.writerow(row)
            handle.flush()
            if sample_count % 10 == 0:
                print(f"{row['elapsed_seconds']}s | {phase} epoch {epoch} | GPU {row.get('utilization_percent', 'N/A')}% | memory {row.get('memory_used_mib', 'N/A')} MiB", flush=True)
            sample_count += 1
            if stop.wait(interval):
                break


def numeric(rows, key):
    values = []
    for row in rows:
        try:
            values.append(float(row[key]))
        except (ValueError, KeyError):
            pass
    return values


def make_plots(out_dir):
    sys.path.insert(0, str(ROOT / "vendor_packages"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with (out_dir / "gpu_metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    panels = [("utilization_percent", "GPU utilization (%)", "#1769aa"),
              ("memory_used_mib", "GPU memory (MiB)", "#ba3535"),
              ("power_w", "Power (W)", "#2d8050"),
              ("temperature_c", "Temperature (C)", "#b57c12")]
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    for axis, (key, label, color) in zip(axes, panels):
        points = [(float(row["elapsed_seconds"]), float(row[key])) for row in rows
                  if row.get(key, "").replace(".", "", 1).isdigit()]
        if not points:
            raise ValueError("No real GPU samples for " + key)
        x, y = zip(*points)
        axis.plot(x, y, color=color, linewidth=1.2)
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
    axes[0].set_ylim(0, 105)
    phases = [json.loads(line) for line in (out_dir / "phases.jsonl").read_text(encoding="utf-8").splitlines()]
    origin = json.loads((out_dir / "run_metadata.json").read_text(encoding="utf-8"))["start_timestamp"]
    for phase, following in zip(phases, phases[1:]):
        if phase["phase"] == "validation":
            for axis in axes:
                axis.axvspan(phase["timestamp"] - origin, following["timestamp"] - origin,
                             color="#f5b041", alpha=0.2)
    for phase in phases:
        if phase["phase"] == "train":
            x = phase["timestamp"] - origin
            axes[0].axvline(x, color="gray", alpha=0.3, linestyle=":")
            axes[0].text(x, 101, str(phase["epoch"]), fontsize=8, ha="center")
    axes[0].set_title("GPU monitoring (numbers = epoch starts; shading = validation)")
    axes[-1].set_xlabel("Elapsed time (seconds)")
    fig.tight_layout()
    fig.savefig(out_dir / "gpu_monitor.png", dpi=160)
    fig.savefig(out_dir / "gpu_monitor.svg")
    plt.close(fig)

    train_rows = [row for row in rows if row["phase"] == "train"]
    utilization = numeric(train_rows, "utilization_percent")
    summary = {"samples": len(rows), "training_samples": len(train_rows),
               "failed_samples": sum(row["status"] != "ok" for row in rows),
               "training_utilization_mean": sum(utilization) / len(utilization) if utilization else None,
               "training_utilization_min": min(utilization) if utilization else None,
               "training_utilization_max": max(utilization) if utilization else None,
               "peak_device_memory_mib": max(numeric(rows, "memory_used_mib"))}
    (out_dir / "gpu_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def write_run_report(out_dir, summary):
    with (out_dir / "train_history.csv").open(newline="", encoding="utf-8") as handle:
        history = list(csv.DictReader(handle))
    metadata = json.loads((out_dir / "run_metadata.json").read_text(encoding="utf-8"))
    config = json.loads((out_dir / "training_config.json").read_text(encoding="utf-8"))
    selected = max(history, key=lambda row: float(row["val_balanced_acc"]))
    duration = metadata["end_timestamp"] - metadata["start_timestamp"]
    lines = ["# 本机 GPU 训练与监控记录", "",
             f"显卡：{metadata['gpu']}。本次使用 CUDA，在本机真实训练了 {len(history)} 轮。",
             f"训练与监控总时长：约 {duration / 60:.2f} 分钟（包含加载、训练、验证、保存与进程退出，不含绘图）。", "",
             f"训练图片 {sum(config['train_counts'].values())} 张，验证图片 {config['val_count']} 张。",
             "模型：EfficientNet-B0，ImageNet 预训练权重，全参数微调；batch size 64，AdamW，类别加权交叉熵，混合精度。", "",
             "## GPU 波动", "", "![真实 GPU 监控](gpu_monitor.png)", "",
             f"采集约每 {metadata['interval_seconds']} 秒一次，实际间隔还包括 nvidia-smi 查询耗时。共 {summary['samples']} 次采样，其中训练阶段 {summary['training_samples']} 次，查询失败 {summary['failed_samples']} 次。",
             f"训练阶段 GPU 利用率：最低 {summary['training_utilization_min']}%，最高 {summary['training_utilization_max']}%，采样平均 {summary['training_utilization_mean']:.2f}%。",
             f"整段记录的显存使用峰值：{summary['peak_device_memory_mib']:.0f} MiB。", "",
             "图上数字标记每轮训练开始，淡橙色区域标记验证阶段。显存、利用率、温度、功率均为真实采样；未对曲线进行人为平滑。",
             "这里记录的是整张 GPU 0 的使用情况，包含桌面、浏览器和其他程序的占用，不等同于训练进程独占资源。阶段标记与采样存在约一个采样间隔的时间偏差。利用率高低不能直接说明识别准确率。", "",
             "## Loss 与准确率", "", "![训练损失](loss_curve.png)", "", "![训练准确率](accuracy_curve.png)", "",
             f"训练 loss：{float(history[0]['train_loss']):.6f} → {float(history[-1]['train_loss']):.6f}。",
             f"验证 loss：{float(history[0]['val_loss']):.6f} → {float(history[-1]['val_loss']):.6f}。",
             f"按四类平均召回率选择第 {selected['epoch']} 轮检查点，该轮验证准确率 {float(selected['val_acc']) * 100:.2f}%，四类平均召回率 {float(selected['val_balanced_acc']) * 100:.2f}%。", "",
             "训练和验证曲线可以有波动，不要求每轮都下降。训练损失下降不等于实际场景全部识别正确。验证集用于选择模型，上述数字不是独立测试准确率。", "",
             "## 原始证据与使用", "",
             "- gpu_metrics.csv：带 UTC 时间戳、训练阶段和轮次的 GPU 原始采样。",
             "- phases.jsonl：训练、验证、检查点保存等阶段的时间戳。",
             "- training.log：本次训练的完整控制台记录。",
             "- train_history.csv：每轮 loss、准确率和四类平均召回率。",
             "- training_config.json、run_metadata.json：训练设置、启动命令与运行时间。",
             "- best_model.pth：此次实验选中的检查点。", "",
             "本次是单独的新训练实验，未修改网页默认模型路径，未替换之前的模型，也没有将外部测试图片加入训练。与之前 12 轮记录属于不同运行，应分别汇报。",
             "下次可双击项目根目录的 05_GPU监控训练.bat，从头训练并同步采集，每次写入新的时间戳目录。"]
    (out_dir / "训练与GPU监控说明.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Train a separate model while recording real NVIDIA GPU metrics")
    parser.add_argument("--out-dir", default="outputs/gpu_training_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--data-dir", default="data/domestic_garbage")
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()
    if args.epochs < 1 or args.interval < 0.5:
        parser.error("epochs must be positive; interval must be at least 0.5 seconds")
    gpu = gpu_sample()
    out_dir = (ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    phase_path = out_dir / "phases.jsonl"
    command = [sys.executable, "-u", "train.py", "--model", "efficientnet_b0", "--epochs", str(args.epochs),
               "--data-dir", args.data_dir,
               "--batch-size", "64", "--num-workers", "4", "--lr", "0.0001", "--class-weight",
               "--label-smoothing", "0", "--amp", "--deduplicate", "--selection-metric", "balanced_accuracy",
               "--patience", "0", "--output-dir", str(out_dir), "--phase-log", str(phase_path)]
    metadata = {"start_timestamp": time.time(), "gpu": gpu[0], "interval_seconds": args.interval,
                "command": command, "scope": "whole GPU 0, includes other applications; sampling time is approximate"}
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    stop = threading.Event()
    worker = threading.Thread(target=monitor, args=(out_dir / "gpu_metrics.csv", phase_path,
                              time.monotonic(), args.interval, stop), daemon=True)
    worker.start()
    print("GPU:", gpu[0], flush=True)
    print("Output:", out_dir, flush=True)
    process = None
    try:
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        with (out_dir / "training.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
            result = process.wait()
    except KeyboardInterrupt:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait()
        raise
    finally:
        stop.set()
        worker.join(timeout=15)
    metadata.update({"end_timestamp": time.time(), "exit_code": result})
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if result:
        raise RuntimeError("Training failed; see " + str(out_dir / "training.log"))
    subprocess.run([sys.executable, "scripts/plot_training_curves.py", "--history", str(out_dir / "train_history.csv"),
                    "--out-dir", str(out_dir)], cwd=ROOT, check=True)
    summary = make_plots(out_dir)
    write_run_report(out_dir, summary)
    print("GPU summary:", json.dumps(summary), flush=True)
    print("Training and GPU plots complete.", flush=True)


if __name__ == "__main__":
    main()
