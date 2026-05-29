"""
run_pipeline.py - NDVI端到端一键流水线
依次执行: 重建(step1-4) → 评估 → 训练 → 最终对比
"""
import os, sys, time, json, subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RECON_DIR = os.path.join(BASE_DIR, "02_reconstruction")
TRAIN_DIR = os.path.join(BASE_DIR, "03_training")
RESULT_DIR = os.path.join(BASE_DIR, "04_results")

def run_step(name, script_path):
    print(f"\n{'#'*60}")
    print(f"# 执行: {name}")
    print(f"{'#'*60}")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=os.path.dirname(script_path),
        capture_output=False,
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"  [ERROR] {name} 失败! 退出码: {result.returncode}")
        return False
    print(f"  {name} 完成, 耗时: {elapsed:.0f}s")
    return True

def main():
    total_t0 = time.time()
    print("="*60)
    print("NDVI 端到端流水线")
    print("="*60)

    steps = [
        ("Step1: 异常值过滤", os.path.join(RECON_DIR, "step1_filter.py")),
        ("Step2: 半月重采样", os.path.join(RECON_DIR, "step2_resample.py")),
        ("Step3: 缺失值插值", os.path.join(RECON_DIR, "step3_interpolation.py")),
        ("Step4: 时间序列平滑", os.path.join(RECON_DIR, "step4_smoothing.py")),
        ("评估: 重建效果",   os.path.join(RECON_DIR, "evaluate_reconstruction.py")),
        ("训练: 预测模型",   os.path.join(TRAIN_DIR, "train_all.py")),
    ]

    for name, script in steps:
        ok = run_step(name, script)
        if not ok:
            print(f"\n流水线在 [{name}] 失败，终止执行")
            sys.exit(1)

    # 最终对比
    print(f"\n{'='*60}")
    print("流水线执行完成!")
    print(f"{'='*60}")

    # 输出结果汇总
    recon_json = os.path.join(RESULT_DIR, "reconstruction", "reconstruction_metrics.json")
    train_json = os.path.join(RESULT_DIR, "training", "all_models_summary.json")

    if os.path.exists(recon_json):
        with open(recon_json) as f:
            recon = json.load(f)
        print("\n--- 重建方法对比 ---")
        for method, metrics in recon.items():
            print(f"  {method}: RMSE={metrics.get('RMSE_mean','N/A'):.4f}, SMAPE={metrics.get('SMAPE_mean','N/A'):.2f}%")

    if os.path.exists(train_json):
        with open(train_json) as f:
            train = json.load(f)
        print("\n--- 预测模型对比 ---")
        print(f"  {'模型':<22} {'MSE':>10} {'MAE':>8} {'RMSE':>8} {'SMAPE':>8}")
        for name, r in train.items():
            print(f"  {name:<22} {r['MSE_mean']:>10.6f} {r['MAE_mean']:>8.4f} {r['RMSE_mean']:>8.4f} {r['SMAPE_mean']:>7.2f}%")

    print(f"\n总耗时: {time.time()-total_t0:.0f}s")

if __name__ == "__main__":
    main()
