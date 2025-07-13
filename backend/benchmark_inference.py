import requests
import time
import argparse
import json

def run_benchmark(endpoint_url, filenames):
    payload = {"filenames": filenames}
    
    start = time.time()
    response = requests.post(endpoint_url, json=payload)
    duration = time.time() - start

    if response.status_code != 200:
        print(f"❌ {endpoint_url} failed: {response.status_code} - {response.text}")
        return None, duration

    try:
        predictions = response.json()
    except Exception as e:
        print(f"❌ Failed to decode JSON: {e}")
        return None, duration

    return predictions, duration

def main():
    parser = argparse.ArgumentParser(description="Benchmark /predict vs /predict_optimized")
    parser.add_argument("--host", type=str, default="http://localhost:8000", help="Backend API host")
    parser.add_argument("--count", type=int, default=10, help="Number of images to benchmark")
    parser.add_argument("--list", type=str, default="uploaded_images/image_list.txt", help="File with list of image filenames")
    args = parser.parse_args()

    with open(args.list, "r") as f:
        all_filenames = [line.strip() for line in f.readlines()]
    
    filenames = all_filenames[:args.count]

    print(f"▶️  Benchmarking {len(filenames)} images...")

    # PyTorch
    torch_url = f"{args.host}/predict"
    torch_preds, torch_time = run_benchmark(torch_url, filenames)
    print(f"⏱  PyTorch: {torch_time:.3f} seconds")

    # OpenVINO
    ov_url = f"{args.host}/predict_optimized"
    ov_preds, ov_time = run_benchmark(ov_url, filenames)
    print(f"⚡ OpenVINO: {ov_time:.3f} seconds")

    print("\n📊 Summary:")
    print(f"🧠 PyTorch (slow):    {torch_time:.3f}s")
    print(f"🚀 OpenVINO (fast):   {ov_time:.3f}s")
    print(f"💡 Speedup:           {torch_time / ov_time:.2f}x\n" if ov_time else "")

    # Optional: Save output
    with open("benchmark_results.json", "w") as out:
        json.dump({
            "torch_time": torch_time,
            "openvino_time": ov_time,
            "torch_predictions": torch_preds,
            "openvino_predictions": ov_preds
        }, out, indent=2)
        print("✅ Saved results to benchmark_results.json")

if __name__ == "__main__":
    main()
