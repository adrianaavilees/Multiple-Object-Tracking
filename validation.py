import pandas as pd
import os
import matplotlib.pyplot as plt
from multipleObjectTracking import process_video
import numpy as np

# Dictionary with ground truth values for each video
GROUND_TRUTH = {
    "video1.mp4": {"up": 6, "down": 2},
    "video2.mp4": {"up": 5, "down": 7},
    "video3.mp4": {"up": 3, "down": 10},
    "video4.mp4": {"up": 8, "down": 24},
}

SHOW_VIDEO = False  # Set to True to visualize the video during the validation
CSV_FILE = "car_count_comparison.csv"

if __name__ == "__main__":
    # If the CSV file already exists, load it
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        print(f"Loaded existing results from CSV: {CSV_FILE}")

    else:
        results = []

        for video_name, truth in GROUND_TRUTH.items():
            print(f"Processing: {video_name}")

            detected_up, detected_down = process_video(video_name, show_video=SHOW_VIDEO)

            results.append({
                "video": video_name,
                "theoretical_up": truth["up"],
                "theoretical_down": truth["down"],
                "detected_up": detected_up,
                "detected_down": detected_down,
            })

            print(f"RESULTS → Detected UP: {detected_up}, DOWN: {detected_down}")


        df = pd.DataFrame(results)
        df["error_up"] = df["detected_up"] - df["theoretical_up"]
        df["error_down"] = df["detected_down"] - df["theoretical_down"]

        # Save results to CSV
        df.to_csv(CSV_FILE, index=False)
        print("\nResults saved!")

    # Calculate Mean Absolute Error (MAE)
    df["abs_error_up"] = df["error_up"].abs()
    df["abs_error_down"] = df["error_down"].abs()
    MAE_up = df["abs_error_up"].mean()
    MAE_down = df["abs_error_down"].mean()

    print("\n============= FINAL RESULTS ================")
    print(df)
    print(f"\nMAE UP: {MAE_up:.2f}")
    print(f"MAE DOWN: {MAE_down:.2f}")

    # Visual comparison
    colors = ["#64a964", "#ee6767"]
    x = np.arange(len(df["video"]))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Car Counting Comparison", fontsize=16, weight="bold")

    # UP Comparison
    axes[0].bar(x - width/2, df["theoretical_up"], width, label="Real UP", color=colors[0])
    axes[0].bar(x + width/2, df["detected_up"], width, label="Detected UP", color=colors[1])
    axes[0].set_title("Vehicles Moving UP", fontsize=14)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df["video"], rotation=30)
    axes[0].legend()

    # DOWN Comparison
    axes[1].bar(x - width/2, df["theoretical_down"], width, label="Real DOWN", color=colors[0])
    axes[1].bar(x + width/2, df["detected_down"], width, label="Detected DOWN", color=colors[1])
    axes[1].set_title("Vehicles Moving DOWN", fontsize=14)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df["video"], rotation=30)
    axes[1].legend()
    axes[1].set_ylabel("Vehicle Count")

    plt.tight_layout()
    plt.show()

    # Error Visualization
    fig, ax = plt.subplots(figsize=(10, 5))
    bars_up = ax.bar(x - width/2, df["error_up"], width, label="Error UP", color=colors[0])
    bars_down = ax.bar(x + width/2, df["error_down"], width, label="Error DOWN", color=colors[1])
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(df["video"], rotation=30)
    ax.set_title("Counting Error per Video (Detected - Ground Truth)")
    ax.set_ylabel("Error (number of vehicles)")
    ax.legend()
    fig.tight_layout()
    plt.show()
