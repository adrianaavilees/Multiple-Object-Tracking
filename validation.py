import pandas as pd
import matplotlib.pyplot as plt
from multipleObjectTracking import process_video


# Dictionary with ground truth values for each video
GROUND_TRUTH = {
    "video1.mp4": {"up": 6, "down": 2},
    "video2.mp4": {"up": 5, "down": 7},
    "video3.mp4": {"up": 3, "down": 10},
    "video4.mp4": {"up": 8, "down": 24},
}

SHOW_VIDEO = False  # Set to True to visualize the video during the validation



if __name__ == "__main__":
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

    print("\n============= FINAL RESULTS ================")
    print(df)

    # Save results to CSV
    df.to_csv("car_count_comparison.csv", index=False)
    print("\nResults saved!")

    
    df["abs_error_up"] = df["error_up"].abs()
    df["abs_error_down"] = df["error_down"].abs()
    MAE_up = df["abs_error_up"].mean()
    MAE_down = df["abs_error_down"].mean()

    print(f"\nMAE UP: {MAE_up:.2f}")
    print(f"MAE DOWN: {MAE_down:.2f}")

    # Visual comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Car Counting Comparison", fontsize=16)

    # UP Comparison
    axes[0].bar(df["video"], df["theoretical_up"], label="Theoretical UP", alpha=0.6)
    axes[0].bar(df["video"], df["detected_up"], label="Detected UP", alpha=0.6)
    axes[0].set_title("Comparison - Vehicles UP")
    axes[0].legend()

    # DOWN Comparison
    axes[1].bar(df["video"], df["theoretical_down"], label="Theoretical DOWN", alpha=0.6)
    axes[1].bar(df["video"], df["detected_down"], label="Detected DOWN", alpha=0.6)
    axes[1].set_title("Comparison - Vehicles DOWN")
    axes[1].legend()

    plt.show()

    # Error Visualization
    plt.figure(figsize=(8, 4))
    plt.bar(df["video"], df["error_up"], label="Error UP")
    plt.bar(df["video"], df["error_down"], label="Error DOWN", alpha=0.7)
    plt.axhline(0, color="black", linestyle="--")
    plt.title("Error de conteo (Detected - Theoretical)")
    plt.legend()
    plt.show()
