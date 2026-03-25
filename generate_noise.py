import os
import json

def create_noise_config(image_dir, output_file):
    """
    Generate a noise configuration JSON file containing image paths and noise strengths.

    :param image_dir: Path to the dataset directory (should contain category subfolders).
    :param output_file: Path to save the generated JSON file.
    """
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)  # Creates directories if they don't exist

    # Prevent overwriting existing files
    if os.path.exists(output_file):
        print(f"Error: {output_file} already exists. Aborting to prevent modification.")
        return

    noise_strengths = [0.5]
    categories = ["0_AD_AR", "1_XL_XLC"]

    noise_config = {}

    for category in categories:
        category_path = os.path.join(image_dir, category)
        if not os.path.exists(category_path):
            print(f"Warning: {category_path} does not exist.")
            continue

        image_paths = sorted(
            [os.path.join(category_path, file) for file in os.listdir(category_path)
             if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
        )

        noise_config[category] = {
            "strength": noise_strengths,
            "image_path": image_paths
        }

    with open(output_file, "w") as json_file:
        json.dump(noise_config, json_file, indent=4)

    print(f"Noise configuration saved to {output_file}")


if __name__ == "__main__":
    base_dir = ""

    for fold in range(5):
        image_dir = f"{base_dir}/{fold}/train"
        output_file = f"{base_dir}/{fold}/self_full.json"
        print(f"\n=== Fold {fold} ===")
        create_noise_config(image_dir, output_file)
 