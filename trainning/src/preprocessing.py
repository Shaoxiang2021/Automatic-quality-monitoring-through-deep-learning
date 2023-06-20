"""
preprocessing scripts to move the data from
data/raw to data/processed.
"""

import numpy as np
import os
import json
from PIL import Image


def preprocess_mini_mnist(src_dir, out_dir):

    out_json = {
        "info": {
            "mean": np.zeros(3),
            "std": np.zeros(3),
        }
    }

    for split in ["train", "test"]:
        out_json[split] = []

        for label in os.listdir(f"{src_dir}/{split}"):
            print("processing", label)
            curr_out_dir = f"{out_dir}/{split}/{label}"
            os.makedirs(curr_out_dir, exist_ok=True)

            for file_name in os.listdir(f"{src_dir}/{split}/{label}"):
                # open image in greyscale mode
                img_frame = Image.open(f"{src_dir}/{split}/{label}/{file_name}")
                # convert to numpy and scale
                img_np = np.array(img_frame)/255.
                np.save(f"{curr_out_dir}/{file_name.replace('.jpg', '.npy')}", img_np)

                # append path and label
                out_json[split].append({
                    "path": f"{curr_out_dir}/{file_name.replace('.jpg', '.npy')}",
                    "label": int(label)
                })

                if split == "train":
                    # compute stats standardization and save them to json
                    mean = list()
                    std = list()
                    for i in range(img_np.shape[2]):
                        pixels = img_np[:, :, i].ravel()
                        mean.append(np.mean(pixels))
                        std.append(np.std(pixels))

                    out_json["info"]["mean"] += np.array(mean)
                    out_json["info"]["std"] += np.array(std)

    out_json["info"]["mean"] = list(out_json["info"]["mean"]/len(out_json["train"]))
    out_json["info"]["std"] = list(out_json["info"]["std"]/len(out_json["train"]))
    with open(out_dir + "/data.json", "w") as file:
        file.write(json.dumps(out_json, indent=4))

if __name__ == "__main__":
    preprocess_mini_mnist("../data/raw/mini MNIST JPG", out_dir="../data/processed/mini MNIST npy")
