"""
a small example of how to inspect your model
using pytorch's forward hook
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

from model import MyModel


def my_forward_hook(self, input, output):
    """
    a simple forward hook (must define args self, model, output)
    """
    ci = 5
    cj = 5

    fig, axs = plt.subplots(ci, cj)
    for i in range(ci):
        for j in range(cj):
            axs[i, j].imshow(output[0, i*j, :, :].detach().numpy())

    plt.show()


def apply_forward_hook_to_conv_layers(model, hook):
    for layer in model.modules():
        # this iterates through all modules in a model, therefore we check the type of module here
        if isinstance(layer, torch.nn.Conv2d):
            print("appended hook to", layer)
            # register the forward hook
            layer.register_forward_hook(hook)


def main():
    # get model
    model = MyModel(in_channels=3, num_classes=2, channels=32, num_blocks=4)
    # model.load_state_dict(torch.load("../saved_weights/test_001.pt"))
    model.load_state_dict(torch.load(r"C:\Users\tsx10\PythonProjectsJupyter\TUM\FP\Images\trainning\saved_weights\test_001.pt"))

    # appy hook to specific layers
    apply_forward_hook_to_conv_layers(model, my_forward_hook)

    print("-"*150)

    # get sample
    # img = np.load("../data/processed/mini MNIST npy/test/0/10.npy")
    img = np.load(r"C:\Users\tsx10\PythonProjectsJupyter\TUM\FP\Images\trainning\data\processed\mini MNIST npy\test\0\10.npy")
    plt.imshow(img)
    plt.title("Input Image")
    plt.show()

    # prepare sample
    img = torch.tensor(img, dtype=torch.float32)
    img = img.view(1, 1, img.shape[0], img.shape[1])
    img = (img - 0.1304) / 0.2904

    y_hat = model(img)

    print("model prediction:", y_hat)
    print("class probabilities:", torch.softmax(y_hat, dim=-1))


if __name__ == "__main__":
    main()

