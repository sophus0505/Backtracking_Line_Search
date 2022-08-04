import numpy as np

import matplotlib

matplotlib.use("pgf")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json

from tabulate import tabulate

plt.rcParams.update(
    {
        "font.family": "serif",  # use serif/main font for text elements
        "text.usetex": True,  # use inline math for ticks
        "pgf.rcfonts": False,  # don't setup fonts from rc parameters
    }
)


def set_size(width_pt, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to sit nicely in our document.
    This function is taken from https://jwalton.info/Matplotlib-latex-PGF/.

    Parameters
    ----------
    width_pt: float
            Document width in points
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5 ** 0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


def plot_acc():

    plt.style.use("seaborn")

    with open("history/optimizers/optimizers_cifar10.json") as json_file:
        data = json.load(json_file)
    with open("history/optimizers/models_applied_cifar10.json") as json_file:
        data_mbt = json.load(json_file)

    data = data["0.1"]

    bgd_methods = ["MBT-GD", "MBT-MMT", "MBT-NAG"]
    gd_methods = ["SGD", "MMT", "NAG"]
    size = (set_size(426)[0] * 2, set_size(426)[1])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=size)
    for key, val in data_mbt.items():

        value = val["100"]
        acc_val = value["acc_valid"]
        loss_val = value["loss_valid"]
        x = np.arange(0, len(acc_val) + 1, 5)

        acc_val = acc_val[::5] + [acc_val[-1]]
        loss_val = loss_val[::5] + [loss_val[-1]]
        if key == "SGD":
            ax1.plot(x, acc_val, marker="d", linewidth=2, label="MBT-" + "GD")
            ax2.plot(x, loss_val, marker="d", linewidth=2, label="MBT-" + "GD")
        else:
            ax1.plot(x, acc_val, marker="d", linewidth=2, label="MBT-" + key)
            ax2.plot(x, loss_val, marker="d", linewidth=2, label="MBT-" + key)

    for key, value in data.items():
        label = key
        acc_val = value["acc_valid"]
        loss_val = value["loss_valid"]
        x = np.arange(0, len(acc_val) + 1, 5)

        acc_val = acc_val[::5] + [acc_val[-1]]
        loss_val = loss_val[::5] + [loss_val[-1]]

        if label in bgd_methods or label in ["RMSprop", "Adamax", "2W-BGD"]:
            continue
        else:
            ax1.plot(x, acc_val, marker=".", label=key, alpha=0.5)
            ax2.plot(x, loss_val, marker=".", label=key, alpha=0.5)

    ax1.set_title("Validation accuracy")
    ax1.set_xlabel("epochs")
    ax1.set_ylabel("accuracy")

    ax2.set_title("Validation loss")
    ax2.set_xlabel("epochs")
    ax2.set_ylabel("loss")
    ax2.legend(loc="lower left", bbox_to_anchor=(1, 0.25), ncol=1)

    plt.tight_layout()
    plt.savefig("figures/accuracies.pgf", bbox_inches="tight", pad_inches=0)
    plt.show()


def print_lr_acc():
    with open("history/optimizers/optimizers_cifar10.json") as json_file:
        data = json.load(json_file)

    with open("history/optimizers/models_applied_cifar10.json") as json_file:
        data_mbt = json.load(json_file)

    bgd_methods = ["2W-BGD", "MBT-MMT", "MBT-NAG"]
    for method in bgd_methods:
        for key, val in data.items():
            data[key].pop(method)

    table = {}
    table["learning rates"] = list(data["0.1"].keys()) + list(data_mbt.keys())
    for key, value in data.items():
        col = []
        for model, val in data[key].items():
            acc = val["acc_valid"]
            acc_final = acc[-1]
            col.append(acc_final)
        table[key] = col + [" - "] * 3

    final_col = [" - "] * 8
    for key, value in data_mbt.items():
        val = value["100"]
        acc = val["acc_valid"][-1]
        final_col.append(acc)

    table["backtrack"] = final_col

    print(tabulate(table, headers="keys", tablefmt="fancy_grid"))
    print(tabulate(table, headers="keys", tablefmt="latex"))


def print_lr_loss():
    with open("history/optimizers/optimizers_cifar10.json") as json_file:
        data = json.load(json_file)

    with open("history/optimizers/models_applied_cifar10.json") as json_file:
        data_mbt = json.load(json_file)

    bgd_methods = ["2W-BGD", "MBT-MMT", "MBT-NAG"]
    for method in bgd_methods:
        for key, val in data.items():
            data[key].pop(method)

    table = {}
    table["learning rates"] = list(data["0.1"].keys()) + list(data_mbt.keys())
    for key, value in data.items():
        col = []
        for model, val in data[key].items():
            loss = val["loss_valid"]
            loss_final = float(loss[-1])
            col.append(f"{loss_final:.2f}")
        table[key] = col + [" - "] * 3

    final_col = [" - "] * 8
    for key, value in data_mbt.items():
        val = value["100"]
        loss = float(val["loss_valid"][-1])
        final_col.append(f"{loss:.2f}")

    table["backtrack"] = final_col

    print(tabulate(table, headers="keys", tablefmt="fancy_grid"))
    print(tabulate(table, headers="keys", tablefmt="latex"))


def plot_time():
    plt.style.use("seaborn")

    with open("history/optimizers/optimizers_time2_cifar10.json") as json_file:
        data = json.load(json_file)
    data.pop("0.1")

    with open("history/optimizers/models_applied_cifar10.json") as json_file:
        data_mbt = json.load(json_file)

    size = (set_size(426)[0], set_size(426)[1])
    plt.figure(figsize=size)

    for key, val in data_mbt.items():
        value = val["100"]
        times = value["time"]
        if key == "SGD":
            times[0] = times[1]
        x = np.arange(25)

        if key == "SGD":
            plt.plot(x, times[:25], marker="d", linewidth=2, label="MBT-" + "GD")
        else:
            plt.plot(x, times[:25], marker="d", linewidth=2, label="MBT-" + key)

    bgd_methods = ["MBT-GD", "MBT-MMT", "MBT-NAG"]
    for key, val in data.items():
        if key in ["RMSprop", "Adamax", "2W-BGD"]:
            continue
        x = np.arange(len(val))
        if key == "SGD":
            val[0] = val[1]
        if key in bgd_methods:
            continue
        else:
            plt.plot(x, val, marker=".", label=key, alpha=0.75)

    plt.legend(loc="lower left", bbox_to_anchor=(1, 0.25), ncol=1)
    # plt.legend(loc="lower center", bbox_to_anchor=(0.5, 0.4), ncol=3)
    plt.title("Time spent per epoch")
    plt.xlabel("epoch")
    plt.ylabel("s")
    plt.tight_layout()
    plt.savefig("figures/times.pgf", bbox_inches="tight", pad_inches=0)

    plt.show()


def print_ab():
    plt.style.use("seaborn")
    with open("history/optimizers/models_ab3_cifar10.json") as json_file:
        data = json.load(json_file)
    table = {" beta \ alpha ->": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
    frame = {}
    for alpha, dictionary in data.items():
        col = []
        frame[alpha] = []
        for beta, dic in dictionary.items():

            acc = dic["acc_valid"]
            acc_final = acc[-1]
            frame[alpha].append(acc_final)
            col.append(f"{acc_final:.2f}")
        table[alpha] = col

    df = pd.DataFrame(frame, index=[i / 10 for i in range(1, 10)])
    df = df.transpose()
    # display(df)

    print(df.to_string())
    plt.figure(figsize=set_size(345))
    sns.heatmap(df, annot=True, cbar=False)
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$\alpha$")
    plt.show()
    print(tabulate(table, headers="keys", tablefmt="fancy_grid"))
    print(tabulate(df, headers="keys", tablefmt="fancy_grid"))

    print(tabulate(df, headers="keys", tablefmt="latex"))

    plt.savefig("figures/alpha_beta.pgf", bbox_inches="tight", pad_inches=0)


if __name__ == "__main__":
    # plot_acc()
    # print_lr_acc()
    # print_lr_loss()

    plot_time()
    # print_ab()
