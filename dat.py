import base64
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# This module builds charts for the web app and converts them into images that can be shown in HTML.

def fig_to_base64(fig):
    """Convert a Matplotlib figure into a base64 string for embedding in HTML."""
    # Save the chart into an in-memory image buffer so it can be embedded in the page.
    img = BytesIO()
    fig.savefig(img, format="png", bbox_inches="tight", facecolor="#0d0d0d")
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close(fig)
    return plot_url


def create_cpu_category_pie(cpu_info):
    """Create a pie chart of CPU distribution by category."""
    # Group the CPU data by category and count how many items fall into each group.
    categories = [info["category"] for info in cpu_info]
    category_counts = pd.Series(categories).value_counts()

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="#0d0d0d")
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
    ax.pie(
        category_counts.values,
        labels=category_counts.index,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        textprops={"color": "#e6e6e6", "fontsize": 11, "weight": "bold"},
    )
    ax.set_title("CPU Distribution by Category", color="#f8fafc", fontsize=14, weight="bold", pad=20)
    return fig_to_base64(fig)


def create_cpu_benchmark_histogram(cpu_info):
    """Create a histogram of CPU benchmarks."""
    # Keep only CPUs with a benchmark value so the histogram reflects real performance data.
    marks = [info["cpuMark"] for info in cpu_info if info["cpuMark"]]

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="#0d0d0d")
    ax.hist(marks, bins=40, color="#FF6B6B", edgecolor="#FFB3BA", linewidth=1.5, alpha=0.8)
    ax.set_xlabel("CPU Mark", color="#e6e6e6", fontsize=11, weight="bold")
    ax.set_ylabel("Frequency", color="#e6e6e6", fontsize=11, weight="bold")
    ax.set_title("CPU Benchmark Distribution", color="#f8fafc", fontsize=14, weight="bold", pad=20)
    ax.set_facecolor("#070707")
    ax.spines["left"].set_color("#2d2d2d")
    ax.spines["bottom"].set_color("#2d2d2d")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(colors="#e6e6e6")
    ax.grid(axis="y", alpha=0.2, linestyle="--", color="#4b4b4b")
    return fig_to_base64(fig)


def create_cpu_category_bar(cpu_info):
    """Create a seaborn bar chart for average CPU benchmark by category."""
    # Stop early if there is no CPU data to summarize.
    if not cpu_info:
        return None

    df = pd.DataFrame(cpu_info)
    summary = df.groupby("category", sort=False)["cpuMark"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="#0d0d0d")
    palette = sns.color_palette("husl", n_colors=len(summary))
    sns.barplot(data=summary, x="category", y="cpuMark", hue="category", palette=palette, dodge=False, ax=ax, legend=False)
    ax.set_title("Average CPU Benchmark by Category", color="#f8fafc", fontsize=14, weight="bold", pad=20)
    ax.set_xlabel("Category", color="#e6e6e6", fontsize=11, weight="bold")
    ax.set_ylabel("Average CPU Mark", color="#e6e6e6", fontsize=11, weight="bold")
    ax.set_facecolor("#070707")
    ax.spines["left"].set_color("#2d2d2d")
    ax.spines["bottom"].set_color("#2d2d2d")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(colors="#e6e6e6")
    ax.grid(axis="y", alpha=0.2, linestyle="--", color="#4b4b4b")
    return fig_to_base64(fig)


def create_cores_vs_tdp_scatter(cpu_info):
    """Create a scatter plot of CPU cores vs TDP."""
    # Collect the core count, power draw, and score for each CPU to compare them visually.
    cores = [info["cores"] for info in cpu_info if info["cores"]]
    tdp = [info["TDP"] for info in cpu_info if info["TDP"]]
    marks = [info["cpuMark"] for info in cpu_info if info["cpuMark"]]

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="#0d0d0d")
    scatter = ax.scatter(
        cores[: len(tdp)],
        tdp,
        c=marks[: len(tdp)],
        cmap="viridis",
        s=80,
        alpha=0.7,
        edgecolors="#4ECDC4",
        linewidth=1,
    )
    ax.set_xlabel("Number of Cores", color="#e6e6e6", fontsize=11, weight="bold")
    ax.set_ylabel("TDP (Watts)", color="#e6e6e6", fontsize=11, weight="bold")
    ax.set_title("CPU Cores vs TDP (colored by cpuMark)", color="#f8fafc", fontsize=14, weight="bold", pad=20)
    ax.set_facecolor("#070707")
    ax.spines["left"].set_color("#2d2d2d")
    ax.spines["bottom"].set_color("#2d2d2d")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(colors="#e6e6e6")
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("CPU Mark", color="#e6e6e6", fontsize=10, weight="bold")
    cbar.ax.tick_params(colors="#e6e6e6")
    return fig_to_base64(fig)


def create_gpu_category_pie(gpu_info):
    """Create a pie chart of GPU distribution by category."""
    # Count how many GPUs fall into each category for the pie slices.
    categories = [info["category"] for info in gpu_info]
    category_counts = pd.Series(categories).value_counts()

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="#0d0d0d")
    colors = ["#FFB347", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E2"]
    ax.pie(
        category_counts.values,
        labels=category_counts.index,
        autopct="%1.1f%%",
        colors=colors[: len(category_counts)],
        startangle=45,
        textprops={"color": "#e6e6e6", "fontsize": 10, "weight": "bold"},
    )
    ax.set_title("GPU Distribution by Category", color="#f8fafc", fontsize=14, weight="bold", pad=20)
    return fig_to_base64(fig)


def create_gpu_benchmark_histogram(gpu_info):
    """Create a histogram of GPU benchmarks."""
    # Retain only GPUs that have a benchmark score to build a meaningful histogram.
    marks = [info["G3Dmark"] for info in gpu_info if info["G3Dmark"]]

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="#0d0d0d")
    ax.hist(marks, bins=35, color="#4ECDC4", edgecolor="#7FDFDA", linewidth=1.5, alpha=0.8)
    ax.set_xlabel("G3D Mark", color="#e6e6e6", fontsize=11, weight="bold")
    ax.set_ylabel("Frequency", color="#e6e6e6", fontsize=11, weight="bold")
    ax.set_title("GPU Benchmark Distribution", color="#f8fafc", fontsize=14, weight="bold", pad=20)
    ax.set_facecolor("#070707")
    ax.spines["left"].set_color("#2d2d2d")
    ax.spines["bottom"].set_color("#2d2d2d")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(colors="#e6e6e6")
    ax.grid(axis="y", alpha=0.2, linestyle="--", color="#4b4b4b")
    return fig_to_base64(fig)


def create_gpu_category_bar(gpu_info):
    """Create a seaborn bar chart for average GPU benchmark by category."""
    # Exit early when no GPU data is available for aggregation.
    if not gpu_info:
        return None

    df = pd.DataFrame(gpu_info)
    summary = df.groupby("category", sort=False)["G3Dmark"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="#0d0d0d")
    palette = sns.color_palette("viridis", n_colors=len(summary))
    sns.barplot(data=summary, x="category", y="G3Dmark", hue="category", palette=palette, dodge=False, ax=ax, legend=False)
    ax.set_title("Average GPU Benchmark by Category", color="#f8fafc", fontsize=14, weight="bold", pad=20)
    ax.set_xlabel("Category", color="#e6e6e6", fontsize=11, weight="bold")
    ax.set_ylabel("Average G3D Mark", color="#e6e6e6", fontsize=11, weight="bold")
    ax.set_facecolor("#070707")
    ax.spines["left"].set_color("#2d2d2d")
    ax.spines["bottom"].set_color("#2d2d2d")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(colors="#e6e6e6")
    ax.grid(axis="y", alpha=0.2, linestyle="--", color="#4b4b4b")
    return fig_to_base64(fig)


def create_tdp_vs_price_scatter(gpu_info):
    """Create a scatter plot of GPU TDP vs price."""
    # Collect only GPUs that include TDP, price, and benchmark values for a complete comparison.
    tdps = []
    prices = []
    marks = []

    for info in gpu_info:
        if info["TDP"] and info["price"] and info["G3Dmark"]:
            tdps.append(info["TDP"])
            prices.append(info["price"])
            marks.append(info["G3Dmark"])

    if tdps and prices and marks:
        fig, ax = plt.subplots(figsize=(8, 6), facecolor="#0d0d0d")
        scatter = ax.scatter(tdps, prices, c=marks, cmap="plasma", s=80, alpha=0.7, edgecolors="#FFB347", linewidth=1)
        ax.set_xlabel("TDP (Watts)", color="#e6e6e6", fontsize=11, weight="bold")
        ax.set_ylabel("Price ($)", color="#e6e6e6", fontsize=11, weight="bold")
        ax.set_title("GPU TDP vs Price (colored by G3D Mark)", color="#f8fafc", fontsize=14, weight="bold", pad=20)
        ax.set_facecolor("#070707")
        ax.spines["left"].set_color("#2d2d2d")
        ax.spines["bottom"].set_color("#2d2d2d")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.tick_params(colors="#e6e6e6")
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("G3D Mark", color="#e6e6e6", fontsize=10, weight="bold")
        cbar.ax.tick_params(colors="#e6e6e6")
        return fig_to_base64(fig)
    return None


def create_cpu_boxplot(cpu_info):
    """Create a horizontal bar chart for the top 12 CPUs by benchmark."""
    # Show the strongest CPUs first so the chart highlights the best-performing models.
    if not cpu_info:
        return None

    df = pd.DataFrame(cpu_info)
    top_cpus = df.nlargest(12, "cpuMark")[["name", "cpuMark"]].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 7), facecolor="#0d0d0d")
    colors = sns.color_palette("rainbow", n_colors=len(top_cpus))
    y_pos = np.arange(len(top_cpus))
    ax.barh(y_pos, top_cpus["cpuMark"].values, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([name[:30] for name in top_cpus["name"].values], fontsize=9, color="#e6e6e6")
    ax.set_xlabel("CPU Mark", color="#e6e6e6", fontsize=11, weight="bold")
    ax.set_title("Top 12 CPUs by Benchmark Score", color="#f8fafc", fontsize=14, weight="bold", pad=20)
    ax.set_facecolor("#070707")
    ax.spines["left"].set_color("#2d2d2d")
    ax.spines["bottom"].set_color("#2d2d2d")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(colors="#e6e6e6")
    ax.grid(axis="x", alpha=0.2, linestyle="--", color="#4b4b4b")
    return fig_to_base64(fig)


def create_gpu_boxplot(gpu_info):
    """Create a horizontal bar chart for the top 12 GPUs by benchmark."""
    # Show the highest-scoring GPUs first for a quick performance comparison.
    if not gpu_info:
        return None

    df = pd.DataFrame(gpu_info)
    top_gpus = df.nlargest(12, "G3Dmark")[["name", "G3Dmark"]].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 7), facecolor="#0d0d0d")
    colors = sns.color_palette("cool", n_colors=len(top_gpus))
    y_pos = np.arange(len(top_gpus))
    ax.barh(y_pos, top_gpus["G3Dmark"].values, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([name[:30] for name in top_gpus["name"].values], fontsize=9, color="#e6e6e6")
    ax.set_xlabel("G3D Mark", color="#e6e6e6", fontsize=11, weight="bold")
    ax.set_title("Top 12 GPUs by Benchmark Score", color="#f8fafc", fontsize=14, weight="bold", pad=20)
    ax.set_facecolor("#070707")
    ax.spines["left"].set_color("#2d2d2d")
    ax.spines["bottom"].set_color("#2d2d2d")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(colors="#e6e6e6")
    ax.grid(axis="x", alpha=0.2, linestyle="--", color="#4b4b4b")
    return fig_to_base64(fig)
