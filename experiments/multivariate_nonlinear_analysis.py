import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
from category_encoders.target_encoder import TargetEncoder
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from umap import UMAP
from matplotlib.backends.backend_pdf import PdfPages

import sys

sys.path.append("..")

from utils.dataset import ModelData


def prepare_data(md):
    pbar.desc = "Prepare data"
    md.data = md.data[[c for c in md.data.columns if md.data[c].nunique() > 1]]
    md.features = md.data.columns
    md.data = target_compression(md.data, ["time", "M2"])
    md.target = "target"
    encoder = md.encode_categorical(TargetEncoder(), y=md.get_data("all", "target"))
    pbar.update()
    return md


def target_compression(data: pd.DataFrame, targets: list[str]):
    target_space = data[targets]
    target_space = StandardScaler().fit_transform(target_space)
    data["target"] = UMAP(n_components=1).fit_transform(target_space)
    return data


def compute_mutual_info(df):
    features = df.columns
    mi_matrix = np.zeros((len(features), len(features)))
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            x = df[features[i]].values
            y = df[features[j]].values
            mi = mutual_info_regression(x.reshape(-1, 1), y)[0]
            mi_matrix[i, j] = mi
            mi_matrix[j, i] = mi  # Симметричная матрица
    return mi_matrix


def visualize_mutual_info(mi_matrix, features, pdf):
    pbar.desc = "Mutual Information Matrix"
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        mi_matrix,
        annot=True,
        fmt=".2f",
        xticklabels=features,
        yticklabels=features,
        cmap="bwr",
    )
    plt.title("Mutual Information Matrix (Nonlinear)")
    pdf.savefig()  # Сохраняем в PDF
    plt.close()  # Закрываем текущую figure
    pbar.update()


def visualize_loess(md, pdf):
    pbar.desc = "LOESS"
    g = sns.PairGrid(md.data.sample(500) if len(md.data) > 500 else md.data)
    g.map_diag(sns.histplot)
    g.map_upper(plt.scatter)
    g.map_lower(sns.kdeplot, cmap="Blues_d")
    g.map_offdiag(sns.regplot, lowess=True, scatter_kws={"alpha": 0.3})
    pdf.savefig()  # Сохраняем в PDF
    plt.close()  # Закрываем текущую figure
    pbar.update()


def visualize_umap(md, pdf):
    pbar.desc = "UMAP Visualization of High-Dimensional Data"
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(md.data)
    tsne = UMAP(n_components=2)
    X_tsne = tsne.fit_transform(X_scaled)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        X_tsne[:, 0],
        X_tsne[:, 1],
        alpha=0.6,
        c=md.get_data("all", "target"),
        cmap="bwr",
    )
    cbar = plt.colorbar(scatter)
    cbar.set_label(f"{md.target} Value")
    plt.title("UMAP Visualization of High-Dimensional Data")
    pdf.savefig()  # Сохраняем в PDF
    plt.close()  # Закрываем текущую figure
    pbar.update()


def visualize_linear_correlation(md, pdf):
    pbar.desc = "Linear Correlation Matrix"
    corr_matrix = md.data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="bwr", center=0)
    plt.title("Linear Correlation Matrix")
    pdf.savefig()  # Сохраняем в PDF
    plt.close()  # Закрываем текущую figure
    pbar.update()


# Main execution
if __name__ == "__main__":
    filepath = r"C:\Projects\HypEx\examples\experiments\performance_test\aa_performance_test_result.csv"
    pbar = tqdm(total=9, desc="Load data")
    md = ModelData(filepath)
    pbar.update()

    pbar.desc = "Prepare data"
    md = prepare_data(md)
    pbar.update()

    pbar.desc = "Compute mutual information"
    mi_matrix = compute_mutual_info(md.data)
    pbar.update()

    with PdfPages("output.pdf") as pdf:
        pbar.desc = "Visualize mutual information"
        visualize_mutual_info(mi_matrix, md.features, pdf)
        pbar.update()

        pbar.desc = "Compute and visualize linear correlation"
        visualize_linear_correlation(md, pdf)
        pbar.update()

        pbar.desc = "Compute and visualize LOESS"
        visualize_loess(md, pdf)
        pbar.update()

        pbar.desc = "Compute and visualize UMAP"
        visualize_umap(md, pdf)
        pbar.update()

    pbar.close()
