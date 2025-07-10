import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def plot_missing_values_percentage(df, figsize=(15, 10)):
    # Вычисление доли пропусков для каждого столбца
    missing_values = df.isnull().mean() * 100

    # Сортировка столбцов по доле пропусков
    missing_values = missing_values.sort_values(ascending=False)

    # Построение графика
    plt.figure(figsize=figsize)
    missing_values.plot(kind="bar")
    plt.title("Доля пропусков в процентах для каждого столбца")
    plt.xlabel("Столбцы")
    plt.ylabel("Доля пропусков (%)")
    plt.xticks(rotation=90)
    plt.yticks(range(0, 101, 5))
    plt.tight_layout()
    plt.show()
    return missing_values


def eda_analysis(df, drop_columns=None, output_dir="eda_images"):
    """
    Сохраняет визуализацию EDA в отдельные изображения в указанной папке.

    Parameters:
    df (pd.DataFrame): Входной DataFrame.
    drop_columns (list): Столбцы для удаления перед анализом.
    output_dir (str): Путь к папке для сохранения изображений (по умолчанию "eda_images").
    Returns:
    None
    """
    if drop_columns is not None:
        df = df.drop(columns=drop_columns)

    # Создаем папку, если её нет
    os.makedirs(output_dir, exist_ok=True)

    # 1. Форма DataFrame
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        f"Shape of the DataFrame: {df.shape}",
        fontsize=12,
        ha="center",
        va="center",
    )
    ax.set_title("Shape of the DataFrame")
    fig.savefig(os.path.join(output_dir, "data_shape.png"))
    plt.close(fig)

    # 2. Типы данных
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")
    ax.table(
        cellText=pd.DataFrame(df.dtypes).T.values,
        colLabels=df.columns,
        rowLabels=["Data Type"],
        cellLoc="center",
        loc="center",
    )
    ax.set_title("Data types of each column")
    fig.savefig(os.path.join(output_dir, "data_types.png"))
    plt.close(fig)

    # 3. Пропущенные значения
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")
    ax.table(
        cellText=pd.DataFrame(df.isnull().sum()).T.values,
        rowLabels=["Missing Values"],
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
    )
    ax.set_title("Number of missing values in each column")
    fig.savefig(os.path.join(output_dir, "missing_values.png"))
    plt.close(fig)

    # 4. Визуализация распределений для каждого столбца
    for column in tqdm(df.columns):
        try:
            fig = plt.figure(figsize=(12, 6))
            if df[column].dtype == "object":
                # Гистограмма для категорий
                sns.histplot(y=df[column], stat="percent")
                plt.title(f"Distribution of {column}")
                plt.xlabel("Count")
            else:
                # Гистограмма + ящик с усами для чисел
                plt.subplot(1, 2, 1)
                sns.histplot(df[column], bins=100, stat="percent")
                plt.title(f"Histogram of {column}")
                plt.xlabel(column)
                plt.ylabel("Frequency")

                plt.subplot(1, 2, 2)
                sns.boxplot(y=df[column])
                plt.title(f"Boxplot of {column}")
            plt.ylabel(column)
            plt.tight_layout()
            filename = f"{column}_distribution.png"
            fig.savefig(os.path.join(output_dir, filename))
            plt.close(fig)
        except Exception as e:
            print(f"Ошибка при обработке столбца '{column}':\n{str(e)}")

    print(f"Все изображения сохранены в папку {output_dir}")


def replace_binary(features: pd.DataFrame) -> pd.DataFrame:
    categorical_features = features.select_dtypes(include=["object"])
    print("categorical_features", len(categorical_features.columns))
    nux = categorical_features.nunique()
    nux = nux[nux <= 2].to_dict()
    print("nux\n", nux)
    print(len(nux))
    for col, value in nux.items():
        if value == 2:
            features[col] = pd.to_numeric(features[col], errors="coerce")
        elif value < 2:
            features = features.drop(columns=[col], errors="ignore")
    print(
        "new_categorical_features",
        len(features.select_dtypes(include=["object"]).columns),
    )
    return features
