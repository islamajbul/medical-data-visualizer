import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv("medical_examination.csv")

# 1. Add overweight column
bmi = df["weight"] / ((df["height"] / 100) ** 2)
df["overweight"] = (bmi > 25).astype(int)

# 2. Normalize cholesterol and gluc
df["cholesterol"] = (df["cholesterol"] > 1).astype(int)
df["gluc"] = (df["gluc"] > 1).astype(int)


def draw_cat_plot():
    # 3. Create DataFrame for cat plot
    df_cat = pd.melt(
        df,
        id_vars=["cardio"],
        value_vars=["cholesterol", "gluc", "smoke", "alco", "active", "overweight"],
    )

    # 4. Group and reformat data
    df_cat = (
        df_cat.groupby(["cardio", "variable", "value"])
        .size()
        .reset_index(name="total")
    )

    # 5. Draw the catplot
    fig = sns.catplot(
        x="variable",
        y="total",
        hue="value",
        col="cardio",
        data=df_cat,
        kind="bar",
    ).fig

    # Do not modify
    fig.savefig("catplot.png")
    return fig


def draw_heat_map():
    # 6. Clean data
    df_heat = df[
        (df["ap_lo"] <= df["ap_hi"])
        & (df["height"] >= df["height"].quantile(0.025))
        & (df["height"] <= df["height"].quantile(0.975))
        & (df["weight"] >= df["weight"].quantile(0.025))
        & (df["weight"] <= df["weight"].quantile(0.975))
    ]

    # 7. Correlation matrix
    corr = df_heat.corr()

    # 8. Generate mask
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 9. Set up figure
    fig, ax = plt.subplots(figsize=(12, 12))

    # 10. Draw heatmap
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".1f",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
    )

    # Do not modify
    fig.savefig("heatmap.png")
    return fig

