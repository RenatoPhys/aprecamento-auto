import polars as pl
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------
# Helper: detect numeric vs categorical columns
# ---------------------------------------------------
def split_columns(df: pl.DataFrame, time_col: str):
    numeric = []
    categorical = []
    for c, dt in zip(df.columns, df.dtypes):
        if c == time_col:
            continue
        if dt in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
            numeric.append(c)
        else:
            categorical.append(c)
    return numeric, categorical


# ---------------------------------------------------
# Numerical drift: null% + non-null mean
# ---------------------------------------------------
def drift_numeric(df: pl.DataFrame, time_col: str):
    num_cols, _ = split_columns(df, time_col)

    # % nulls
    null_exprs = [(pl.col(c).is_null().mean() * 100).alias(c) for c in num_cols]
    nulls = df.groupby(time_col).agg(null_exprs).sort(time_col)
    nulls_long = nulls.melt(id_vars=time_col, variable_name="col", value_name="null_pct")

    # mean non-null
    mean_exprs = [pl.col(c).mean().alias(c) for c in num_cols]
    means = df.groupby(time_col).agg(mean_exprs).sort(time_col)
    means_long = means.melt(id_vars=time_col, variable_name="col", value_name="mean")

    return nulls_long, means_long


# ---------------------------------------------------
# Categorical drift: null% + top-k categories
# ---------------------------------------------------
def drift_categorical(df: pl.DataFrame, time_col: str, top_k=5):
    _, cat_cols = split_columns(df, time_col)

    # 1. % nulls
    null_exprs = [(pl.col(c).is_null().mean() * 100).alias(c) for c in cat_cols]
    nulls = df.groupby(time_col).agg(null_exprs).sort(time_col)
    nulls_long = nulls.melt(id_vars=time_col, variable_name="col", value_name="null_pct")

    # 2. Top-k category frequencies
    freq_dfs = []

    for c in cat_cols:
        freq = (
            df
            .groupby([time_col, c])
            .agg(pl.count().alias("count"))
            .with_columns([
                pl.col("count") / pl.col("count").sum().over(time_col) * 100
            ])
            .rename({"count": "pct"})
            .sort(["pct"], descending=True)
        )

        # Keep top-k per month
        freq_topk = (
            freq
            .with_columns(
                pl.rank("dense", descending=True).over(time_col).alias("rank")
            )
            .filter(pl.col("rank") <= top_k)
            .drop("rank")
            .rename({c: "category"})
            .with_columns(pl.lit(c).alias("column"))
        )

        freq_dfs.append(freq_topk)

    freq_final = pl.concat(freq_dfs)

    return nulls_long, freq_final


# ---------------------------------------------------
# Optional: Entropy drift (for categorical)
# ---------------------------------------------------
def categorical_entropy(df, column):
    """Entropy ignoring nulls."""
    freq = (
        df.filter(pl.col(column).is_not_null())
        .groupby(column)
        .count()
        .with_columns((pl.col("count") / pl.col("count").sum()).alias("p"))
        .select("p")
        .to_series()
        .to_list()
    )
    return -np.sum([p * np.log(p) for p in freq])


# ---------------------------------------------------
# Visualization
# ---------------------------------------------------
def plot_numeric_drift(nulls_long, means_long, figsize=(14, 6)):
    num_cols = nulls_long["col"].unique().to_list()

    fig, axes = plt.subplots(len(num_cols), 2, figsize=figsize, sharex=True)
    if len(num_cols) == 1:
        axes = np.array([axes])

    for i, col in enumerate(num_cols):
        df_null = nulls_long.filter(pl.col("col") == col)
        df_mean = means_long.filter(pl.col("col") == col)

        axes[i, 0].plot(df_null["anomes"], df_null["null_pct"])
        axes[i, 0].set_title(f"{col} – % nulls")

        axes[i, 1].plot(df_mean["anomes"], df_mean["mean"])
        axes[i, 1].set_title(f"{col} – mean (non-null)")

    plt.tight_layout()
    plt.show()


def plot_categorical_drift(nulls_long, freq_topk, figsize=(14, 6)):
    cat_cols = nulls_long["col"].unique().to_list()

    for col in cat_cols:
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Null %
        df_null = nulls_long.filter(pl.col("col") == col)
        axes[0].plot(df_null["anomes"], df_null["null_pct"])
        axes[0].set_title(f"{col} – % nulls")

        # Category frequencies
        df_freq = freq_topk.filter(pl.col("column") == col)
        for cat in df_freq["category"].unique():
            tmp = df_freq.filter(pl.col("category") == cat)
            axes[1].plot(tmp["anomes"], tmp["pct"], label=str(cat))

        axes[1].set_title(f"{col} – Top-k category frequencies")
        axes[1].legend()

        plt.tight_layout()
        plt.show()


# ---------------------------------------------------
# MASTER FUNCTION: RUN EVERYTHING
# ---------------------------------------------------
def data_drift_dashboard(df: pl.DataFrame, time_col="anomes", top_k=5):

    numeric_nulls, numeric_means = drift_numeric(df, time_col)
    cat_nulls, cat_freqs = drift_categorical(df, time_col, top_k=top_k)

    print("📊 Plotting numeric drift...")
    if numeric_nulls.height > 0:
        plot_numeric_drift(numeric_nulls, numeric_means)
    else:
        print("No numeric columns detected.")

    print("📊 Plotting categorical drift...")
    if cat_nulls.height > 0:
        plot_categorical_drift(cat_nulls, cat_freqs)
    else:
        print("No categorical columns detected.")

    return numeric_nulls, numeric_means, cat_nulls, cat_freqs



######### PSI

def psi_over_time(
    df: pl.DataFrame,
    time_col: str = "anomes",
    score_col: str = "score",
    n_bins: int = 10,
    method: str = "quantile",    # "quantile" or "uniform"
    custom_bins: list = None
):
    # Get ordered list of periods
    periods = df[time_col].unique().sort().to_list()

    # Determine reference period
    ref_period = periods[0]
    ref_df = df.filter(pl.col(time_col) == ref_period)[score_col]

    # ----- Create bins -----
    if method == "quantile":
        # use quantiles from reference distribution
        qs = np.linspace(0, 1, n_bins + 1)
        bins = np.quantile(ref_df.to_numpy(), qs)

    elif method == "uniform":
        bins = np.linspace(
            df[score_col].min(),
            df[score_col].max(),
            n_bins + 1
        )

    elif method == "custom":
        assert custom_bins is not None, "Pass custom_bins when method='custom'"
        bins = custom_bins

    else:
        raise ValueError("Invalid method. Use quantile, uniform, or custom.")

    # remove duplicates (happens if score has no variance)
    bins = np.unique(bins)

    # ----- helper that returns the % in each bin -----
    def get_dist(s: pl.Series):
        arr = s.to_numpy()
        hist, _ = np.histogram(arr, bins=bins)
        return hist / hist.sum()

    # reference distribution
    ref_dist = get_dist(ref_df)

    # Collect PSI for all periods
    psi_rows = []

    for p in periods:
        cur_df = df.filter(pl.col(time_col) == p)[score_col]
        cur_dist = get_dist(cur_df)
        psi_value = psi_single(ref_dist, cur_dist)
        psi_rows.append((p, psi_value))

    return pl.DataFrame({"anomes": [r[0] for r in psi_rows],
                         "psi":    [r[1] for r in psi_rows]})

