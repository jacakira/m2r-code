
# --------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from ripser import ripser
from matplotlib.dates import MonthLocator, DateFormatter

warnings.filterwarnings("ignore", category=UserWarning, module="ripser")
plt.rcParams["figure.dpi"] = 120

# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------
TICKERS = [
    "BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD",
    "DOGE-USD", "MATIC-USD", "DOT-USD", "SHIB-USD", "TRX-USD",
]
START_DATE = "2022-01-01"
END_DATE   = "2023-01-01"
WINDOWS    = (30, 60)           # rolling windows (days)

# --------------------------------------------------------------------------
# Download price data & make log-returns
# --------------------------------------------------------------------------
print("Downloading daily close prices …")
prices = (
    yf.download(TICKERS, start=START_DATE, end=END_DATE,
                progress=False, auto_adjust=False)["Close"]
      .dropna(how="all")
)

log_ret = np.log(prices).diff().dropna()


returns = pd.DataFrame(
    StandardScaler().fit_transform(log_ret),
    index=log_ret.index,
    columns=log_ret.columns,
)

# --------------------------------------------------------------------------
# Helper : H1 L1 / L2 norms for one window
# --------------------------------------------------------------------------
def h1_norms(matrix: np.ndarray) -> tuple[float, float]:
    diag = ripser(matrix, maxdim=1)["dgms"][1]          # H1 diagram
    finite = diag[np.isfinite(diag[:, 1])]
    life   = finite[:, 1] - finite[:, 0] if finite.size else np.empty(0)
    return life.sum(), np.sqrt((life ** 2).sum())
# --------------------------------------------------------------------------
# Event markers (add this near the top, after PARAMETERS)
# --------------------------------------------------------------------------
EVENT_DATES = [
    pd.Timestamp("2022-05-09"),   # LUNA de-peg
    pd.Timestamp("2022-11-11"),   # FTX bankruptcy filing
]
# --------------------------------------------------------------------------
# Helper : draw single series
# --------------------------------------------------------------------------
def plot_series(series: pd.Series, title: str, fname: str):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(series.index, series, lw=1.2)

    for ev in EVENT_DATES:
        ax.axvline(ev, color="red", linestyle="--", lw=1)

    ax.xaxis.set_major_locator(MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
    ax.xaxis.set_minor_locator(MonthLocator(interval=1, bymonthday=15))

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(title.split(" — ")[0])  # L1 or L2
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)

# --------------------------------------------------------------------------
# Main rolling computation & plotting
# --------------------------------------------------------------------------
for W in WINDOWS:
    print(f"Computing {W}-day rolling norms …")
    dates, L1, L2 = [], [], []

    for i in range(len(returns) - W + 1):
        seg = returns.iloc[i : i + W].values
        d   = returns.index[i + W - 1]
        l1, l2 = h1_norms(seg)
        dates.append(d); L1.append(l1); L2.append(l2)

    df = pd.DataFrame({"L1": L1, "L2": L2}, index=dates)
    df.to_csv(f"h1_norms_W{W}.csv")

    plot_series(df["L1"],
                f"L1 — Rolling {W}-day window",
                f"h1_L1_W{W}.png")
    plot_series(df["L2"],
                f"L2 — Rolling {W}-day window",
                f"h1_L2_W{W}.png")

print("✓ All CSV files & separate PNG figures saved.")

# --------------------------------------------------------------------------
# Extra: realised volatility & average correlation squared 
# --------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter


def realised_vol_paper(segment: np.ndarray) -> float:
    
    return (segment ** 2).sum(axis=0).mean()


def avg_corr_sq(segment: np.ndarray) -> float:
  
    C = np.corrcoef(segment.T)
    iu = np.triu_indices_from(C, k=1)  # strictly upper‑triangular indices
    rho_bar = C[iu].mean()             # \rho_t
    return rho_bar ** 2                # \rho_t^2


# ---- Helper: single‑axis time‑series plot ---------------------------------

def plot_metric(series: pd.Series, label: str, window: int, fname: str):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(series.index, series, lw=1.2)

    for ev in EVENT_DATES:
        ax.axvline(ev, color="red", linestyle="--", lw=1)

    ax.set_title(f"{label} — Rolling {window}-day window")
    ax.set_xlabel("Date")
    ax.set_ylabel(label)

    ax.xaxis.set_major_locator(MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
    ax.xaxis.set_minor_locator(MonthLocator(interval=1, bymonthday=15))

    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)


# ---- Main loop: 30‑day & 60‑day windows ----------------------------------
WINDOWS = (30, 60)

for W in WINDOWS:
    rv_list, corr_sq_list, dates = [], [], []

    for i in range(len(returns) - W + 1):
        seg = returns.iloc[i : i + W].values
        date = returns.index[i + W - 1]

        rv_list.append(realised_vol_paper(seg))
        corr_sq_list.append(avg_corr_sq(seg))  # *** use \rho_t^2 ***
        dates.append(date)

    df_rc = pd.DataFrame(
        {"RV": rv_list, "AVG_CORR_SQ": corr_sq_list}, index=dates
    )
    df_rc.to_csv(f"rv_corr_sq_W{W}.csv")

    # ------- Plot the two series separately -------
    plot_metric(df_rc["RV"], "Realised Volatility", W, f"rv_W{W}.png")
    plot_metric(
        df_rc["AVG_CORR_SQ"],
        "Average Correlation²",
        W,
        f"avg_corr_sq_W{W}.png",
    )

print("✓ Realised volatility & average correlation² files saved as separate figures.")
