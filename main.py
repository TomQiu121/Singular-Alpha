import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# df has features Close, High, Low, Open, Volume
def download_stock_data(ticker, start="2018-01-01", end="2025-01-01"):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    df = df.dropna()
    return df

def construct_features(price_df, vix_df=None, horizon=5):
    df = pd.DataFrame(index=price_df.index)

    close = price_df["Close"]
    high = price_df["High"]
    low = price_df["Low"]
    volume = price_df["Volume"]

    # Add lagged return features
    returns = close.pct_change()
    df["return_1"] = returns.shift(1)
    df["return_2"] = returns.shift(2)
    df["return_5"] = returns.shift(5)
    df["return_10"] = returns.shift(10)

    # Add realized volatility features
    df["vol_5"] = returns.rolling(5).std().shift(1)
    df["vol_10"] = returns.rolling(10).std().shift(1)
    df["vol_20"] = returns.rolling(20).std().shift(1)

    # Add range
    df["range"] = ((high-low) / close).shift(1)

    # volume features
    df["volume_change"] = volume.pct_change().shift(1)

    if vix_df is not None:
        vix = vix_df["Close"].reindex(df.index).ffill()
        df["vix"] = vix.shift(1)
        df["vix_change"] = vix.pct_change().shift(1)

    # Add target feature
    df["target"] = close.shift(-horizon) / close - 1

    return df.dropna()


def svd_regression_fit(X, y, rcond=1e-10):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)

    s_inv = np.array([1 / val if val > rcond else 0 for val in s])

    beta = Vt.T @ np.diag(s_inv) @ U.T @ y
    return beta, s

def svd_regression_predict(X, beta):
    return X @ beta


def pca_regression_fit(X, y, n_components=5):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)

    V_k = Vt[:n_components].T
    Z = X @ V_k

    Z_aug = np.column_stack([np.ones(len(Z)), Z])

    beta_pca, _ = svd_regression_fit(Z_aug, y)

    explained_variance = s[:n_components]**2 / np.sum(s**2)
    explained_variance_ratio = explained_variance.sum()

    return {
        "V_k": V_k,
        "beta_pca": beta_pca,
        "singular_values": s,
        "explained_variance_ratio": explained_variance_ratio,
    }

def pca_regression_predict(X, model):
    Z = X @ model["V_k"]
    Z_aug = np.column_stack([np.ones(len(Z)), Z])
    return Z_aug @ model["beta_pca"]


def laplacian_eigenmaps_fit(X, n_components=5, n_neighbors=10, sigma=None):
    n = X.shape[0]

    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(X)
    distances, indices = nbrs.kneighbors(X)

    # Ignore itself
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    if sigma is None:
        sigma = np.median(distances)

    W = np.zeros((n, n))

    for i in range(n):
        for dist, j in zip(distances[i], indices[i]):
            weight = np.exp(-(dist**2)/(2*sigma**2))
            W[i, j] = weight
            W[j, i] = weight
    D = np.diag(W.sum(axis=1))
    L = D - W

    # Solve Lz = Lambda Dz
    eigenvalues, eigenvectors = eigh(L, D)
    
    # Sort eigenvalues
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    Z = eigenvectors[:, 1:n_components+1]

    return {
        "Z": Z, 
        "X_train": X,
        "eigenvalues": eigenvalues,
        "sigma": sigma,
        "n_neighbors": n_neighbors
    }

def laplacian_eigenmaps_transform(X_test, model):
    X_train = model["X_train"]
    Z_train = model["Z"]
    sigma = model["sigma"]
    n_neighbors = model["n_neighbors"]

    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X_train)
    distances, indices = nbrs.kneighbors(X_test)

    Z_test = []
    for dists, neigh_idx in zip(distances, indices):
        weights = np.exp(-(dists**2)/(2*sigma**2))
        weights = weights / weights.sum()

        z = weights @ Z_train[neigh_idx]
        Z_test.append(z)
    
    return np.array(Z_test)

def laplacian_regression_fit(X, y, n_components=5, n_neighbors=10):
    le_model = laplacian_eigenmaps_fit(X, n_components=n_components, n_neighbors=n_neighbors)

    Z = le_model["Z"]
    Z_aug = np.column_stack([np.ones(len(Z)), Z])

    beta_le, _ = svd_regression_fit(Z_aug, y)
    le_model["beta_le"] = beta_le
    return le_model

def laplacian_regression_predict(X_test, model):
    Z_test = laplacian_eigenmaps_transform(X_test, model)
    Z_test_aug = np.column_stack([np.ones(len(Z_test)), Z_test])

    return Z_test_aug @ model["beta_le"]


def rolling_experiment(data, window=252, pca_components=5, le_components=5, le_neighbors=10):
    features = [c for c in data.columns if c != "target"]
    rows = []


    for t in range(window, len(data)):
        train = data.iloc[t-window:t]
        test = data.iloc[t:t+1]

        X_train = train[features].values
        y_train = train["target"].values

        X_test = test[features].values
        y_test = test["target"].values[0]

        # Standardize using training data only
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # full SVD regression
        X_train_aug = np.column_stack([np.ones(len(X_train_scaled)), X_train_scaled])
        X_test_aug = np.column_stack([np.ones(len(X_test_scaled)), X_test_scaled])

        beta_full, singular_values = svd_regression_fit(X_train_aug, y_train)
        pred_full = svd_regression_predict(X_test_aug, beta_full)[0]

        # condition number of standardized feature matrix
        feature_singular_values = np.linalg.svd(X_train_scaled, compute_uv=False)
        condition_number = feature_singular_values[0] / feature_singular_values[-1]

        # PCA regression
        pca_model = pca_regression_fit(X_train_scaled, y_train, n_components=pca_components)
        pred_pca = pca_regression_predict(X_test_scaled, pca_model)[0]

        # Laplacian Eigenmaps regression
        try:
            le_model = laplacian_regression_fit(X_train_scaled, y_train, n_components=le_components, n_neighbors=le_neighbors)
            pred_le = laplacian_regression_predict(X_test_scaled, le_model)[0]
            le_beta_norm = np.linalg.norm(le_model["beta_le"])
        except Exception:
            pred_le = np.nan
            le_beta_norm = np.nan

        
        rows.append({
            "date": data.index[t],
            "actual": y_test,
            "pred_full": pred_full,
            "pred_pca": pred_pca,
            "pred_le": pred_le,
            "condition_number": condition_number,
            "pca_explained_variance": pca_model["explained_variance_ratio"],
            "full_beta_norm": np.linalg.norm(beta_full),
            "pca_beta_norm": np.linalg.norm(pca_model["beta_pca"]),
            "le_beta_norm": le_beta_norm,
        })
    
    return pd.DataFrame(rows).set_index("date")


def evaluate_results(results):
    results = results.dropna()

    models = {
        "Full SVD": "pred_full",
        "PCA": "pred_pca",
        "Laplacian Eigenmaps": "pred_le",
    }

    summary = []

    for name, col in models.items():
        mse = mean_squared_error(results["actual"], results[col])
        direction_acc = np.mean(np.sign(results["actual"]) == np.sign(results[col]))
        corr = np.corrcoef(results["actual"], results[col])[0, 1]

        summary.append({
            "model": name,
            "mse": mse,
            "directional_accuracy": direction_acc,
            "prediction_correlation": corr,
        })
    
    return pd.DataFrame(summary)

ticker = "SPY"
window = 252
horizon = 5
pca_components = 5
le_components = 5
le_neighbors = 10


stock  = download_stock_data(ticker)
vix = yf.download("^VIX", start="2018-01-01", end="2025-01-01", auto_adjust=True)

data = construct_features(stock, vix_df=vix, horizon=5)

results = rolling_experiment(data, window=252, pca_components=pca_components, le_components=le_components, le_neighbors=le_neighbors)
print(results.head())

summary = evaluate_results(results)
print(summary)

plt.figure(figsize=(10, 5))
plt.plot(results["pca_explained_variance"])
plt.title("Variance Explained by Top PCA Components")
plt.xlabel("Date")
plt.ylabel("Explained Variance Ratio")
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(results["full_beta_norm"], label="Full SVD")
plt.plot(results["pca_beta_norm"], label="PCA")
plt.plot(results["le_beta_norm"], label="Laplacian Eigenmaps")
plt.title("Coefficient Norm Stability")
plt.xlabel("Date")
plt.ylabel("Coefficient Norm")
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(results["actual"], label="Actual", alpha=0.7)
plt.plot(results["pred_full"], label="Full SVD", alpha=0.7)
plt.plot(results["pred_pca"], label="PCA", alpha=0.7)
plt.plot(results["pred_le"], label="Laplacian Eigenmaps", alpha=0.7)
plt.title("Actual vs Predicted Returns")
plt.xlabel("Date")
plt.ylabel("Return")
plt.legend()
plt.show()