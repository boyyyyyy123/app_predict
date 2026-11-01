import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.cluster import KMeans

# Tải dữ liệu từ Yahoo Finance
@st.cache
def load_data(ticker):
    stock_data = yf.download(ticker, start="2020-01-01", end="2023-01-01")
    stock_data['EMA100'] = stock_data['Close'].ewm(span=100, adjust=False).mean()
    stock_data['RDP-5'] = stock_data['Close'].pct_change(5)
    stock_data['RDP-10'] = stock_data['Close'].pct_change(10)
    stock_data['RDP-15'] = stock_data['Close'].pct_change(15)
    stock_data['RDP-20'] = stock_data['Close'].pct_change(20)
    stock_data['RDP+5'] = stock_data['Close'].pct_change(-5)
    stock_data = stock_data.dropna()
    return stock_data

# Danh sách mã chứng khoán
stock_tickers = ['VCB',
 'BID',
 'FPT',
 'HPG',
 'GAS',
 'CTG',
 'VHM',
 'TCB',
 'VIC',
 'VPB',
 'VNM',
 'GVR',
 'MBB',
 'MSN',
 'ACB',
 'MWG',
 'LPB',
 'SAB',
 'HDB',
 'BCM',
 'STB',
 'PLX',
 'VJC',
 'VIB',
 'SSI',
 'SSB',
 'DGC',
 'VRE',
 'SHB',
 'TPB',
 'BVH',
 'POW',
 'EIB',
 'PNJ',
 'REE',
 'KDH',
 'OCB',
 'MSB',
 'GMD',
 'VND',
 'FRT',
 'VGC',
 'KBC',
 'VCI',
 'PDR',
 'HCM',
 'DCM',
 'GEX',
 'NLG',
 'PVD',
 'KDC',
 'CCT',
 'VHC',
 'DIG',
 'HSG',
 'VPI',
 'DPM',
 'TCH',
 'FTS',
 'HAG',
 'CMG',
 'VSH',
 'VIX',
 'VCG',
 'PVT',
 'DXG',
 'DGW',
 'BBS',
 'HDG',
 'BWE',
 'EEV',
 'PC1',
 'SBT',
 'KOS',
 'DBC',
 'PHR',
 'SCS',
 'BMP',
 'CTD',
 'SJS',
 'SZC',
 'BCG',
 'NKG',
 'NT2',
 'IMP',
 'HHV',
 'CII',
 'PAN',
 'PPC',
 'HT1',
 'PTB',
 'HDC',
 'AAA',
 'ANV',
 'TLG',
 'DXS',
 'ASM',
 'CRE',
 'DHC',
 'AGG']

# Lựa chọn mã chứng khoán
selected_ticker = st.selectbox("Chọn mã chứng khoán", stock_tickers)

# Tải dữ liệu
stock_data = load_data(selected_ticker)
X = stock_data[['EMA100', 'RDP-5', 'RDP-10', 'RDP-15', 'RDP-20']]
y = stock_data['RDP+5']

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Sử dụng Grid Search để tìm các tham số tốt nhất cho SVM
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1]
}

grid_search = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

best_C = grid_search.best_params_['C']
best_gamma = grid_search.best_params_['gamma']
st.write(f"Best C: {best_C}, Best gamma: {best_gamma}")

# Huấn luyện mô hình SVM với các tham số tốt nhất
svm_model = SVR(kernel='rbf', C=best_C, gamma=best_gamma)
svm_model.fit(X_train, y_train)

support_vectors = svm_model.support_vectors_
dual_coef = svm_model.dual_coef_[0]
support_vectors_combined = np.hstack((support_vectors, dual_coef.reshape(-1, 1)))

# Phân cụm các vector hỗ trợ bằng K-Means
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
support_vector_clusters = kmeans.fit_predict(support_vectors_combined)
cluster_centers = kmeans.cluster_centers_
means = cluster_centers[:, :-1]
sigmas = np.std(support_vectors, axis=0)

# Huấn luyện mô hình SVM cho mỗi cụm
models = {}
for cluster in range(n_clusters):
    cluster_indices = np.where(support_vector_clusters == cluster)[0]
    X_cluster = support_vectors[cluster_indices]
    y_cluster = dual_coef[cluster_indices]
    
    model = SVR(kernel='rbf', C=best_C, gamma=best_gamma)
    model.fit(X_cluster, y_cluster)
    models[cluster] = model

def gaussian_membership(x, mean, sigma):
    return np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))

def compute_memberships(X, means, sigmas):
    memberships = np.zeros((X.shape[0], means.shape[0]))
    for i in range(means.shape[0]):
        membership = np.ones(X.shape[0])
        for j in range(X.shape[1]):
            membership *= gaussian_membership(X[:, j], means[i, j], sigmas[j])
        memberships[:, i] = membership
    return memberships

def fuzzy_svm_predict(models, X, fuzzy_memberships):
    predictions = []
    for i in range(X.shape[0]):
        cluster_predictions = np.array([models[cluster].predict([X[i]])[0] for cluster in range(n_clusters)])
        weighted_prediction = np.dot(cluster_predictions, fuzzy_memberships[i]) / np.sum(fuzzy_memberships[i])
        predictions.append(weighted_prediction)
    return np.array(predictions)

fuzzy_memberships_val = compute_memberships(X_val, means, sigmas)
y_pred = fuzzy_svm_predict(models, X_val, fuzzy_memberships_val)

# Evaluation metrics
mse = mean_squared_error(y_val, y_pred)
nmse = mse / np.var(y_val)
mae = mean_absolute_error(y_val, y_pred)
ds = np.mean(np.abs((y_val - y_pred) / (y_val + y_pred)))

# Streamlit app
st.title("Fuzzy SVM Model Evaluation")
st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
st.write(f"**Normalized Mean Squared Error (NMSE):** {nmse:.2f}")
st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
st.write(f"**Difference in Symmetry (DS):** {ds:.2f}")

# Plot predictions vs actual values
st.write("### Predictions vs Actual Values")
results_df = pd.DataFrame({"Actual": y_val, "Predicted": y_pred})
st.line_chart(results_df)
