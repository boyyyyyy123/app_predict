import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.cluster import KMeans

# Tải dữ liệu từ Yahoo Finance (sử dụng st.cache_data thay cho @st.cache cũ)
@st.cache_data
def load_data(ticker):
    # Cổ phiếu Việt Nam trên Yahoo Finance cần thêm hậu tố .VN
    symbol = f"{ticker}.VN" if not ticker.endswith(".VN") else ticker
    stock_data = yf.download(symbol, start="2020-01-01", end="2023-01-01", progress=False)
    
    if stock_data.empty:
        return pd.DataFrame()
    
    # Xử lý trường hợp yfinance trả về MultiIndex columns
    if isinstance(stock_data.columns, pd.MultiIndex):
        close_series = stock_data['Close'][symbol] if symbol in stock_data['Close'] else stock_data['Close'].iloc[:, 0]
    else:
        close_series = stock_data['Close']

    df = pd.DataFrame(index=stock_data.index)
    df['Close'] = close_series
    df['EMA100'] = df['Close'].ewm(span=100, adjust=False).mean()
    df['RDP-5'] = df['Close'].pct_change(5)
    df['RDP-10'] = df['Close'].pct_change(10)
    df['RDP-15'] = df['Close'].pct_change(15)
    df['RDP-20'] = df['Close'].pct_change(20)
    df['RDP+5'] = df['Close'].pct_change(-5)
    
    df = df.dropna()
    return df

# Danh sách mã chứng khoán
stock_tickers = [
    'VCB', 'BID', 'FPT', 'HPG', 'GAS', 'CTG', 'VHM', 'TCB', 'VIC', 'VPB',
    'VNM', 'GVR', 'MBB', 'MSN', 'ACB', 'MWG', 'LPB', 'SAB', 'HDB', 'BCM',
    'STB', 'PLX', 'VJC', 'VIB', 'SSI', 'SSB', 'DGC', 'VRE', 'SHB', 'TPB',
    'BVH', 'POW', 'EIB', 'PNJ', 'REE', 'KDH', 'OCB', 'MSB', 'GMD', 'VND',
    'FRT', 'VGC', 'KBC', 'VCI', 'PDR', 'HCM', 'DCM', 'GEX', 'NLG', 'PVD',
    'KDC', 'CCT', 'VHC', 'DIG', 'HSG', 'VPI', 'DPM', 'TCH', 'FTS', 'HAG',
    'CMG', 'VSH', 'VIX', 'VCG', 'PVT', 'DXG', 'DGW', 'BBS', 'HDG', 'BWE',
    'EEV', 'PC1', 'SBT', 'KOS', 'DBC', 'PHR', 'SCS', 'BMP', 'CTD', 'SJS',
    'SZC', 'BCG', 'NKG', 'NT2', 'IMP', 'HHV', 'CII', 'PAN', 'PPC', 'HT1',
    'PTB', 'HDC', 'AAA', 'ANV', 'TLG', 'DXS', 'ASM', 'CRE', 'DHC', 'AGG'
]

# Giao diện chính
st.title("Fuzzy SVM Stock Price Prediction")
selected_ticker = st.selectbox("Chọn mã chứng khoán", stock_tickers)

# Tải dữ liệu
stock_data = load_data(selected_ticker)

if stock_data.empty or len(stock_data) < 50:
    st.error(f"Không tìm thấy dữ liệu giao dịch hoặc dữ liệu quá ít cho mã {selected_ticker}. Vui lòng thử mã khác!")
    st.stop()

X = stock_data[['EMA100', 'RDP-5', 'RDP-10', 'RDP-15', 'RDP-20']].values
y = stock_data['RDP+5'].values

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Grid Search tìm siêu tham số tối ưu
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1]
}

grid_search = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

best_C = grid_search.best_params_['C']
best_gamma = grid_search.best_params_['gamma']
st.write(f"**Best C:** {best_C} | **Best Gamma:** {best_gamma}")

# Huấn luyện mô hình SVM cơ sở
svm_model = SVR(kernel='rbf', C=best_C, gamma=best_gamma)
svm_model.fit(X_train, y_train)

support_vectors = svm_model.support_vectors_
dual_coef = svm_model.dual_coef_[0]
support_vectors_combined = np.hstack((support_vectors, dual_coef.reshape(-1, 1)))

# Phân cụm Support Vectors bằng K-Means
n_clusters = min(5, len(support_vectors))
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
support_vector_clusters = kmeans.fit_predict(support_vectors_combined)
cluster_centers = kmeans.cluster_centers_
means = cluster_centers[:, :-1]
sigmas = np.std(support_vectors, axis=0) + 1e-6  # Thêm epsilon tránh chia cho 0

# Huấn luyện mô hình cho từng cụm (xử lý an toàn cụm có ít phần tử)
models = {}
fallback_model = SVR(kernel='rbf', C=best_C, gamma=best_gamma).fit(support_vectors, dual_coef)

for cluster in range(n_clusters):
    cluster_indices = np.where(support_vector_clusters == cluster)[0]
    if len(cluster_indices) >= 2:
        X_cluster = support_vectors[cluster_indices]
        y_cluster = dual_coef[cluster_indices]
        model = SVR(kernel='rbf', C=best_C, gamma=best_gamma)
        model.fit(X_cluster, y_cluster)
        models[cluster] = model
    else:
        models[cluster] = fallback_model

def gaussian_membership(x, mean, sigma):
    return np.exp(-((x - mean) ** 2) / (2 * (sigma ** 2)))

def compute_memberships(X_input, means_mat, sigmas_vec):
    memberships = np.zeros((X_input.shape[0], means_mat.shape[0]))
    for i in range(means_mat.shape[0]):
        membership = np.ones(X_input.shape[0])
        for j in range(X_input.shape[1]):
            membership *= gaussian_membership(X_input[:, j], means_mat[i, j], sigmas_vec[j])
        memberships[:, i] = membership
    return memberships

def fuzzy_svm_predict(cluster_models, X_input, memberships):
    predictions = []
    for i in range(X_input.shape[0]):
        cluster_predictions = np.array([cluster_models[c].predict(X_input[i:i+1])[0] for c in range(n_clusters)])
        weight_sum = np.sum(memberships[i])
        if weight_sum == 0:
            weighted_pred = np.mean(cluster_predictions)
        else:
            weighted_pred = np.dot(cluster_predictions, memberships[i]) / weight_sum
        predictions.append(weighted_pred)
    return np.array(predictions)

fuzzy_memberships_val = compute_memberships(X_val, means, sigmas)
y_pred = fuzzy_svm_predict(models, X_val, fuzzy_memberships_val)

# Tính toán các chỉ số đánh giá
mse = mean_squared_error(y_val, y_pred)
var_y = np.var(y_val)
nmse = mse / var_y if var_y != 0 else 0
mae = mean_absolute_error(y_val, y_pred)
denom = np.abs(y_val + y_pred)
ds = np.mean(np.where(denom != 0, np.abs((y_val - y_pred) / denom), 0))

# Hiển thị kết quả
st.subheader("Model Evaluation")
col1, col2, col3, col4 = st.columns(4)
col1.metric("MSE", f"{mse:.4f}")
col2.metric("NMSE", f"{nmse:.4f}")
col3.metric("MAE", f"{mae:.4f}")
col4.metric("DS", f"{ds:.4f}")

# Biểu đồ so sánh
st.write("### Predictions vs Actual Values")
results_df = pd.DataFrame({"Actual": y_val, "Predicted": y_pred})
st.line_chart(results_df)
