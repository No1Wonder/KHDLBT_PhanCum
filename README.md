Video:
https://youtu.be/GZzPS3zGJMA?si=kRqRZQ2_XUSKKnz1

Code Colab
https://colab.research.google.com/drive/1sHZIy9G5MHu5voq0OybyRZG5nJNc4OYV?usp=sharing
# ==============================================================================
# BƯỚC 1: ĐỌC DỮ LIỆU VÀ ĐỊNH DANH HỆ THỐNG
# ==============================================================================
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Tự động tìm file điểm trên Colab
possible_names = ["TỔNG HỢP ĐIỂM K58KTP.csv", "TỔNG HỢP ĐIỂM K58KTP - Trang tính1.csv"]
file_name = None

for name in possible_names:
    if os.path.exists(name) or os.path.exists("/content/" + name):
        file_name = name if os.path.exists(name) else "/content/" + name
        break

if file_name is None:
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    file_name = csv_files[0] if csv_files else "TỔNG HỢP ĐIỂM K58KTP.csv"

df_raw = pd.read_csv(file_name, header=None)

# ==============================================================================
# BƯỚC 2: TIỀN XỬ LÝ VÀ LÀM SẠCH DỮ LIỆU (DATA CLEANING)
# ==============================================================================
mssv_list = df_raw.iloc[1, 3:].values
name_list = df_raw.iloc[2, 3:].values
student_ids = [f"{m} - {n}" for m, n in zip(mssv_list, name_list)]

course_codes = df_raw.iloc[4:, 1].values
course_names = df_raw.iloc[4:, 2].values
scores_matrix = df_raw.iloc[4:, 3:].copy()

scores_matrix = scores_matrix.astype(str).apply(lambda x: x.str.replace(',', '.', regex=False))
scores_matrix = scores_matrix.apply(pd.to_numeric, errors='coerce')

is_valid_student = scores_matrix.notna().any(axis=0)
excluded_students = [student_ids[i] for i, valid in enumerate(is_valid_student) if not valid]

scores_valid = scores_matrix.loc[:, is_valid_student].fillna(0.0)
students_valid_names = [student_ids[i] for i, valid in enumerate(is_valid_student) if valid]

# ==============================================================================
# BƯỚC 3: TRÍCH XUẤT KHÔNG GIAN ĐẶC TRƯNG (ĐÃ GOM KỸ NĂNG VÀO CƠ SỞ)
# ==============================================================================
co_so_keywords = ['Toán', 'Giải tích', 'Đại số', 'Lý', 'Vật lý', 'Xác suất', 'Rời rạc', 'Logic']
ky_nang_keywords = ['Tiếng Anh', 'ENG', 'Thể chất', 'Bóng', 'Giao tiếp', 'Môi trường', 'Pháp luật', 'Tư tưởng', 'Lịch sử', 'Chủ nghĩa']

co_so_indices = []
chuyen_nganh_indices = []

for i, (code, name) in enumerate(zip(course_codes, course_names)):
    name_str = str(name)
    code_str = str(code)

    # LOGIC MỚI: Nếu trúng từ khóa Cơ sở HOẶC từ khóa Kỹ năng/Đại cương -> Xếp chung vào nhóm Cơ sở
    if any(kw in name_str for kw in co_so_keywords) or any(kw in name_str for kw in ky_nang_keywords) or any(kw in code_str for kw in ['ENG', 'PED', 'BAS']):
        co_so_indices.append(i)
    else:
        # Các môn lập trình, dự án phần mềm chuyên sâu còn lại giữ nguyên là Chuyên ngành
        chuyen_nganh_indices.append(i)

# Tính điểm trung bình theo cấu trúc nhóm mới
mean_co_so = scores_valid.iloc[co_so_indices, :].mean(axis=0).values
mean_chuyen_nganh = scores_valid.iloc[chuyen_nganh_indices, :].mean(axis=0).values
X = np.column_stack((mean_co_so, mean_chuyen_nganh))

# ==============================================================================
# BƯỚC 4: HUẤN LUYỆN MÔ HÌNH K-MEANS & ĐỊNH DANH PHÂN CẤP TÂM CỤM
# ==============================================================================
kmeans = KMeans(n_clusters=3, random_state=42, n_init=15)
labels = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_

cs_centers = centroids[:, 0]
cn_centers = centroids[:, 1]

idx_lech_cs = np.argmax(cs_centers)
remaining_indices = [i for i in range(3) if i != idx_lech_cs]
idx_lech_cn = remaining_indices[np.argmax(cn_centers[remaining_indices])]
idx_toan_dien = [i for i in range(3) if i not in [idx_lech_cs, idx_lech_cn]][0]

cluster_mapping = {
    idx_toan_dien: "CỤM 1: TOÀN DIỆN / HỌC ĐỀU",
    idx_lech_cs: "CỤM 2: LỆCH LÝ THUYẾT ĐẠI CƯƠNG (Mạnh Cơ sở & Kỹ năng)",
    idx_lech_cn: "CỤM 3: LỆCH THỰC HÀNH (Mạnh Chuyên ngành)"
}

text_labels = [cluster_mapping[num] for num in labels]

groups = {"CỤM 1: TOÀN DIỆN / HỌC ĐỀU": [], "CỤM 2: LỆCH LÝ THUYẾT ĐẠI CƯƠNG (Mạnh Cơ sở & Kỹ năng)": [], "CỤM 3: LỆCH THỰC HÀNH (Mạnh Chuyên ngành)": []}
for name, label_idx in zip(students_valid_names, labels):
    groups[cluster_mapping[label_idx]].append(name)

# ==============================================================================
# BƯỚC 5: TRỰC QUAN HÓA BẢNG FLAT ĐẦU RA
# ==============================================================================
list_1 = groups["CỤM 1: TOÀN DIỆN / HỌC ĐỀU"]
list_2 = groups["CỤM 2: LỆCH LÝ THUYẾT ĐẠI CƯƠNG (Mạnh Cơ sở & Kỹ năng)"]
list_3 = groups["CỤM 3: LỆCH THỰC HÀNH (Mạnh Chuyên ngành)"]

max_len = max(len(list_1), len(list_2), len(list_3), len(excluded_students))
pad_list = lambda lst, target: lst + [""] * (target - len(lst))

col_name_1 = f"CỤM 1: TOÀN DIỆN ({len(list_1)} SV)"
col_name_2 = f"CỤM 2: LỆCH ĐẠI CƯƠNG KỸ NĂNG ({len(list_2)} SV)"
col_name_3 = f"CỤM 3: LỆCH CHUYÊN NGÀNH ({len(list_3)} SV)"
col_name_4 = f"BỊ LOẠI ({len(excluded_students)} SV)"

df_output = pd.DataFrame({
    col_name_1: pad_list(list_1, max_len),
    col_name_2: pad_list(list_2, max_len),
    col_name_3: pad_list(list_3, max_len),
    col_name_4: pad_list(excluded_students, max_len)
})

print("\n" + "="*95)
print("BẢNG PHÂN CỤM SINH VIÊN K58KTP BẰNG K-MEANS (ĐÃ GOM MÔN KỸ NĂNG VÀO CƠ SỞ)")
print("="*95)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1500)
from IPython.display import display
display(df_output)

# ==============================================================================
# BƯỚC 6: VẼ SƠ ĐỒ PHÂN CỤM ĐỂ ĐỐI CHIẾU (SCATTER PLOT)
# ==============================================================================
print("\n" + "="*95)
print("ĐANG KHỞI TẠO ĐỒ THỊ KHÔNG GIAN PHÂN CỤM CẬP NHẬT...")
print("="*95)

plt.figure(figsize=(10, 7))

color_palette = {
    "CỤM 1: TOÀN DIỆN / HỌC ĐỀU": "#2ce3b9",
    "CỤM 2: LỆCH LÝ THUYẾT ĐẠI CƯƠNG (Mạnh Cơ sở & Kỹ năng)": "#ffb703",
    "CỤM 3: LỆCH THỰC HÀNH (Mạnh Chuyên ngành)": "#fb8500"
}

sns.scatterplot(
    x=mean_co_so,
    y=mean_chuyen_nganh,
    hue=text_labels,
    palette=color_palette,
    s=100,
    alpha=0.8,
    edgecolor='w'
)

plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    s=350,
    marker='*',
    c='red',
    edgecolor='black',
    linewidth=1.5,
    label='Tâm cụm (Centroid)'
)

lims = [0, 4]
plt.plot(lims, lims, alpha=0.5, color='gray', linestyle='--', label='Đường cân bằng điểm')

plt.title('SƠ ĐỒ PHÂN CỤM K-MEANS CẬP NHẬT (KỸ NĂNG THUỘC TRỤC CƠ SỞ)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Điểm TB nhóm môn CƠ SỞ & KỸ NĂNG MỀM', fontsize=12)
plt.ylabel('Điểm TB nhóm môn CHUYÊN NGÀNH', fontsize=12)

plt.xlim(0, 4.2)
plt.ylim(0, 4.2)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.show()
