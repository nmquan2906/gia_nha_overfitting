import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

df = pd.read_csv('gia_nha.csv')
X = df[['Dien_Tich_m2', 'So_Phong_Ngu', 'Khoang_Cach_Trung_Tam_km']]
y = df['Gia_Ty_VND']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("DEMO HIỆN TƯỢNG OVERFITTING")

mo_hinh_overfit = DecisionTreeRegressor(max_depth=None, random_state=42)
mo_hinh_overfit.fit(X_train, y_train)

sai_so_train_1 = mean_absolute_error(y_train, mo_hinh_overfit.predict(X_train))
sai_so_test_1 = mean_absolute_error(y_test, mo_hinh_overfit.predict(X_test))

print("1. KHI ĐỘ PHỨC TẠP CỦA MÔ HÌNH QUÁ CAO:")
print(f" -> Lệch giá trên tập Training : {sai_so_train_1:.2f} Tỷ VNĐ")
print(f" -> Lệch giá trên tập Test     : {sai_so_test_1:.2f} Tỷ VNĐ")
print("Mô hình quá khớp với dữ liệu training, nhưng thất bại trên dữ liệu bên ngoài.\n")

mo_hinh_tot = DecisionTreeRegressor(max_depth=3, random_state=42)
mo_hinh_tot.fit(X_train, y_train)

sai_so_train_2 = mean_absolute_error(y_train, mo_hinh_tot.predict(X_train))
sai_so_test_2 = mean_absolute_error(y_test, mo_hinh_tot.predict(X_test))

print("2. ÁP DỤNG REGULARIZATION (Giảm độ phức tạp):")
print(f" -> Lệch giá trên tập Training : {sai_so_train_2:.2f} Tỷ VNĐ)")
print(f" -> Lệch giá trên tập Test     : {sai_so_test_2:.2f} Tỷ VNĐ")
print("Chấp nhận hy sinh độ chính xác tập Training để giữ được tính tổng quát.")