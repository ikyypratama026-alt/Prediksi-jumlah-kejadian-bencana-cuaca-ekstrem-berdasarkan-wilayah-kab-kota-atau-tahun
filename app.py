from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

app = Flask(__name__)

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv("data_bencana.csv")

print("ISI DATA:")
print(df.head())

print("\nNAMA KOLOM:")
print(df.columns)

# =========================
# PILIH DATA (SUDAH SESUAI DATASET KAMU)
# =========================
X = df[['tahun']]
y = df['jumlah_cuaca_ekstrem']

# =========================
# SPLIT DATA (UNTUK EVALUASI)
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# TRAIN MODEL
# =========================
model = LinearRegression()
model.fit(X_train, y_train)

# =========================
# EVALUASI MODEL
# =========================
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n=== HASIL EVALUASI ===")
print("MAE:", round(mae, 2))
print("MSE:", round(mse, 2))
print("R2:", round(r2, 2))

# =========================
# ROUTE WEB
# =========================


@app.route('/', methods=['GET', 'POST'])
def index():
    hasil = None

    if request.method == 'POST':
        tahun = int(request.form['tahun'])
        prediksi = model.predict([[tahun]])
        hasil = round(prediksi[0], 2)

    return render_template('index.html', hasil=hasil)


# =========================
# RUN APP
# =========================
if __name__ == '__main__':
    app.run(debug=True)
