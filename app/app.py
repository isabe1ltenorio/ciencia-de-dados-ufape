import os
import glob
from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import io
import base64
import sklearn
import datetime

app = Flask(__name__)

def get_all_variables():
    csv_files = glob.glob(os.path.join(app.root_path, '..', '*.csv'))
    all_vars = set()
    for file in csv_files:
        df = pd.read_csv(file)
        all_vars.update(df.columns)
    return sorted(list(all_vars))

@app.route('/')
def index():
    variables = get_all_variables()
    return render_template('index.html', variables=variables, results=None)

@app.route('/regressao', methods=['POST'])
def regressao():
    variables = get_all_variables()
    x_vars = request.form.getlist('x_vars')
    y_var = request.form.get('y_var')

    if not x_vars or not y_var:
        return render_template('index.html', error='Selecione pelo menos uma variável independente (X) e uma dependente (y).', variables=variables)

    if len(y_var.split()) > 1:
        return render_template('index.html', error='Selecione apenas uma variável dependente (y).', variables=variables)

    csv_files = glob.glob(os.path.join(app.root_path, '..', '*.csv'))
    df = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)

    required_vars = x_vars + [y_var]
    df_filtered = df.dropna(subset=required_vars)

    for var in required_vars:
        df_filtered[var] = pd.to_numeric(df_filtered[var], errors='coerce')

    df_filtered = df_filtered.dropna(subset=required_vars)


    X = df_filtered[x_vars]
    y = df_filtered[y_var]

    if X.empty or y.empty:
        return render_template('index.html', error='Não há dados numéricos suficientes para as variáveis selecionadas.', variables=variables)


    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)

    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    plt.figure(figsize=(10, 5))

    # Scatter plot
    plt.subplot(1, 2, 1)
    # Since we can have multiple X, we plot the first one for simplicity
    plt.scatter(X.iloc[:, 0], y, color='blue', alpha=0.5)
    # To plot the regression line, we need to handle multiple dimensions.
    # A common approach is to plot predicted vs actual values.
    plt.scatter(y_pred, y, color='red', alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    plt.title('Regressão Linear: Previsto vs. Real')
    plt.xlabel('Valores Previstos')
    plt.ylabel('Valores Reais')


    # Residuals plot
    plt.subplot(1, 2, 2)
    plt.scatter(y_pred, y - y_pred, color='green')
    plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), color='red', linewidth=2)
    plt.title('Gráfico de Resíduos')
    plt.xlabel('Valores Previstos')
    plt.ylabel('Resíduos')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    results = {
        'mse': mse,
        'r2': r2,
        'plot_url': plot_url,
        'sklearn_version': sklearn.__version__,
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    return render_template('index.html', results=results, variables=variables)


if __name__ == '__main__':
    app.run(debug=True)
