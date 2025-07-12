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

@app.route('/')
def index():
    csv_files = glob.glob(os.path.join(app.root_path, '..', '*.csv'))
    csv_files = [os.path.basename(file) for file in csv_files]
    return render_template('index.html', csv_files=csv_files, results=None)

@app.route('/regressao', methods=['POST'])
def regressao():
    csv_file = request.form.get('csv_file')
    if not csv_file:
        return render_template('index.html', error='Nenhum arquivo CSV selecionado.')

    df = pd.read_csv(os.path.join(app.root_path, '..', csv_file))
    variables = df.columns.tolist()

    if request.form.get('submit_vars'):
        x_vars = request.form.getlist('x_vars')
        y_var = request.form.get('y_var')

        if not x_vars or not y_var:
            return render_template('index.html', error='Selecione pelo menos uma variável independente (X) e uma dependente (y).', csv_file=csv_file, variables=variables)

        if len(y_var.split()) > 1:
            return render_template('index.html', error='Selecione apenas uma variável dependente (y).', csv_file=csv_file, variables=variables)

        X = df[x_vars]
        y = df[y_var]

        model = LinearRegression()
        model.fit(X, y)

        y_pred = model.predict(X)

        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        plt.figure(figsize=(10, 5))

        # Scatter plot
        plt.subplot(1, 2, 1)
        plt.scatter(X, y, color='blue')
        plt.plot(X, y_pred, color='red', linewidth=2)
        plt.title('Regressão Linear')
        plt.xlabel(x_vars[0])
        plt.ylabel(y_var)

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

        return render_template('index.html', results=results, csv_file=csv_file, variables=variables)

    return render_template('index.html', csv_file=csv_file, variables=variables)


if __name__ == '__main__':
    app.run(debug=True)
