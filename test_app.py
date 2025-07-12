import unittest
import os
import pandas as pd
from app.app import app

class AppTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

        # Create dummy csv files for testing
        self.test_data1 = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [5, 4, 3, 2, 1],
            'C': ['text1', 'text2', 'text3', 'text4', 'text5']
        })
        self.test_csv1_path = 'test1.csv'
        self.test_data1.to_csv(self.test_csv1_path, index=False)

        self.test_data2 = pd.DataFrame({
            'D': [10, 20, 30, 40, 50],
            'E': [50, 40, 30, 20, 10]
        })
        self.test_csv2_path = 'test2.csv'
        self.test_data2.to_csv(self.test_csv2_path, index=False)


    def tearDown(self):
        os.remove(self.test_csv1_path)
        os.remove(self.test_csv2_path)

    def test_index_page(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn('<h1>Aplicação de Regressão Linear</h1>'.encode('utf-8'), response.data)
        self.assertIn(b'A', response.data)
        self.assertIn(b'B', response.data)
        self.assertIn(b'C', response.data)
        self.assertIn(b'D', response.data)
        self.assertIn(b'E', response.data)

    def test_regression_success(self):
        response = self.app.post('/regressao', data={
            'x_vars': ['A'],
            'y_var': 'B'
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn('<h2>Resultados da Regressão</h2>'.encode('utf-8'), response.data)
        self.assertIn('<strong>R²:</strong>'.encode('utf-8'), response.data)
        self.assertIn(b'<img src="data:image/png;base64,', response.data)

    def test_regression_no_vars(self):
        response = self.app.post('/regressao', data={})
        self.assertEqual(response.status_code, 200)
        self.assertIn('Selecione pelo menos uma variável independente (X) e uma dependente (y).'.encode('utf-8'), response.data)

    def test_regression_non_numeric_data(self):
        response = self.app.post('/regressao', data={
            'x_vars': ['A'],
            'y_var': 'C'
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn('Não há dados numéricos suficientes para as variáveis selecionadas.'.encode('utf-8'), response.data)

if __name__ == '__main__':
    unittest.main()
