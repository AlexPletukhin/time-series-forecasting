from statsmodels.tsa.arima.model import ARIMA
import numpy as np

class ARIMAModel:

    def __init__(self, order=(5,1,0)):
        self.order = order
        self.model = None

    def fit(self, series):
        self.model = ARIMA(series, order=self.order).fit()

    def predict(self, steps):
        return self.model.forecast(steps=steps)