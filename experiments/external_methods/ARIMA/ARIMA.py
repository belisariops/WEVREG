from statsmodels.tsa.statespace.sarimax import SARIMAX


class ARIMA:
    def __init__(self, order=(0, 1, 0), seasonal_order=(1, 0, 1, 12)):
        self.model = None
        self.model_fit = None
        self.order = order
        self.seasonal_order = seasonal_order

    def fit(self, x, y):
        self.model = SARIMAX(y, exog=x, order=self.order, seasonal_order=self.seasonal_order)
        self.model_fit = self.model.fit()

    def predict(self, x):
        return self.model_fit.forecast(len(x), exog=x)

    def predict_uncertainty(self, x):
        forecast = self.model_fit.get_forecast(len(x), exog=x)
        return forecast.predicted_mean, forecast.conf_int()
