import numpy as np
import pandas as pd
import statsmodels.api as sm


def VAR(data, maxlags):
    '''Vector Auto-Regressive (VAR) model fit.
    
    Fits multivariate data and returns:
    results  -- VAR fit
    forecast -- VAR forecast for a specified number of steps
    '''

    model = sm.tsa.VAR(data)
    results = model.fit(maxlags)

    lag_order = results.k_ar
    forecast = results.forecast(data.values[-lag_order:], forecast_steps)

    return results, forecast


if __name__ == '__main__':

    # Load example data - fspeptide train embeddings
    filename = '../unsupervised/embeddings/fspeptide/train_embeddings.npy'
    data = pd.DataFrame(np.load(filename))

    # Set VAR parameters
    maxlags = 10
    forecast_steps = 5000

    # Run VAR and plot data with forecast
    results, forecast = VAR(data, maxlags)    
    results.plot_forecast(forecast_steps)

    # See a summary of regression results
    # results.summary()
