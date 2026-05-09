# Abstract Type Hierarchy for DSToolkit Models

"""
    AbstractToolkitModel

Root abstract type for all DSToolkit models.
"""
abstract type AbstractToolkitModel end

"""
    TabularModel <: AbstractToolkitModel

Abstract type for tabular data models (regression and classification).
"""
abstract type TabularModel <: AbstractToolkitModel end

"""
    RegressionModel <: TabularModel

Abstract type for regression models.
"""
abstract type RegressionModel <: TabularModel end

"""
    ClassificationModel <: TabularModel

Abstract type for classification models.
"""
abstract type ClassificationModel <: TabularModel end

"""
    TimeSeriesModel <: AbstractToolkitModel

Abstract type for time series models (statistical and deep learning).
"""
abstract type TimeSeriesModel <: AbstractToolkitModel end

"""
    StatTimeSeriesModel <: TimeSeriesModel

Abstract type for statistical time series models (ARIMA, ETS).
"""
abstract type StatTimeSeriesModel <: TimeSeriesModel end

"""
    DeepTimeSeriesModel <: TimeSeriesModel

Abstract type for deep learning time series models (RNN, LSTM, GRU).
"""
abstract type DeepTimeSeriesModel <: TimeSeriesModel end
