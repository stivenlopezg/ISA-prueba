import shap
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from IPython.display import display


def barplot(df: pd.DataFrame, col: str):
    """
    Take a dataframe and graph the barchart of the variable of interest
    :param df: dataframe
    :param col: variable of interest
    :return:
    """
    aux = df[col].value_counts().sort_values(ascending=True)
    bars = tuple(aux.index.tolist())
    values = aux.values.tolist()
    y_pos = np.arange(len(bars))
    colors = ['lightblue'] * len(bars)
    colors[-1] = 'blue'
    plt.figure(figsize=(16, 10))
    plt.barh(y_pos, values, color=colors)
    plt.title(f'{col} bar chart')
    plt.yticks(y_pos, bars)
    return plt.show()


def univariate_graph(df: pd.DataFrame, column: str):
    """
    Take a dataframe and graph the histogram and boxplot of the variable of interest
    :param df: dataframe
    :param column: variable of interest
    :return:
    """
    plt.figure(figsize=(14, 10))
    plt.subplot(1, 2, 1)
    sns.distplot(df[column])
    plt.subplot(1, 2, 2)
    sns.boxplot(data=df, y=column)
    return plt.show()


def boxplot(df: pd.DataFrame, column: str, label: str = None):
    """
    take a dataframe and graph the boxplot of the variable of interest
    :param df: dataframe
    :param column: variable of interest
    :param label: target variable of predict
    :return:
    """
    plt.figure(figsize=(14, 10))
    if label is None:
        return sns.boxplot(data=df, y=column, linewidth=2.5, palette='Blues')
    else:
        return sns.boxplot(data=df, x=label, y=column, linewidth=2.5, palette='Blues')


def plot_feature_importance(model, data: pd.DataFrame):
    """
    Graph the importance of the variables
    :param model: model
    :param data: DataFrame
    :return:
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data)
    return shap.summary_plot(shap_values, data, plot_type='bar', plot_size=(14, 10))


def radar_chart(df: pd.DataFrame, columns: list, label: str, title: str):
    """
    Radar chart filtering by the categories of the target variable
    :param title: title
    :param df: DataFrame
    :param columns: columns of interest
    :param label: target variable
    :return:
    """
    categories = sorted(df[label].unique().tolist())
    table = np.round(df.groupby(by=label, as_index=True)[columns].mean(), decimals=2)
    figure = go.Figure()
    for categorie in categories:
        figure.add_trace(go.Scatterpolar(r=table.loc[categorie].tolist(),
                                         theta=columns,
                                         fill='toself',
                                         name=categorie))
    figure.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                         showlegend=True, title=title)
    return figure.show()


def plot_locally_interpretation(model, data: pd.DataFrame, indice: int):
    """
    Local interpretation graph for a given observation
    :param indice: observation index
    :param model: model
    :param data: DataFrame
    :return:
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data)
    shap.initjs()
    for i in range(0, 3):
        display(shap.force_plot(explainer.expected_value[i], shap_values[i][indice, :], data.iloc[indice, :]))
    return None


def locally_interpretation(model, data: pd.DataFrame, indice: int):
    """
    Local interpretation DataDrame for a given observation
    :param indice: observation index
    :param model: model
    :param data: DataFrame
    :return: pd.DataFrame
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data)
    first = pd.DataFrame(shap_values[0][indice, :], columns=data.columns, index=['Ok'])
    second = pd.DataFrame(shap_values[1][indice, :], columns=data.columns, index=['Alerta'])
    third = pd.DataFrame(shap_values[2][indice, :], columns=data.columns, index=['Error'])
    df = pd.concat([first, second, third])
    return display(df.transpose())
