import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

import shap
import plotly.express as px
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, DotProduct, WhiteKernel, RBF, ConstantKernel

import random



def SHAP_explain(BO_model, autoscaled_x, x):
    with st.spinner():
        explainer = shap.KernelExplainer(BO_model.predict, autoscaled_x)
        shap_values = explainer.shap_values(autoscaled_x)

        plt.figure()
        shap.summary_plot(shap_values, autoscaled_x, feature_names=x.columns)
        plt.savefig('shap_summary_plot.png')
    
    return 'shap_summary_plot.png'

def plot_scatter(data, x_col, y_col, z_col, color_col):
    # Streamlitを使って3D散布図を簡単に表示する
    # https://zenn.dev/nishijima13/articles/4891b9a05b2854
    fig = px.scatter_3d(
        data,
        x=x_col,
        y=y_col,
        z=z_col,
        color=color_col,
    )
    # グラフレイアウト設定
    #fig.update_layout(title="3D Point Cloud", width=500, height=500)
    fig.update_layout(width=500, height=500)
    # マーカーサイズ設定
    fig.update_traces(marker_size=2)
    # グラフ表示
    st.plotly_chart(fig, use_container_width=True)

def sample_generation(limit_data):  # ->generation_sample
    n_samples = 30

    limits = limit_data.values
    x = [[random.uniform(lower, upper) for (upper, lower) in limits] for _ in range(n_samples)]
    return np.array(x)

def BO(data, generation_sample):    # -> next_samples, BO_model, autoscaled_x
    # https://github.com/hkaneko1985/design_of_experiments/blob/master/Python/bayesianoptimization.py
    # def bayesianoptimization(X, y, candidates_of_X, acquisition_function_flag, cumulative_variance=None):
    """
    Bayesian optimization
    
    Gaussian process regression model is constructed between X and y.
    A candidate of X with the highest acquisition function is selected using the model from candidates of X.

    Parameters
    ----------
    X: numpy.array or pandas.DataFrame
        m x n matrix of X-variables of training dataset (m is the number of samples and n is the number of X-variables)
    y: numpy.array or pandas.DataFrame
        m x 1 vector of a y-variable of training dataset
    candidates_of_X: numpy.array or pandas.DataFrame
        Candidates of X
    acquisition_function_flag: int
        1: Mutual information (MI), 2: Expected improvement(EI), 
        3: Probability of improvement (PI) [0: Estimated y-values]
    cumulative_variance: numpy.array or pandas.DataFrame
        cumulative variance in mutual information (MI)[acquisition_function_flag=1]

    Returns
    -------
    selected_candidate_number : int
        selected number of candidates_of_X
    selected_X_candidate : numpy.array
        selected X candidate
    cumulative_variance: numpy.array
        cumulative variance in mutual information (MI)[acquisition_function_flag=1]
    """
    X = data.iloc[:, 1:]
    y = data.iloc[:, 0]
    candidates_of_X = generation_sample
    acquisition_function_flag = 2
    cumulative_variance = None

    X = np.array(X)
    y = np.array(y)
    if cumulative_variance is None:
        cumulative_variance = np.empty(len(y))
    else:
        cumulative_variance = np.array(cumulative_variance)

    relaxation_value = 0.01
    delta = 10 ** -6
    alpha = np.log(2 / delta)

    autoscaled_X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    autoscaled_candidates_of_X = (candidates_of_X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    autoscaled_y = (y - y.mean(axis=0)) / y.std(axis=0, ddof=1)
    gaussian_process_model = GaussianProcessRegressor(ConstantKernel() * RBF() + WhiteKernel(), alpha=0)
    gaussian_process_model.fit(autoscaled_X, autoscaled_y)
    autoscaled_estimated_y_test, autoscaled_std_of_estimated_y_test = gaussian_process_model.predict(
        autoscaled_candidates_of_X, return_std=True)

    if acquisition_function_flag == 1:
        acquisition_function_values = autoscaled_estimated_y_test + alpha ** 0.5 * (
                (autoscaled_std_of_estimated_y_test ** 2 + cumulative_variance) ** 0.5 - cumulative_variance ** 0.5)
        cumulative_variance = cumulative_variance + autoscaled_std_of_estimated_y_test ** 2
    elif acquisition_function_flag == 2:
        acquisition_function_values = (autoscaled_estimated_y_test - max(autoscaled_y) - relaxation_value) * \
                                      norm.cdf((autoscaled_estimated_y_test - max(autoscaled_y) - relaxation_value) /
                                               autoscaled_std_of_estimated_y_test) + \
                                      autoscaled_std_of_estimated_y_test * \
                                      norm.pdf((autoscaled_estimated_y_test - max(autoscaled_y) - relaxation_value) /
                                               autoscaled_std_of_estimated_y_test)
    elif acquisition_function_flag == 3:
        acquisition_function_values = norm.cdf(
            (autoscaled_estimated_y_test - max(autoscaled_y) - relaxation_value) / autoscaled_std_of_estimated_y_test)
    elif acquisition_function_flag == 0:
        acquisition_function_values = autoscaled_estimated_y_test

    selected_candidate_number = np.where(acquisition_function_values == max(acquisition_function_values))[0][0]
    selected_X_candidate = candidates_of_X[selected_candidate_number, :]

    #return selected_candidate_number, selected_X_candidate, cumulative_variance
    return selected_X_candidate, gaussian_process_model, autoscaled_X
 
def tabs_set(data, limit_data):
    tab_titles = ["データの確認", "実験点の提案", "解析", "予測"]
    tabs = st.tabs(tab_titles)

    ss = st.session_state
    
    with tabs[0]:
        st.write('実験データの確認')
        st.write(data)
        st.write('limitデータの確認')
        st.write(limit_data)

        st.subheader('散布図')
        x_col = st.selectbox("X軸の列を選択", data.columns)
        y_col = st.selectbox("Y軸の列を選択", data.columns)
        z_col = st.selectbox("Z軸の列を選択", data.columns)
        color_col = st.selectbox("色の列を選択", data.columns)

        plot_scatter(data, x_col, y_col, z_col, color_col)

    with tabs[1]:
        if st.checkbox("実験点の提案"):
            if 'next_samples' not in ss:
                generation_sample = sample_generation(limit_data)
                BO_rsults = BO(data, generation_sample)
                next_samples = BO_rsults[0]
                BO_model = BO_rsults[1]
                autoscaled_x = BO_rsults[2]
                x = BO_rsults[3]
                
                # --------------------省略----------------------------------

                ss['next_samples'] = next_samples
                ss['BO_model'] = BO_model
                ss['autoscaled_x'] = autoscaled_x
                ss['x'] = x

                # --------------------省略----------------------------------
                y_estimated_y_plot = [[0, 1],[0,1]]
                y_estimated_y_r2 = 0
                y_estimated_y_mae = 0
                y_estimated_y_rmse = 0
                y_estimated_y_in_cv_plot = [[0, 1],[0,1]]
                y_estimated_y_in_cv_r2 = 0
                y_estimated_y_in_cv_mae = 0
                y_estimated_y__in_cv_rmse = 0

                col1, col2 = st.columns(2)
                
                with col1:
                        st.subheader('トレーニングデータの予測結果')
                        st.plotly_chart(y_estimated_y_plot)
                        st.write('r^2 for training data :', y_estimated_y_r2)
                        st.write('RMSE for training data :', y_estimated_y_mae)
                        st.write('MAE for training data :', y_estimated_y_rmse)
                
                with col2:
                        st.subheader('クロスバリデーションによる予測結果')
                        st.plotly_chart(y_estimated_y_in_cv_plot)
                        st.write('r^2 in cross-validation :', y_estimated_y_in_cv_r2)
                        st.write('RMSE in cross-validation :', y_estimated_y_in_cv_mae)
                        st.write('MAE in cross-validation :', y_estimated_y__in_cv_rmse)
                
                st.subheader('提案された実験点')
                st.write(next_samples)
            else:
                col1, col2 = st.columns(2)
                
                with col1:
                        st.subheader('トレーニングデータの予測結果')
                        st.plotly_chart(ss['y_estimated_y_plot'])

                with col2:
                        st.subheader('クロスバリデーションによる予測結果')
                        st.plotly_chart(ss['y_estimated_y_in_cv_plot'])

                st.subheader('提案された実験点')
                st.write(ss['next_samples'])
                
    with tabs[2]:
        if st.checkbox('SHAP解析', help='「実験点の提案」タブでモデルを構築してから'):
            if 'SHAP_plot' not in ss:
                pass
                # --------------------省略----------------------------------
                ss['SHAP_plot'] = SHAP_explain(ss['BO_model'],
                                               ss['autoscaled_x'],
                                               ss['x'])
                # で表示

def main():
    st.set_page_config(
        page_title='実験点提案@ベイズ最適化',
        layout='wide',
        page_icon = 'random'
        )
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title('実験点提案@ベイズ最適化')

    uploaded_file = st.file_uploader('データを読み込んで下さい。', type=['xlsx'], key = 'train_data')
    if uploaded_file:
        data = pd.read_excel(uploaded_file, sheet_name=0)
        limit_data = pd.read_excel(uploaded_file, sheet_name=1, index_col=0)
        
        tabs_set(data, limit_data)


if __name__ == "__main__":
    random.seed(42)  # シードを設定
    main()
