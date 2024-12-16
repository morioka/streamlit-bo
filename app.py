def SHAP_explain(BO_model, autoscaled_x, x):
    with st.spinner():
        explainer = shap.KernelExplainer(BO_model.predict, autoscaled_x)
        shap_values = explainer.shap_values(autoscaled_x)

        plt.figure()
        shap.summary_plot(shap_values, autoscaled_x, feature_names=x.columns)
        plt.savefig('shap_summary_plot.png')
    
    return 'shap_summary_plot.png'

 
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
                # --------------------省略----------------------------------
                pass

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
