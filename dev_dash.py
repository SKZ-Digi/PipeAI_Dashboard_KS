import logic
import shap
import streamlit as st
shap.initjs()
st.set_option('deprecation.showPyplotGlobalUse', False)

def run_app():

    st.sidebar.image(logic.add_logo(logo_path="images/logo_pipeai.png", width=291, height=100)) #hier
    hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
    st.markdown(hide_menu_style, unsafe_allow_html=True)

    X_train, y_train, X_val, y_val= logic.load_and_split_data()
    model = logic.load_model(logic.return_model_path())

    st.header("Explanation of Used Model:")

    st.subheader("Visualizations:")

    option = st.selectbox( 'Select Explanation', ('SHAP Summary Plot','SHAP Feature Impact Plot'))
    # 'SHAP Dependence Plot',

    if option == 'SHAP Feature Impact Plot':
        #plt = shap.summary_plot(shap_values, X_val)

        explainer = logic.load_shap_explainer()
        shap_values = logic.load_shap_values("models/shap_values_global.sav")
        logic.st_shap(shap.force_plot(explainer.expected_value[0], shap_values[0], X_train), height= 400)
        st.markdown('The plot consists of many force plots, each of which explains the prediction of an instance. We rotate the force plots vertically and place them side by side according to their clustering similarity.')

    if option == 'SHAP Summary Plot':
        explainer = logic.load_shap_explainer()
        shap_values = logic.load_shap_values("models/shap_values_global.sav")
        plt = shap.summary_plot(shap_values[0], X_train)
        st.pyplot(plt)
        st.markdown('The summary plot combines feature importance with feature effects. Each point on the summary plot is a Shapley value for a feature and an instance. The position on the y-axis is determined by the feature and on the x-axis by the Shapley value. The color represents the value of the feature from low to high. Overlapping points are jittered in y-axis direction, so we get a sense of the distribution of the Shapley values per feature. The features are ordered according to their importance.')
        #if option == 'SHAP Dependence Plot':
    #
    #    plt = shap.dependence_plot(7,shap_values[0],X_val,feature_names=X_val.columns)
    #    #plt = shap.plots.heatmap(shap_values)
    #    st.pyplot(plt)

    st.write('Plot explanations used from christophm.github.io(2022)')

def app():
    st.title('Explainable Intelligent System')
    run_app()
