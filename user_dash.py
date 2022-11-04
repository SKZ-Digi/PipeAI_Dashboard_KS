import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import tensorflow as tf
from lime import lime_tabular
import streamlit.components.v1 as components
import logic
from lorem_text import lorem
import shap
import streamlit as st
shap.initjs()

st.set_option('deprecation.showPyplotGlobalUse', False)


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def inital_run():
    print('###########################################################################################################################################################################################################################################################################################################################')

    i = np.random.randint(2, 67)
    X_train, y_train, X_val, y_val, y_train_int, y_test_int = logic.load_data_stored()
    model = logic.load_model(logic.return_model_path())
    proba,pred = logic.make_predicion(X_val, y_val, model, i)
    explainer = shap.KernelExplainer(model.predict_proba, X_train)
    rule_exp = logic.load_anchors(i, X_train, X_val, y_train_int)
    lime_exp = logic.lime_explainer(X_train,i,model,X_val)

    return i, X_train,y_train,X_val,y_val,model,proba, pred,explainer,lime_exp, rule_exp






def run_app():

    i, X_train,y_train,X_val,y_val,model,proba, pred,explainer,lime_exp, rule_exp = inital_run()



    #i = np.random.randint(0, X_val.shape[0])
    st.header("Input Information:")

    st.sidebar.image(logic.add_logo(logo_path="images/logo_pipeai.png", width=291, height=100)) #hier
    hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
    st.markdown(hide_menu_style, unsafe_allow_html=True)

    st.dataframe(pd.DataFrame(X_val.iloc[i]), width= 1000)



    st.text('Prediction: '+pred)
    st.button('Provide Decision Alternatives')
    st.header("Explanation for Prediction:")

    option = st.selectbox( 'Select Explanation', ('SHAP Local Explanation','Rulset','Lime Plot'))

    if option == 'SHAP Local Explanation':
        st.markdown('The following plot shows the impact of different features for a single prediction. '
                    'Thereby, the base value represents the mean prediction of historical data. '
                    'That is, features colored red represent an impact that the XAI model assumes will lead to an iO prediction. '
                    'In contrast, blue colored features represent an impact towards an niO prediction. Lastly, f(x) represents the XAI model prediction score.')
        plot = logic.show_force_plot(explainer, X_val,i)
        logic.st_shap(plot, None)


    if option == 'Rulset':
        logic.anchor_show_notebook(rule_exp)
        st.markdown('The following table illustrate different ruleset that the models used to make a prediction for a specific observation.')


    if option == 'Lime Plot':
        logic.lime_show_notebook(lime_exp)
        st.markdown("The output of LIME is a list of explanations, reflecting the contribution of each feature to the prediction of a data sample. This provides local interpretability, and it also allows to determine which feature changes will have most impact on the prediction."
                    'An explanation is created by approximating the underlying model locally by an interpretable one. Interpretable models are e.g. linear models with strong regularisation, decision tree’s, etc. The interpretable models are trained on small perturbations of the original instance and should only provide a good local approximation. Source Hulstaert (2018)')

    st.header("Hypothetical Scenario:")

    st.markdown('The welding process is the last step in a long chain of pre-processing steps. This means that if the processed modules are defective and are not treated accordingly, enormous damage can occur. The model takes the present confidence the classification:')
    st.text('Prediction: '+pred)
    st.text('Confidence of prediction: '+str(proba)+"%")

    image = Image.open('EDA/hyposcen.png')
    st.image(image, caption='(a) Welding process of two tubes. (b) Poorly welded tube. Adopted from Wikipedia (2022) and Lee et al. (2012)')

    st.markdown('If, contrary to expectations, the component is damaged, this can result in serious consequences. In the plastic welding process, two tubes are welded together, as shown in the following example (a). If poorly welded pipes are used, this may result in the loss of the goods being transported. This is, the soil may be contaminated by toxic substances. The consequences can be serious and also generate high costs.')




def app():
    st.title('Explainable Intelligent System')
    run_app()