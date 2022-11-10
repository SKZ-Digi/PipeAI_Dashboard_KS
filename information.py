import streamlit as st
from PIL import Image
import logic



def app():
    st.title('Information about used Algorithm')

    st.subheader('General Information about Model')
    st.markdown('An artificial neural network (ANN) is an information processing paradigm that is inspired by the way the biological nervous system such as a human brain processes information. The algorithm is composed of a large number of highly interconnected processing elements (neurons) working in unison to solve a specific problem. The input layer receives new cases, which are calculated within the hidden layers. Here, an “activation function” decides whether a neuron should be activated or not by calculating the weighted sum and further adding bias to it. This finally activates an output on the output layer.')


    st.subheader('Model Architecture')
    image = Image.open('EDA/model_classification.png')
    st.image(image, caption='Deep Learning Model Description')

    st.subheader('Model Performance Metrics')
    #st.text('Confusion Matrix:')
    #X_train, y_train, X_val, y_val = logic.load_and_split_data()
    #model = logic.load_model(logic.return_model_path())
    #acc,f1,ps,rs,cm = logic.rate_model(X_val,y_val, model)
    #st.pyplot()
    #st.text('Accuracy of model is ' + str(round(acc, 2))+"%")  # - Information about algorithm
    #st.text('F1-score of model is ' + str(round(f1, 2)))
    #st.text('Precision score of model is ' + str(round(ps, 2)))
    #st.text('Recall score of model is ' + str(round(rs, 2)))
    st.text('Confusion Matrix:')
    image2 = Image.open('EDA/conf_matrix_new.png')
    st.image(image2)
    st.text('Accuracy of model is 92.19%')  # - Information about algorithm
    st.text('F1-score of model is 0.93')
    st.text('Precision score of model is 0.94')
    st.text('Recall score of model is 0.91')

    st.sidebar.image(logic.add_logo(logo_path="images/logo_pipeai.png", width=291, height=100)) #hier
    hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
    st.markdown(hide_menu_style, unsafe_allow_html=True)


