
import streamlit as st
from PIL import Image
import logic
import seaborn as sns


def boxplot(X_train_new):
    import matplotlib.pyplot as plt
    cmap = sns.color_palette("Set3")
    st.write('#### Boxplot')
    st.write('Representation of location and distribution measures per feature as well as the corresponding quantiles.')
    fig, ax = plt.subplots(figsize=(10,7))
    plt.xticks(rotation=45)
    st.write(sns.boxplot(data=X_train_new, palette=cmap))
    st.pyplot()

def heatmap(X_train):
    Var_Corr = X_train.corr()
    import matplotlib.pyplot as plt
    st.write('#### Heatmap')
    st.write('Representation of correlation and cluster between different features within the dataset.')
    fig, ax = plt.subplots(figsize=(10,10))
    st.write(sns.clustermap(Var_Corr, xticklabels=Var_Corr.columns, yticklabels=Var_Corr.columns, annot=True))
    st.pyplot()


def app():
    st.title('History')

    #string = lorem.words(100)
    #st.markdown(string)

    st.subheader('Historical Data')

    X_train, y_train, X_val, y_val = logic.load_and_split_data()
    X_train['Weld_factor'] = y_train
    st.dataframe(X_train)

    st.subheader('Feature Trend')

    image = Image.open('EDA/df_PP_Training.png')
    st.write('The following figure is showing the temporary progression of test series per feature.')
    st.image(image, caption='Feature Trend Analysis')


    st.subheader('Distribution of Historical Data')

    boxplot(X_train)
    heatmap(X_train)

    st.sidebar.image(logic.add_logo(logo_path="images/logo_pipeai.png", width=291, height=100)) #hier
    hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
    st.markdown(hide_menu_style, unsafe_allow_html=True)
    
    
    
    
    