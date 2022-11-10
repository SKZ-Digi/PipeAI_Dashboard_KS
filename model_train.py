import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from lorem_text import lorem
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def load_data():
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target
    # st.text('Data loaded')
    return X, Y, iris


def preprocess_data(X, Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    from sklearn.ensemble import RandomForestClassifier
    # st.text('Model starts training')
    model = RandomForestClassifier(n_estimators=10)
    model.fit(X_train, y_train)
    # st.text('Model trained')
    return model


def predict(dt, X_test, y_test):
    predict_test = dt.predict(X_test)
    acc = accuracy_score(y_test, predict_test)
    return acc, predict_test


def show_input_data(X_train, i):
    st.dataframe(X_train[i])


def show_lime_plot(model, X_test, X_train, iris, i):
    import lime
    import lime.lime_tabular

    explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=iris.feature_names,
                                                       class_names=iris.target_names, discretize_continuous=True)
    exp = explainer.explain_instance(X_test[i], model.predict_proba, num_features=5, top_labels=2)
    html = exp.as_html()
    return html


def show_shap_plot(model, X_test, X_train, iris, i):
    import shap
    import matplotlib.pyplot as plt
    #shap.initjs()
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    explainer = shap.KernelExplainer(model.predict_proba, X_train)
    shap_values = explainer.shap_values(X_test.iloc[0, :])
    plot = shap.force_plot(explainer.expected_value[0], shap_values[0], X_test.iloc[0, :], feature_names=iris['feature_names'])
    return plot

def st_shap(plot, height):
    import shap
    import streamlit.components.v1 as components
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

def show_shap_summary(model, X_train, X_test,iris):
    import shap
 
    #explainer = shap.Explainer(model.predict_proba, X_train)
    #shap_values = explainer(X_train)
    #sum = shap.summary_plot(shap_values, show = False)
    import matplotlib.pyplot as plt
    #plt.savefig('summary_plt.png')
    
    import sklearn
    from sklearn.model_selection import train_test_split
    import numpy as np
    import shap
    import time

    X_train,X_test,Y_train,Y_test = train_test_split(*shap.datasets.iris(), test_size=0.2, random_state=0)
    
    from sklearn.ensemble import RandomForestClassifier
    rforest = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=0)
    rforest.fit(X_train, Y_train)

    # explain all the predictions in the test set
    explainer = shap.KernelExplainer(rforest.predict_proba, X_train)
    shap_values = explainer.shap_values(X_test)
    plt =shap.force_plot(explainer.expected_value[0], shap_values[0], X_test)
    return plt

def run_app():
    import streamlit.components.v1 as components
    X, Y, iris = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, Y)
    model = train_model(X_train, y_train)
    acc = predict(model, X_test, y_test)
    st.dataframe(X)
    st.text('Acc is ' + str(round(acc * 100, 2)) + " %")  # - Information about algorithm
    i = np.random.randint(0, X_test.shape[0])
    show_input_data(X_train, i)

    col1, col2 = st.columns(2)
    shap = show_shap_plot(model, X_test, X_train, iris, i)
    with col1:
        st_shap(shap)

    limeplot = show_lime_plot(model, X_test, X_train, iris, i)
    with col2:
        st.components.v1.html(limeplot)

    summary = show_shap_summary(model,X_train)


def run_app():
    import streamlit.components.v1 as components
    X, Y, iris = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, Y)
    model = train_model(X_train, y_train)
    acc,predict_test =predict(model, X_test, y_test)
    
    
    
    #st.dataframe(X)
    
    i = np.random.randint(0, X_test.shape[0])
    
    data1 = pd.DataFrame(data= np.c_[iris['data']],
                     columns= iris['feature_names'])
    
    st.header("Input Information:")
    
    #how_input_data(X, i)
    st.dataframe(data1.iloc[i].T)
    
    st.text('Model acc. is ' + str(round(acc * 100, 2)) + " %")  # - Information about algorithm
    
    st.header("Explanation for Prediction:")
    
    st.subheader("Local Explaination")
    
    pred = predict_test[i]
    st.text('Prediction: '+ str(iris.target_names[pred]))
    
    col1, col2 = st.columns(2)
    shap = show_shap_plot(model, X_test, X_train, iris, i)
    #ith col1:
    st_shap(shap, None)

    limeplot = show_lime_plot(model, X_test, X_train, iris, i)
    #ith col2:
    st.components.v1.html(limeplot)
    
    string = lorem.words(50)
    st.markdown(string)
    
    st.subheader("Global Explaination")
    
    plt = show_shap_summary(model,X_train, X_test,iris)
    st_shap(plt, 400)
    #st.image('summary_plt.png')
    
    string = lorem.words(50)
    st.markdown(string)
    
    st.header("Hypothetical Scenario:")
    
    string = lorem.words(100)
    st.markdown(string)

def app():
    st.title('XAI-Dashboard pipeAI')
    run_app()
