import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import tensorflow as tf
from anchor import anchor_tabular
import shap
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
import pickle
from PIL import Image


def add_logo(logo_path, width, height):
    logo = Image.open(logo_path)
    modified_logo = logo.resize((width, height))
    return modified_logo

def load_and_split_data():

    df = pd.read_parquet('data/data_processed_pipeAIKunststoff')

    # Use only the FEAT columns plus the Label column
    select_cols = [col for col in df if col.startswith('FEAT')]
    select_cols.append('LABEL_weld_factor')
    df = df[select_cols]

    # Separate into materials
    df_PP = df[df['FEAT_Material_PP'] == 1]
    df_PVC = df[df['FEAT_Material_PVC'] == 1]

    df_PP.drop(columns=['FEAT_Material_PP', 'FEAT_Material_PVC'], inplace=True)
    df_PVC.drop(columns=['FEAT_Material_PP', 'FEAT_Material_PVC'], inplace=True)

    df = df_PP

    df['LABEL_weld_factor'] = df['LABEL_weld_factor'].mask(df['LABEL_weld_factor'] >= 0.80, 1)
    df['LABEL_weld_factor'] = df['LABEL_weld_factor'].mask(df['LABEL_weld_factor'] < 0.80, 0)
    X = df.drop(columns=['LABEL_weld_factor'])
    Y =  df['LABEL_weld_factor']

    ros = RandomOverSampler(random_state=42)
    smote = SMOTE()
    x_smote, y_smote = smote.fit_resample(X, Y)
    df_new = x_smote
    df_new['LABEL_weld_factor'] = y_smote

    split = 0.7
    seed = 42

    #df_new = shuffle(df_new, random_state = seed)
    df_train = df_new.iloc[0:int(split*df.shape[0])]
    df_val = df_new.iloc[int(split*df.shape[0]):]

    train_std = df_train.mean()
    train_mean = df_train.std()

    y_train = df_train['LABEL_weld_factor']
    y_test = df_val['LABEL_weld_factor']


    df_train_scaled = (df_train - train_mean) / train_std
    df_val_scaled = (df_val - train_mean) / train_std

    X_train = df_train_scaled.drop(columns=['LABEL_weld_factor'])
    X_test = df_val_scaled.drop(columns=['LABEL_weld_factor'])

    return X_train, y_train, X_test, y_test

def load_data_stored():
    split = 0.7
    seed = 42
    df_new2 = pd.read_csv('data/data_sampled_preprocessed.csv')
    df_new2 = df_new2.drop(columns= ['Unnamed: 0'])
    df_train = df_new2.iloc[0:int(split*df_new2.shape[0])]
    df_val = df_new2.iloc[int(split*df_new2.shape[0]):]

    train_std = df_train.mean()
    train_mean = df_train.std()

    y_train = df_train['LABEL_weld_factor']
    y_test = df_val['LABEL_weld_factor']

    y_train_int = df_train['LABEL_weld_factor'].apply(np.int64)
    y_test_int = df_val['LABEL_weld_factor'].apply(np.int64)

    df_train_scaled = (df_train - train_mean) / train_std
    df_val_scaled = (df_val - train_mean) / train_std

    X_train = df_train_scaled.drop(columns=['LABEL_weld_factor'])
    X_test = df_val_scaled.drop(columns=['LABEL_weld_factor'])

    return X_train, y_train, X_test, y_test, y_train_int, y_test_int


def load_model(path):

    new_model = pickle.load(open(path, 'rb'))
    return new_model

def rate_model(X_test,y_test, model):


    from sklearn.metrics import plot_confusion_matrix
    from sklearn.metrics import confusion_matrix
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    ps = precision_score(y_test, pred)
    rs = recall_score(y_test, pred)
    cm = confusion_matrix(y_test, pred)
    plot_confusion_matrix(model, X_test, y_test)

    return acc,f1,ps,rs,cm

def make_predicion(X_val,y_val, model,i):
    proba = pd.DataFrame(model.predict_proba(X_val[i-1:i]))
    pred = model.predict(X_val[i-1:i])

    prob_null = round(proba.iloc[0,0]*100,2)
    prob_one = round(proba.iloc[0,1]*100,2)

    predi = [proba[0]]
    if pred[0] == 0:
        return prob_null,'niO'

    if pred[0] == 1:
        return prob_one,'iO'


def shap_values(model, X_train, X_val):
    ex = shap.KernelExplainer(model.predict, X_train)
    shap_values = ex.shap_values(X_val)

    return shap_values

def lime_explainer(X_train,i,clf,X_test):

    clf_prob = lambda x: clf.predict_proba(x).astype(float)

    from lime import lime_tabular

    explainer = lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=X_train.columns,
        class_names= ['niO','iO'],
        mode='classification'
    )
    exp = explainer.explain_instance(
        X_test.iloc[i],
        clf_prob,
        #num_features = 8
    )
    return exp

def show_force_plot(explainer,X_test,i):
    #st.dataframe(X_test.iloc[i,:])
    shap_values = explainer.shap_values(X_test.iloc[i,:])
    plot = shap.force_plot(explainer.expected_value[0], shap_values[0], X_test.iloc[i,:])
    return plot

def st_shap(plot, height):
    import shap
    import streamlit.components.v1 as components

    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


def load_shap_values(name):
    import pickle

    #st.write(os.getcwd())
    #arr = os.listdir()
    #st.write(arr)

    file_to_read = open(name, "rb")
    loaded_object = pickle.load(file_to_read)
    file_to_read.close()
    return loaded_object

def load_shap_explainer():
    import pickle
    file_to_read = open('models/explainer_global.sav', "rb")
    loaded_object = pickle.load(file_to_read)
    file_to_read.close()
    return loaded_object

def lime_show_notebook(explainer, height = None):
    import streamlit.components.v1 as components
    components.html(explainer.as_html(), height=700)

def anchor_show_notebook(explainer, height = None):
    import streamlit.components.v1 as components
    components.html(explainer.as_html(), height=1200)

def return_model_path():
    return 'models/clf_model.sav'

def load_and_show_model(path,to_path):
    model = load_model(path)
    tf.keras.utils.plot_model(model, to_file=to_path, show_shapes=True)

def load_anchors(i, X_train, X_val, y_train_int):
    from sklearn.neural_network import MLPClassifier
    X_train_num = X_train.to_numpy()
    clf = MLPClassifier((32,32,32,32,16,1),random_state=1, max_iter=500, activation='relu').fit(X_train_num, y_train_int)

    feature_names = X_train.columns.to_list()
    class_names = ['niO','iO']

    explainer = anchor_tabular.AnchorTabularExplainer(
        class_names,
        feature_names,
        X_train_num
    )

    X_test_num = X_val.to_numpy()
    exp = explainer.explain_instance(X_test_num[i], clf.predict, threshold=0.95)
    return exp
