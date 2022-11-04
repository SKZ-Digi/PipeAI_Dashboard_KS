import streamlit as st
from PIL import Image
import logic
from PIL import Image

def app():
    st.title('Associative Information about Use Case')

    st.header('Description of Use Case')

    image = Image.open('EDA/information_process.png')
    st.image(image,caption= 'Illustration of Use case',width= 600)

    st.markdown('The welding process goes as follows: Two plastic cuboids are heated at the end faces by a heating element and then welded together by pressing them against each other. A heat flux sensor is installed in the heating element, the voltage signal of which represents a measure of the heat flux flowing into the right-hand specimen. The welded specimens were then freed from the weld bead using a router and measured to failure in a three-point bend test to determine the weld factor. The weld factor is the ratio of the flexural strength of the welded specimen to the flexural strength of the raw material. If the weld factor is 1, then the strength of the weld is as high as that of the base material.')


    st.header('Design of Experiment')

    st.markdown('For the experiement various test points were approached, with 5 welds being made at each test point. In the experimental plan, the heating element temperature T_Heiz and the heating time t_anwere varied. Two materials PP and PVC were welded. During the welding process, the following parameters were recorded at a cycle rate of 5 Hz:')
    image = Image.open('EDA/Table_UseCase_method.jpg')
    st.image(image,caption= 'Captured Features', width= 600)

    st.markdown('In addition, the cross-sectional areas of the specimens were measured, as well as the surface temperature of the heating element at the beginning of each weld manually with an infrared thermometer. Time series data is also available from the three-point bend test, but the bending strength at the time of failure was extracted and used to calculate the weld factor. The weld factor can thus be used as a target value.')

    st.header('Data Processing')

    st.markdown('The raw data are available in time series (PARQUET data format), with the welds having different durations. All time series have been sychronized so that they start with the first contact of plastic body with heating element. The rest of the above data is available in the form of an Excel spreadsheet (data_summary.xlsx), with each sample defined by a unique identification number (META_ID). Some welds were defective, which is why they are marked with "-1" in the "Training" column.'
                'Based on expert knowledge, features were extracted from the time series that represent the most important points of the respective time series or describe the course of the curves. These are explained in more detail in the appendix.'
                'All data are combined into a single PARQUET file during preprocessing and the origin of the data is marked with a prefix. The PARQUET file can be represented in tabular form, with each row containing the data for one weld (with a one-to-one META_ID). The time series data are summarized, in which they are deposited as array in a cell. The TIMESERIES_Timestamp[s] column contains the time stamps of the time series belonging to the weld of the respective row. To plot the time series data, ergo the values of the TIMESERIES_Timestamp[s] cell can be used as x-values and the values of a time series cell, such as TIMESERIES_force_right[N], can be used as y-values.'
                'The following prefixes are assigned:)')

    image = Image.open('EDA/Table_UseCase_data.jpg')
    st.image(image,caption= 'Preprocssed Features',width= 600)

    st.write('Author: Jonathan Lambers, Company: German Plastics Center')
    st.write('See Lambers and Balzer (2022) for more information and data')

    st.sidebar.image(logic.add_logo(logo_path="images/logo_pipeai.png", width=291, height=100)) #hier
    hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
    st.markdown(hide_menu_style, unsafe_allow_html=True)