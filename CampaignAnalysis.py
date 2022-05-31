# Importing libraries
import pandas as pd
import numpy as np
from st_aggrid import AgGrid

from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot
import plotly.figure_factory as ff

import streamlit as st
import pandas as pd
import numpy as np
import scipy as sp
import plotly as py
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import datetime as dt
import matplotlib.pyplot as plt
from PIL import Image
from sklearn import datasets
from statistics import mean

import json
import requests
import hydralit_components as hc
from streamlit_lottie import st_lottie

from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

from streamlit_option_menu import option_menu
from pandas_profiling import ProfileReport
#import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
st.set_page_config(layout = 'wide')
#-----------------------------------------------------------------------------#
# Function to load animations
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)
#-----------------------------------------------------------------------------#
# Main Title
#st.title('Marketing Campaign Key Influencers and Customer Segmentation')
st.markdown(f"""
        <h1>
            <h1 style="vertical-align:center;font-size:55px;padding-left:200px;color:#636EFA;padding-top:5px;margin-left:0em";>
            Marketing Campaign Key Influencers and Customer Segmentation
        </h1>""",unsafe_allow_html = True)
#-----------------------------------------------------------------------------#
# Sidebar - To collect user input dataframe
with st.sidebar.header('Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
#-----------------------------------------------------------------------------#
# Menu bar
selected = option_menu(
    menu_title = None,
    options = ["Home", "Data", "Key Influencers", "RFM Analysis", "Segment Behavior", "Demographics"],
    menu_icon = "cast",
    icons = ['house', 'cloud-upload', 'key', 'gear', 'bar-chart', 'pie-chart'],
    default_index = 0,
    orientation = "horizontal",
    styles = {"nav-link-selected":{"background-color":"#636EFA"}}
)
#-----------------------------------------------------------------------------#
# Component1: Home
if selected == "Home":
    # Splitting page into 2 columns
    col1, col2 = st.columns([3,2])
    with col1:
        st.markdown(f"""
                <h2>
                    <h4 style="vertical-align:center;font-size:45px;color:#8dbdff;padding-left:200px;padding-top:5px;margin-left:0em";>
                    Welcome!
                </h2>""",unsafe_allow_html = True)

        st.markdown(f"""
                <h4>
                    <h4 style="vertical-align:center;font-size:30px;color:#ffb0d9;padding-left:200px;padding-top:5px;margin-left:0em";>
                    Boost Your Marketing Campaign Performance

                    1- Determine Your Key Influencers
                    2- Uncover Hidden Customer Segments
                    3- Understand Customer Behavior by Segment
                    4- Explore Customer Demographics
                </h4>""",unsafe_allow_html = True)

        st.markdown(f"""
                <h4>
                    <h4 style="vertical-align:center;font-size:20px;color:#636EFA;padding-left:200px;padding-top:5px;margin-left:0em";>
                    <em> The only way to win at content marketing is for the reader to say “this was written specifically for me!” - Jamie Turner <em>
                </h4>""",unsafe_allow_html = True)

    with col2:
        coding_lottie = load_lottiefile("coding.json")
        st_lottie(coding_lottie,
        speed = 1,
        reverse = False,
        loop = True,
        quality = 'high',
        height = 500,
        width = 500,
        key = None
        )
#-----------------------------------------------------------------------------#
# Component 2: Data
if selected == "Data":

    # Splitting page into 2 columns
    col1, col2 = st.columns([3, 1])
    with col1:
        # Displays the dataset
        if uploaded_file is not None:
            df_X = pd.read_csv(uploaded_file)
            st.subheader('A Glimpse of Your Data')

            st.dataframe(df_X)
            st.write('Your dataset contains', df_X.shape[0], 'rows and', df_X.shape[1], 'columns.')
        else:
            st.info('Awaiting for CSV file to be uploaded.')
            if st.button("Press to see last quarter's data"):
                data = pd.read_csv('ifood_df.csv')
                df_X = data.copy()
                df_X.drop(columns = 'Response', axis = 1, inplace = True)
                X = df_X.copy()

                y = data['Response']
                df = pd.concat( [X,y], axis=1)
                st.subheader("A Glimpse of the Last Quarter's Data")

                st.dataframe(df)
                st.write('This dataset contains', df.shape[0], 'rows and', df.shape[1], 'columns.')
        with col2:
            upload_data_lottie = load_lottiefile("upload1.json")
            st_lottie(upload_data_lottie,
            speed = 1,
            reverse = False,
            loop = True,
            quality = 'high',
            height = 190,
            width = 250,
            key = None)
#-----------------------------------------------------------------------------#
# Component 3: Key Influencers
if selected == "Key Influencers":

    # Dataset of the user
    if uploaded_file is not None:
        df_X = pd.read_csv(uploaded_file)

        ### Displaying feature importance ###

        # Subsetting X
        df_XX = df_X.copy()
        df_XX.drop(columns = 'Response', axis = 1, inplace = True)
        X = df_XX.copy()

        # Subsetting y
        y = df_X['Response']

        # Data splitting
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 1)

        # Model
        lr = LogisticRegression(random_state = 1)

        # Define min max MinMaxScaler
        scaler = MinMaxScaler()

        # Transform data
        scaled = scaler.fit_transform(X_train)

        # Train the classifier
        lr.fit(scaled, y_train)

        # Get feature importance
        importance = lr.coef_[0]
        importance_abs = abs(importance)

        # Summarize feature importance
        column_names = df_X.columns.tolist()
        feature_importance = pd.DataFrame(zip(column_names, importance, importance_abs),
                                      columns = ['Feature Name', 'Actual Coefficient', 'Coefficient'])


        # Sorting features in descending order
        feature_importance.sort_values(by = "Coefficient", ascending=False, inplace=True)

        ### Output ###
        col1, col2, col3, col4, col5, col6 = st.columns(6)

        with col1:
            theme_override = {'bgcolor': '#d6eaff','title_color': '#636EFA', 'content_color': '000000', 'icon': 'bi bi-key', 'icon_color': 'blue'}
            hc.info_card(title = 'Influencer 1', content = feature_importance.iloc[0,0],
            theme_override=theme_override)
        with col2:
            theme_override = {'bgcolor': '#d6eaff','title_color': '#636EFA', 'content_color': '000000', 'icon': 'bi bi-key', 'icon_color': 'blue'}
            hc.info_card(title = 'Influencer 2', content = feature_importance.iloc[1,0],
            theme_override=theme_override)
        with col3:
            theme_override = {'bgcolor': '#d6eaff','title_color': '#636EFA', 'content_color': '000000', 'icon': 'bi bi-key', 'icon_color': 'blue'}
            hc.info_card(title = 'Influencer 3', content = feature_importance.iloc[2,0],
            theme_override=theme_override)
        with col4:
            theme_override = {'bgcolor': '#d6eaff','title_color': '#636EFA', 'content_color': '000000', 'icon': 'bi bi-key', 'icon_color': 'blue'}
            hc.info_card(title = 'Influencer 4', content = feature_importance.iloc[3,0],
            theme_override=theme_override)
        with col5:
            theme_override = {'bgcolor': '#d6eaff','title_color': '#636EFA', 'content_color': '000000', 'icon': 'bi bi-key', 'icon_color': 'blue'}
            hc.info_card(title = 'Influencer 5', content = feature_importance.iloc[4,0],
            theme_override=theme_override)
        with col6:
            theme_override = {'bgcolor': '#d6eaff','title_color': '#636EFA', 'content_color': '000000', 'icon': 'bi bi-key', 'icon_color': 'blue'}
            hc.info_card(title = 'Influencer 6', content = feature_importance.iloc[5,0],
            theme_override=theme_override)
        st.write(' ')
        # columns in row 2
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
                <h4 style="vertical-align:center;font-size:30px;color:#636EFA;padding-left:200px;padding-top:5px;margin-left:0em";>
                Campaign Response Rates
                </h2>""",unsafe_allow_html = True)

            campaign_response = df_X['Response'].sum()
            campaign_noresponse = df_X['Response'].shape[0]

            data = {'response_type' : ['Responded','NoResponse'],
                    'count' : [campaign_response, campaign_noresponse]}
            df_response = pd.DataFrame(data)

            # pie chart
            fig = px.pie(df_response, values = 'count', names = 'response_type')

            colors = ['#b3d4ff','#ffb0d9']
            fig.update_traces(textfont_size=15, marker=dict(colors=colors))

            st.plotly_chart(fig)

        with col3:
            # Display the table
            st.markdown(f"""
                <h4 style="vertical-align:center;font-size:30px;color:#636EFA;padding-left:200px;padding-top:5px;margin-left:0em";>
                Influencer Importance
                </h2>""",unsafe_allow_html = True)
            st.write(feature_importance)

            # Explore the features further
            if st.button('Run statistical test'):
                profile = ProfileReport(df, explorative = True)
                st_profile_report(profile)

        with col2:
            influencer2_lottie = load_lottiefile("influencer2.json")
            st_lottie(influencer2_lottie,
            speed = 1,
            reverse = False,
            loop = True,
            quality = 'high',
            height = 500,
            width = 550,
            key = None)
# ---------------------------------------------------------------
    # Our dataset
    else:
        data = pd.read_csv('ifood_df.csv')
        df_X = data.copy()
        df_X.drop(columns = 'Response', axis = 1, inplace = True)
        X = df_X.copy()
        y = data['Response']
        df = pd.concat( [X,y], axis=1)

        ### Displaying feature importance ###
        # Subsetting X
        df_XX = df.copy()
        df_XX.drop(columns = 'Response', axis = 1, inplace = True)
        X = df_XX.copy()

        # Subsetting y
        y = df['Response']

        # Data splitting
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 1)

        # Model
        lr = LogisticRegression(random_state = 1)

        # Define min max MinMaxScaler
        scaler = MinMaxScaler()

        # Transform data
        scaled = scaler.fit_transform(X_train)

        # Train the classifier
        lr.fit(scaled, y_train)

        # Get feature importance
        importance = lr.coef_[0]
        importance_abs = abs(importance)

        # Summarize feature importance
        column_names = df_X.columns.tolist()
        feature_importance = pd.DataFrame(zip(column_names, importance, importance_abs),
                                      columns = ['Feature Name', 'Actual Coefficient', 'Coefficient'])


        # Sorting features in descending order
        feature_importance.sort_values(by = "Coefficient", ascending=False, inplace=True)

        ### Output ###
        col1, col2, col3, col4, col5, col6 = st.columns(6)

        with col1:
            theme_override = {'bgcolor': '#d6eaff','title_color': '#636EFA', 'content_color': '000000', 'icon': 'bi bi-key', 'icon_color': 'blue'}
            hc.info_card(title = 'Influencer 1', content = feature_importance.iloc[0,0],
            theme_override=theme_override)

        with col2:
            theme_override = {'bgcolor': '#d6eaff','title_color': '#636EFA', 'content_color': '000000', 'icon': 'bi bi-key', 'icon_color': 'blue'}
            hc.info_card(title = 'Influencer 2', content = feature_importance.iloc[1,0],
            theme_override=theme_override)

        with col3:
            theme_override = {'bgcolor': '#d6eaff','title_color': '#636EFA', 'content_color': '000000', 'icon': 'bi bi-key', 'icon_color': 'blue'}
            hc.info_card(title = 'Influencer 3', content = feature_importance.iloc[2,0],
            theme_override=theme_override)
        with col4:
            theme_override = {'bgcolor': '#d6eaff','title_color': '#636EFA', 'content_color': '000000', 'icon': 'bi bi-key', 'icon_color': 'blue'}
            hc.info_card(title = 'Influencer 4', content = feature_importance.iloc[3,0],
            theme_override=theme_override)
        with col5:
            theme_override = {'bgcolor': '#d6eaff','title_color': '#636EFA', 'content_color': '000000', 'icon': 'bi bi-key', 'icon_color': 'blue'}
            hc.info_card(title = 'Influencer 5', content = feature_importance.iloc[4,0],
            theme_override=theme_override)
        with col6:
            theme_override = {'bgcolor': '#d6eaff','title_color': '#636EFA', 'content_color': '000000', 'icon': 'bi bi-key', 'icon_color': 'blue'}
            hc.info_card(title = 'Influencer 6', content = feature_importance.iloc[5,0],
            theme_override=theme_override)

        st.write(' ')

        # columns in row 2
        col1, col2, col3 = st.columns(3)

        with col1:
            # Visualizing customer response to marketing campaigns
            # Visualizing the impact of top 5 features
            st.markdown(f"""
                <h4 style="vertical-align:center;font-size:30px;color:#636EFA;padding-left:200px;padding-top:5px;margin-left:0em";>
                Influencer Impact
                </h2>""",unsafe_allow_html = True)
            features_pos = [2, 4]
            features_neg = [1, 3, 5]
            fig = go.Figure()

            # Negative
            fig.add_trace(go.Bar(x=features_neg, y=[2.92, 1.90, 1.74],
                            base=[-2.92,-1.90,-1.74],
                            marker_color='#ffb0d9',
                            name='Negative Impact            ',
                            hovertext = ['Recency', 'NumStorePurchases', 'Teenhome']))
            # Positive
            fig.add_trace(go.Bar(x=features_pos, y=[2.65, 1.88],
                            base=0,
                            marker_color='#b3d4ff',
                            name='Positive Impact            ',
                            hovertext = ['Customer_Days', 'MntMeatProducts']
                            ))

            # Remove gridlines
            fig.update_layout(xaxis=dict(showgrid=False),
                          yaxis=dict(showgrid=False))

            fig.update_layout(autosize=False,
            height = 500,
            width = 700)

            st.plotly_chart(fig)

        with col2:
            # Loading lottie
            influencer2_lottie = load_lottiefile("influencer2.json")
            st_lottie(influencer2_lottie,
            speed = 1,
            reverse = False,
            loop = True,
            quality = 'high',
            height = 500,
            width = 500,
            key = None)

        with col3:
            # Display the table
            st.markdown(f"""
                <h4 style="vertical-align:center;font-size:30px;color:#636EFA;padding-left:200px;padding-top:5px;margin-left:0em";>
                Influencer Importance
                </h2>""",unsafe_allow_html = True)
            st.write('')
            st.write('')
            st.write('')
            st.write(feature_importance)
            st.write('')
            # Explore the features further
            if st.button('Run statistical test'):
                profile = ProfileReport(df, explorative = True)
                st_profile_report(profile)

#-----------------------------------------------------------------------------#
# Component 4: RFM Analysis
if selected == "RFM Analysis":

    # Dataset of the user
    if uploaded_file is not None:
        df_X = pd.read_csv(uploaded_file)

        # Adding 2 new columns
        df_X['Monetary'] = df_X['MntWines'] + df_X['MntFruits'] + df_X['MntMeatProducts'] + df_X['MntFishProducts'] + df_X['MntSweetProducts'] + df_X['MntGoldProds'] + df_X['MntRegularProds']
        df_X['Frequency'] = df_X['NumDealsPurchases'] + df_X['NumWebPurchases'] + df_X['NumCatalogPurchases'] + df_X['NumStorePurchases']

        ### Calculate Recency, Frequency, Monetary ###
        # Subsetting the dataframe to only include the features: 'Recency', 'Frequency', 'Monetary'
        df_RFM = df_X.copy()
        columns = ['Recency', 'Frequency', 'Monetary']
        df_RFM = df_RFM[columns]

        # Plot RFM Distributions
        col1, col2, col3 = st.columns(3)

        with col1:
            # Plot the distribution of R
            x = df_RFM['Recency']
            hist_data = [x]
            group_labels = ['df_RFM']

            colors = ['#8dbdff']
            fig = ff.create_distplot(hist_data, group_labels, colors = colors)
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=False)

            # Aligning title
            fig.update_layout(
                title={
                    'text': "Recency Distribution",
                    'x':0.49,
                    'xanchor': 'center',
                    'yanchor': 'top'})
            st.plotly_chart(fig)

        with col2:
            # Plot the distribution of F
            x = df_RFM['Frequency']
            hist_data = [x]
            group_labels = ['df_RFM']

            colors = ['#ffb0d9']
            fig = ff.create_distplot(hist_data, group_labels, colors = colors)
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=False)

            # Aligning title
            fig.update_layout(
                title={
                    'text': "Frequency Distribution",
                    'x':0.49,
                    'xanchor': 'center',
                    'yanchor': 'top'})
            st.plotly_chart(fig)

        with col3:
            # Plot the distribution of M
            x = df_RFM['Monetary']
            hist_data = [x]
            group_labels = ['df_RFM']

            colors = ['#636EFA']
            fig = ff.create_distplot(hist_data, group_labels, colors = colors)
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=False)

            # Aligning title
            fig.update_layout(
                title={
                    'text': "Monetary Distribution",
                    'x':0.49,
                    'xanchor': 'center',
                    'yanchor': 'top'})
            st.plotly_chart(fig)

        # Choose number of clusters
        K = 3
        model = KMeans(n_clusters = K, random_state = 1)
        model.fit(df_RFM)
        df_RFM['Cluster'] = model.labels_

        # Change the cluster columns data type into discrete values
        df_RFM['Segment'] = 'Segment ' + df_RFM['Cluster'].astype('str')

        # Calculate average values for each cluster and return size of each
        df_agg = df_RFM.groupby('Segment').agg({
                                                'Recency' : 'mean',
                                                'Frequency' : 'mean',
                                                'Monetary' : ['mean', 'count']}).round(0)

        df_agg.columns = df_agg.columns.droplevel()
        df_agg.columns = ['RecencyMean', 'FrequencyMean', 'MonetaryMean', 'Count']
        df_agg['Percent'] = round((df_agg['Count']/df_agg.Count.sum())*100,2)

        # Reset the index
        df_agg = df_agg.reset_index()

        # RE-ARRANGING ONLY FOR VISUALIZATION PURPOSES
        test = {'Segment 0': 'Segment 2', 'Segment 2': 'Segment 0', 'Segment 1': 'Segment 1'}
        df_RFM['seg'] = df_RFM['Segment']
        df_RFM = df_RFM.replace({'seg':test})

        # Visualizing the customer segments (clusters)
        col1, col2, col3 = st.columns(3)
        with col1:
            # 2D Scatterplot
            fig = px.scatter(df_agg, x = 'RecencyMean', y = 'MonetaryMean', size = 'FrequencyMean',
                color = 'Segment', hover_name = 'Segment', size_max = 100, color_discrete_sequence = ['#8dbdff', '#ffb0d9', '#636EFA'])

            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=False)

            # Aligning title
            fig.update_layout(
                title={
                    'text': "Segment Spending as a Function of Recency",
                    'x':0.48,
                    'y':0.99,
                    'xanchor': 'center',
                    'yanchor': 'top'})
            st.plotly_chart(fig)

        with col2:
            # 3D Scatterplot
            fig = px.scatter_3d(df_RFM, x = 'Recency', y = 'Frequency', z = 'Monetary',
                        color = 'seg', hover_name = 'Segment', opacity = 0.5, color_discrete_sequence = ['#8dbdff', '#ffb0d9', '#636EFA'])
            fig.update_traces(marker = dict(size = 3), selector = dict(mode = 'markers'))

            # Aligning title
            fig.update_layout(
                title={
                    'text': "Customer Segments",
                    'x':0.48,
                    'y':0.99,
                    'xanchor': 'center',
                    'yanchor': 'top'})
            st.plotly_chart(fig)

        with col3:
            # Pie
            fig = px.pie(df_agg, values = 'Percent', names = 'Segment',
                        color = 'Segment', opacity = 0.9, color_discrete_sequence = ['#8dbdff', '#ffb0d9', '#636EFA'])

            # Aligning title
            fig.update_layout(
                title={
                    'text': "Segment Size",
                    'x':0.48,
                    'y':0.99,
                    'xanchor': 'center',
                    'yanchor': 'top'})
            st.plotly_chart(fig)

        # Adding cluster label to each row in original dataframe
        df_X['Cluster'] = model.labels_
        df_X['Segment'] = 'Segment ' + df_X['Cluster'].astype('str')

    # Our dataset
    else:
        data = pd.read_csv('ifood_df.csv')

        # Adding 2 new columns
        data['Monetary'] = data['MntWines'] + data['MntFruits'] + data['MntMeatProducts'] + data['MntFishProducts'] + data['MntSweetProducts'] + data['MntGoldProds'] + data['MntRegularProds']
        data['Frequency'] = data['NumDealsPurchases'] + data['NumWebPurchases'] + data['NumCatalogPurchases'] + data['NumStorePurchases']

        ### Calculate Recency, Frequency, Monetary ###
        # Subsetting the dataframe to only include the features: 'Recency', 'Frequency', 'Monetary'
        df_RFM = data.copy()
        columns = ['Recency', 'Frequency', 'Monetary']
        df_RFM = df_RFM[columns]

        # Plot RFM Distributions
        col1, col2, col3 = st.columns(3)

        with col1:
            # Plot the distribution of R
            x = df_RFM['Recency']
            hist_data = [x]
            group_labels = ['df_RFM']

            colors = ['#8dbdff']
            fig = ff.create_distplot(hist_data, group_labels, colors = colors)
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=False)

            # Aligning title
            fig.update_layout(
                title={
                    'text': "Recency Distribution",
                    'x':0.49,
                    'xanchor': 'center',
                    'yanchor': 'top'})
            st.plotly_chart(fig)

        with col2:
            # Plot the distribution of F
            x = df_RFM['Frequency']
            hist_data = [x]
            group_labels = ['df_RFM']

            colors = ['#ffb0d9']
            fig = ff.create_distplot(hist_data, group_labels, colors = colors)
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=False)

            # Aligning title
            fig.update_layout(
                title={
                    'text': "Frequency Distribution",
                    'x':0.49,
                    'xanchor': 'center',
                    'yanchor': 'top'})
            st.plotly_chart(fig)

        with col3:
            # Plot the distribution of M
            x = df_RFM['Monetary']
            hist_data = [x]
            group_labels = ['df_RFM']

            colors = ['#636EFA']
            fig = ff.create_distplot(hist_data, group_labels, colors = colors)
            fig.update_layout(title_text = 'Monetary')
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=False)

            # Aligning title
            fig.update_layout(
                title={
                    'text': "Monetary Distribution",
                    'x':0.49,
                    'xanchor': 'center',
                    'yanchor': 'top'})
            st.plotly_chart(fig)

        # Choose number of clusters
        K = 3
        model = KMeans(n_clusters = K, random_state = 1)
        model.fit(df_RFM)
        df_RFM['Cluster'] = model.labels_

        # Change the cluster columns data type into discrete values
        df_RFM['Segment'] = 'Segment ' + df_RFM['Cluster'].astype('str')

        # Calculate average values for each cluster and return size of each
        df_agg = df_RFM.groupby('Segment').agg({
                                                'Recency' : 'mean',
                                                'Frequency' : 'mean',
                                                'Monetary' : ['mean', 'count']}).round(0)

        df_agg.columns = df_agg.columns.droplevel()
        df_agg.columns = ['RecencyMean', 'FrequencyMean', 'MonetaryMean', 'Count']
        df_agg['Percent'] = round((df_agg['Count']/df_agg.Count.sum())*100,2)

        # Reset the index
        df_agg = df_agg.reset_index()

        # RE-ARRANGING ONLY FOR VISUALIZATION PURPOSES
        test = {'Segment 0': 'Segment 2', 'Segment 2': 'Segment 0', 'Segment 1': 'Segment 1'}
        df_RFM['seg'] = df_RFM['Segment']
        df_RFM = df_RFM.replace({'seg':test})

        # Visualizing the customer segments (clusters)
        col1, col2, col3 = st.columns(3)
        with col1:
            # 2D Scatterplot
            fig = px.scatter(df_agg, x = 'RecencyMean', y = 'MonetaryMean', size = 'FrequencyMean',
                color = 'Segment', hover_name = 'Segment', size_max = 100, color_discrete_sequence = ['#8dbdff', '#ffb0d9', '#636EFA'])
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=False)

            # Aligning title
            fig.update_layout(
                title={
                    'text': "Segment Spending as a Function of Recency",
                    'x':0.48,
                    'y':0.99,
                    'xanchor': 'center',
                    'yanchor': 'top'})
            st.plotly_chart(fig)

        with col2:
            # 3D Scatterplot
            fig = px.scatter_3d(df_RFM, x = 'Recency', y = 'Frequency', z = 'Monetary',
                        color = 'seg', opacity = 0.5, color_discrete_sequence = ['#8dbdff', '#ffb0d9', '#636EFA'])
            fig.update_traces(marker = dict(size = 3), selector = dict(mode = 'markers'))

            # Aligning title
            fig.update_layout(
                title={
                    'text': "Customer Segments",
                    'x':0.48,
                    'y':0.99,
                    'xanchor': 'center',
                    'yanchor': 'top'})
            st.plotly_chart(fig)

        with col3:
            # Pie
            fig = px.pie(df_agg, values = 'Percent', names = 'Segment',
                        color = 'Segment', opacity = 0.9, color_discrete_sequence = ['#8dbdff', '#ffb0d9', '#636EFA'])
            # Aligning title
            fig.update_layout(
                title={
                    'text': "Segment Size",
                    'x':0.48,
                    'y':0.99,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            st.plotly_chart(fig)

        # Adding cluster label to each row in original dataframe
        data['Cluster'] = model.labels_
        data['Segment'] = 'Segment ' + data['Cluster'].astype('str')
#-----------------------------------------------------------------------------#
# Component 5: Segment Behavior
if selected == "Segment Behavior":

    # Output
    option = st.selectbox(
        'Select a customer segment',
        ('Segment 0', 'Segment 1', 'Segment 2'))

    # Dataset of the user
    if uploaded_file is not None:
        df_X = pd.read_csv(uploaded_file)

        # Adding 2 new columns
        df_X['Monetary'] = df_X['MntWines'] + df_X['MntFruits'] + df_X['MntMeatProducts'] + df_X['MntFishProducts'] + df_X['MntSweetProducts'] + df_X['MntGoldProds'] + df_X['MntRegularProds']
        df_X['Frequency'] = df_X['NumDealsPurchases'] + df_X['NumWebPurchases'] + df_X['NumCatalogPurchases'] + df_X['NumStorePurchases']

        ### Calculate Recency, Frequency, Monetary ###
        # Subsetting the dataframe to only include the features: 'Recency', 'Frequency', 'Monetary'
        df_RFM = df_X.copy()
        columns = ['Recency', 'Frequency', 'Monetary']
        df_RFM = df_RFM[columns]

        # Choose number of clusters
        K = 3
        model = KMeans(n_clusters = K, random_state = 1)
        model.fit(df_RFM)
        df_RFM['Cluster'] = model.labels_

        # Change the cluster columns data type into discrete values
        df_RFM['Segment'] = 'Segment ' + df_RFM['Cluster'].astype('str')

        # Calculate average values for each cluster and return size of each
        df_agg = df_RFM.groupby('Segment').agg({
                                                'Recency' : 'mean',
                                                'Frequency' : 'mean',
                                                'Monetary' : ['mean', 'count']}).round(0)

        df_agg.columns = df_agg.columns.droplevel()
        df_agg.columns = ['RecencyMean', 'FrequencyMean', 'MonetaryMean', 'Count']
        df_agg['Percent'] = round((df_agg['Count']/df_agg.Count.sum())*100,2)

        # Reset the index
        df_agg = df_agg.reset_index()

        # Adding cluster label to each row in original dataframe
        df_X['Cluster'] = model.labels_
        df_X['Segment'] = 'Segment ' + df_X['Cluster'].astype('str')

        if option == 'Segment 0':

            ### Treemap for segment 0 ###
            # Step 1- Subsetting the df for Cluster 0
            cluster0 = df_X[df_X['Segment'] == 'Segment 0']

            # Step 2- Calculating the amount spent ($) for each product category
            wine0 = cluster0['MntWines'].sum()
            fruit0 = cluster0['MntFruits'].sum()
            meat0 = cluster0['MntMeatProducts'].sum()
            fish0 = cluster0['MntFishProducts'].sum()
            sweet0 = cluster0['MntSweetProducts'].sum()
            gold0 = cluster0['MntGoldProds'].sum()
            regular0 = cluster0['MntRegularProds'].sum()

            # Getting the name of the cluster
            name0 = cluster0.iloc[0,-1]

            # Step 3- df_0
            data = {'Segment' : [name0, name0, name0, name0, name0, name0, name0],
                   'ProductCategory' : ['Wine', 'Fruit', 'Meat', 'Fish', 'Sweets', 'Gold', 'Regular Products'],
                   'Monetary' : [wine0, fruit0, meat0, fish0, sweet0, gold0, regular0]}
            df_0 = pd.DataFrame(data)

            # Step 4- Plot Treemap
            fig2 = px.treemap(df_0,
                             names = 'ProductCategory',
                             values = 'Monetary',
                             parents = 'Segment',
                             color = 'ProductCategory',
                             color_discrete_sequence = ['#f9c9e2', '#d6eaff',
                                                        '#f7e4f8', '#8dbdff', '#8dbdff', '#ffb0d9', '#b3d4ff'])
            # Aligning title
            fig2.update_layout(
                title={
                    'text': "Spending by Product Category",
                    'x':0.48,
                    #'y':0.99,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            ### Pie chart for segment 0 ###
            # Step 1- Subsetting the df for Cluster 0
            pie0 = cluster0[cluster0['Segment'] == 'Segment 0']

            # Step 2- Calculating the sum for each channel
            web0 = pie0['NumWebPurchases'].sum()
            store0 = pie0['NumStorePurchases'].sum()
            catalog0 = pie0['NumCatalogPurchases'].sum()

            # Getting the name of the cluster
            name0 = cluster0.iloc[0,-1]

            # Step 3- df_0
            data = {'Segment' : [name0, name0, name0],
                   'Channel' : ['Website', 'In-Store', 'Catalog'],
                   'Count' : [web0, store0, catalog0]}
            df_0 = pd.DataFrame(data)

            # Step 4- Plot pie
            fig6 = px.pie(df_0, values = 'Count', names = 'Channel', color = 'Channel',
            color_discrete_sequence = ['#8dbdff', '#b3d4ff', '#d6eaff'])

            # Aligning title
            fig6.update_layout(
                title={
                    'text': "Shopping Channels",
                    'y':0.99,
                    'x':0.49,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            ### Scatterplot for segment 0 ###
            # Relationship between web visits and deals purchased
            fig4 = px.scatter(cluster0, x='NumWebVisitsMonth', y='NumDealsPurchases',
                                trendline="ols", color_discrete_sequence = ['#ffb0d9'])

            # Remove gridlines
            fig4.update_layout(xaxis=dict(showgrid=False),
                            yaxis=dict(showgrid=False))

            # Aligning title
            fig4.update_layout(
                title={
                    'text': "Deals Purchased by Monthly Webiste Visits",
                    'x':0.48,
                    'y':0.99,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            ### Stacked Barchart for segment 0 ###
            # Amount spent on each product category by each marital status of cluster 0
            # divorced
            divorced0 = cluster0[cluster0['marital_Divorced'] == 1]
            divorced_wine0 = divorced0['MntWines'].sum()
            divorced_fruit0 = divorced0 ['MntFruits'].sum()
            divorced_meat0 = divorced0 ['MntMeatProducts'].sum()
            divorced_fish0 = divorced0 ['MntFishProducts'].sum()
            divorced_sweet0 = divorced0 ['MntSweetProducts'].sum()
            divorced_gold0 = divorced0 ['MntGoldProds'].sum()
            divorced_regular0 = divorced0 ['MntRegularProds'].sum()
            data = {'MaritalStatus' : ['Divorced', 'Divorced', 'Divorced', 'Divorced', 'Divorced', 'Divorced', 'Divorced'],
                    'ProductCategory' : ['Wine', 'Fruit', 'Meat', 'Fish', 'Sweets', 'Gold', 'Regular Products'],
                    'AmountSpent' : [divorced_wine0, divorced_fruit0, divorced_meat0, divorced_fish0, divorced_sweet0, divorced_gold0, divorced_regular0]}
            divorced0 = pd.DataFrame(data)

            # married
            married0 = cluster0[cluster0['marital_Married'] == 1]
            married_wine0 = married0['MntWines'].sum()
            married_fruit0 = married0 ['MntFruits'].sum()
            married_meat0 = married0 ['MntMeatProducts'].sum()
            married_fish0 = married0 ['MntFishProducts'].sum()
            married_sweet0 = married0 ['MntSweetProducts'].sum()
            married_gold0 = married0 ['MntGoldProds'].sum()
            married_regular0 = married0 ['MntRegularProds'].sum()
            data = {'MaritalStatus' : ['Married', 'Married', 'Married', 'Married', 'Married', 'Married', 'Married'],
                    'ProductCategory' : ['Wine', 'Fruit', 'Meat', 'Fish', 'Sweets', 'Gold', 'Regular Products'],
                    'AmountSpent' : [married_wine0, married_fruit0, married_meat0, married_fish0, married_sweet0, married_gold0, married_regular0]}
            married0 = pd.DataFrame(data)

            # single
            single0 = cluster0[cluster0['marital_Single'] == 1]
            single_wine0 = single0['MntWines'].sum()
            single_fruit0 = single0 ['MntFruits'].sum()
            single_meat0 = single0 ['MntMeatProducts'].sum()
            single_fish0 = single0 ['MntFishProducts'].sum()
            single_sweet0 = single0 ['MntSweetProducts'].sum()
            single_gold0 = single0 ['MntGoldProds'].sum()
            single_regular0 = single0 ['MntRegularProds'].sum()
            data = {'MaritalStatus' : ['Single', 'Single', 'Single', 'Single', 'Single', 'Single', 'Single'],
                    'ProductCategory' : ['Wine', 'Fruit', 'Meat', 'Fish', 'Sweets', 'Gold', 'Regular Products'],
                    'AmountSpent' : [single_wine0, single_fruit0, single_meat0, single_fish0, single_sweet0, single_gold0, single_regular0]}
            single0 = pd.DataFrame(data)

            # together
            together0 = cluster0[cluster0['marital_Together'] == 1]
            together_wine0 = together0['MntWines'].sum()
            together_fruit0 = together0 ['MntFruits'].sum()
            together_meat0 = together0 ['MntMeatProducts'].sum()
            together_fish0 = together0 ['MntFishProducts'].sum()
            together_sweet0 = together0 ['MntSweetProducts'].sum()
            together_gold0 = together0 ['MntGoldProds'].sum()
            together_regular0 = together0 ['MntRegularProds'].sum()
            data = {'MaritalStatus' : ['Together', 'Together', 'Together', 'Together', 'Together', 'Together', 'Together'],
                    'ProductCategory' : ['Wine', 'Fruit', 'Meat', 'Fish', 'Sweets', 'Gold', 'Regular Products'],
                    'AmountSpent' : [together_wine0, together_fruit0, together_meat0, together_fish0, together_sweet0, together_gold0, together_regular0]}
            together0 = pd.DataFrame(data)

            #widow
            widow0 = cluster0[cluster0['marital_Widow'] == 1]
            widow_wine0 = widow0['MntWines'].sum()
            widow_fruit0 = widow0 ['MntFruits'].sum()
            widow_meat0 = widow0 ['MntMeatProducts'].sum()
            widow_fish0 = widow0 ['MntFishProducts'].sum()
            widow_sweet0 = widow0 ['MntSweetProducts'].sum()
            widow_gold0 = widow0 ['MntGoldProds'].sum()
            widow_regular0 = widow0 ['MntRegularProds'].sum()
            data = {'MaritalStatus' : ['Widow', 'Widow', 'Widow', 'Widow', 'Widow', 'Widow', 'Widow'],
                    'ProductCategory' : ['Wine', 'Fruit', 'Meat', 'Fish', 'Sweets', 'Gold', 'Regular Products'],
                    'AmountSpent' : [widow_wine0, widow_fruit0, widow_meat0, widow_fish0, widow_sweet0, widow_gold0, widow_regular0]}
            widow0 = pd.DataFrame(data)

            # Combining all the dataframes
            frames = [divorced0, married0, single0, together0, widow0]
            result = pd.concat(frames)

            # Plotting the stacked Barchart
            fig5 = px.bar(result, x='MaritalStatus', y= 'AmountSpent', color = 'ProductCategory',
                            color_discrete_sequence = ['#8dbdff', '#ffb0d9', '#f7e4f8','#b3d4ff',
                                                        '#f7e4f8','#ffb0d9','#d6eaff'])

            # Remove gridlines
            fig5.update_layout(xaxis=dict(showgrid=False),
                          yaxis=dict(showgrid=False)
            )

            # Aligning title
            fig5.update_layout(
                title={
                    'text': "Spending by Marital Status",
                    'x':0.48,
                    'y':0.99,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            ### Horizontal bar chart for segment 0 ###
            # Interaction with the 5 campaigns

            # Step 1- Getting the values
            df0_total_cmp1 = cluster0['AcceptedCmp1'].sum()
            df0_total_cmp2 = cluster0['AcceptedCmp2'].sum()
            df0_total_cmp3 = cluster0['AcceptedCmp3'].sum()
            df0_total_cmp4 = cluster0['AcceptedCmp4'].sum()
            df0_total_cmp5 = cluster0['AcceptedCmp5'].sum()

            # Step2 - DataFrame
            data_0 = {'Segment' : [name0, name0, name0, name0, name0],
                    'Campaign' : ['Campaign1', 'Campaign2', 'Campaign3', 'Campaign4', 'Campaign5'],
                    'Count' : [df0_total_cmp1, df0_total_cmp2, df0_total_cmp3, df0_total_cmp4, df0_total_cmp5]}

            df_0 = pd.DataFrame(data_0)

            fig1 = go.Figure(go.Bar(
                                    x = data_0['Count'],
                                    y = data_0['Campaign'],
                                    marker_color = ['#ffb0d9', '#f9c9e2', '#8dbdff','#b3d4ff', '#d6eaff'],
                                    orientation = 'h'))
            # Remove gridlines
            fig1.update_layout(xaxis=dict(showgrid=False),
                          yaxis=dict(showgrid=False))

            # Aligning title
            fig1.update_layout(
                title={
                    'text': "Response to Previous Campaigns",
                    'x':0.48,
                    #'y':0.99,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            ### Scatterplot ###
            # Tenure and Spending
            # changing customer days from continuous to discrete
            cluster0 ['Tenure'] = round(cluster0['Customer_Days']/365)
            cluster0 ['Tenure'] = cluster0 ['Tenure'].astype('int64')

            years = cluster0 ['Tenure'].unique().tolist()

            count = cluster0 ['Tenure'].value_counts().tolist()

            # money spent by tenure = 6
            tenure6 = cluster0 [cluster0 ['Tenure'] == 6]
            monetary6 = mean(tenure6['Monetary'])

            # money spent by tenure = 7
            tenure7 = cluster0 [cluster0 ['Tenure'] == 7]
            monetary7 = mean(tenure7['Monetary'])

            # money spent by tenure = 8
            tenure8 = cluster0 [cluster0 ['Tenure'] == 8]
            monetary8 = mean(tenure8['Monetary'])

            data = {'Tenure':years,
                    'count':count,
                    'Monetary':[monetary8, monetary7, monetary6],
                    'segment':['Segment 0', 'Segment 0', 'Segment 0']}
            df_tenure = pd.DataFrame(data)

            fig3 = px.scatter(df_tenure, x="Tenure", y="Monetary",
            	         size="count", hover_name="segment", log_x=True, size_max=60,
                         color = 'segment', color_discrete_sequence = ["#ffb0d9"])
            # Remove gridlines
            fig3.update_layout(xaxis=dict(showgrid=False),
                          yaxis=dict(showgrid=False))

            # Aligning title
            fig3.update_layout(
                title={
                    'text': "Spending by Tenure",
                    'x':0.48,
                    #'y':0.99,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            # Plotting the charts
            col1, col2, col3 = st.columns(3)
            with col1:
                st.plotly_chart(fig1)
                st.plotly_chart(fig4)
            with col2:
                st.plotly_chart(fig2)
                st.plotly_chart(fig5)
            with col3:
                st.plotly_chart(fig3)
                st.plotly_chart(fig6)

        if option == 'Segment 1':
            ### Treemap for segment 1 ###
            # Step 1- Subsetting the df for Cluster 1
            cluster1 = df_X[df_X['Segment'] == 'Segment 1']

            # Step 2- Calculating the amount spent ($) for each product category
            wine1 = cluster1['MntWines'].sum()
            fruit1 = cluster1['MntFruits'].sum()
            meat1 = cluster1['MntMeatProducts'].sum()
            fish1 = cluster1['MntFishProducts'].sum()
            sweet1 = cluster1['MntSweetProducts'].sum()
            gold1 = cluster1['MntGoldProds'].sum()
            regular1 = cluster1['MntRegularProds'].sum()

            # Getting the name of the cluster
            name1 = cluster1.iloc[0,-1]

            # Step 3- df_1
            data = {'Segment' : [name1, name1, name1, name1, name1, name1, name1],
                   'ProductCategory' : ['Wine', 'Fruit', 'Meat', 'Fish', 'Sweets', 'Gold', 'Regular Products'],
                   'Monetary' : [wine1, fruit1, meat1, fish1, sweet1, gold1, regular1]}
            df_1 = pd.DataFrame(data)

            # Step 4- Plot Treemap
            fig2 = px.treemap(df_1,
                            names = 'ProductCategory',
                             values = 'Monetary',
                             parents = 'Segment',
                             color = 'ProductCategory',
                             color_discrete_sequence = ['#f9c9e2', '#d6eaff',
                             '#f7e4f8', '#8dbdff', '#8dbdff', '#ffb0d9', '#b3d4ff'])
            # Aligning title
            fig2.update_layout(
                title={
                    'text': "Spending by Product Category",
                    'x':0.48,
                    #'y':0.99,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            ### Pie chart for segment 1 ###
            # Step 1- Subsetting the df for Cluster 1
            pie1 = cluster1[cluster1['Segment'] == 'Segment 1']

             # Step 2- Calculating the sum for each channel
            web1 = pie1['NumWebPurchases'].sum()
            store1 = pie1['NumStorePurchases'].sum()
            catalog1 = pie1['NumCatalogPurchases'].sum()

            # Getting the name of the cluster
            name1 = cluster1.iloc[0,-1]

            # Step 3- df_1
            data = {'Segment' : [name1, name1, name1],
                   'Channel' : ['Website', 'In-Store', 'Catalog'],
                   'Count' : [web1, store1, catalog1]}
            df_1 = pd.DataFrame(data)

            # Step 4- Plot pie
            fig6 = px.pie(df_1, values = 'Count', names = 'Channel', color = 'Channel',
            color_discrete_sequence = ['#8dbdff', '#b3d4ff', '#d6eaff'])

            # Aligning title
            fig6.update_layout(
                title={
                    'text': "Shopping Channels",
                    'y':0.99,
                    'x':0.49,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            ### Scatterplot for segment 1 ###
            # Relationship between web visits and deals purchased
            fig4 = px.scatter(cluster1, x='NumWebVisitsMonth', y='NumDealsPurchases',
                                trendline="ols", color_discrete_sequence = ['#ffb0d9'])

            # Remove gridlines
            fig4.update_layout(xaxis=dict(showgrid=False),
                            yaxis=dict(showgrid=False))

            # Remove gridlines
            fig4.update_layout(xaxis=dict(showgrid=False),
                            yaxis=dict(showgrid=False))

            # Aligning title
            fig4.update_layout(
                title={
                    'text': "Deals Purchased by Monthly Webiste Visits",
                    'x':0.48,
                    'y':0.99,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            ### Stacked Barchart for segment 1 ###
            # Amount spent on each product category by each marital status of cluster 2
            # divorced
            divorced1 = cluster1[cluster1['marital_Divorced'] == 1]
            divorced_wine1 = divorced1['MntWines'].sum()
            divorced_fruit1 = divorced1 ['MntFruits'].sum()
            divorced_meat1 = divorced1 ['MntMeatProducts'].sum()
            divorced_fish1 = divorced1 ['MntFishProducts'].sum()
            divorced_sweet1 = divorced1 ['MntSweetProducts'].sum()
            divorced_gold1 = divorced1 ['MntGoldProds'].sum()
            divorced_regular1 = divorced1 ['MntRegularProds'].sum()
            data = {'MaritalStatus' : ['Divorced', 'Divorced', 'Divorced', 'Divorced', 'Divorced', 'Divorced', 'Divorced'],
                    'ProductCategory' : ['Wine', 'Fruit', 'Meat', 'Fish', 'Sweets', 'Gold', 'Regular Products'],
                    'AmountSpent' : [divorced_wine1, divorced_fruit1, divorced_meat1, divorced_fish1, divorced_sweet1, divorced_gold1, divorced_regular1]}
            divorced1 = pd.DataFrame(data)

            # married
            married1 = cluster1[cluster1['marital_Married'] == 1]
            married_wine1 = married1['MntWines'].sum()
            married_fruit1 = married1 ['MntFruits'].sum()
            married_meat1 = married1 ['MntMeatProducts'].sum()
            married_fish1 = married1 ['MntFishProducts'].sum()
            married_sweet1 = married1 ['MntSweetProducts'].sum()
            married_gold1 = married1 ['MntGoldProds'].sum()
            married_regular1 = married1 ['MntRegularProds'].sum()
            data = {'MaritalStatus' : ['Married', 'Married', 'Married', 'Married', 'Married', 'Married', 'Married'],
                    'ProductCategory' : ['Wine', 'Fruit', 'Meat', 'Fish', 'Sweets', 'Gold', 'Regular Products'],
                    'AmountSpent' : [married_wine1, married_fruit1, married_meat1, married_fish1, married_sweet1, married_gold1, married_regular1]}
            married1 = pd.DataFrame(data)

            # single
            single1 = cluster1[cluster1['marital_Single'] == 1]
            single_wine1 = single1['MntWines'].sum()
            single_fruit1 = single1 ['MntFruits'].sum()
            single_meat1 = single1 ['MntMeatProducts'].sum()
            single_fish1 = single1 ['MntFishProducts'].sum()
            single_sweet1 = single1 ['MntSweetProducts'].sum()
            single_gold1 = single1 ['MntGoldProds'].sum()
            single_regular1 = single1 ['MntRegularProds'].sum()
            data = {'MaritalStatus' : ['Single', 'Single', 'Single', 'Single', 'Single', 'Single', 'Single'],
                    'ProductCategory' : ['Wine', 'Fruit', 'Meat', 'Fish', 'Sweets', 'Gold', 'Regular Products'],
                    'AmountSpent' : [single_wine1, single_fruit1, single_meat1, single_fish1, single_sweet1, single_gold1, single_regular1]}
            single1 = pd.DataFrame(data)

            # together
            together1 = cluster1[cluster1['marital_Together'] == 1]
            together_wine1 = together1['MntWines'].sum()
            together_fruit1 = together1 ['MntFruits'].sum()
            together_meat1 = together1 ['MntMeatProducts'].sum()
            together_fish1 = together1 ['MntFishProducts'].sum()
            together_sweet1 = together1 ['MntSweetProducts'].sum()
            together_gold1 = together1 ['MntGoldProds'].sum()
            together_regular1 = together1 ['MntRegularProds'].sum()
            data = {'MaritalStatus' : ['Together', 'Together', 'Together', 'Together', 'Together', 'Together', 'Together'],
                    'ProductCategory' : ['Wine', 'Fruit', 'Meat', 'Fish', 'Sweets', 'Gold', 'Regular Products'],
                    'AmountSpent' : [together_wine1, together_fruit1, together_meat1, together_fish1, together_sweet1, together_gold1, together_regular1]}
            together1 = pd.DataFrame(data)

            #widow
            widow1 = cluster1[cluster1['marital_Widow'] == 1]
            widow_wine1 = widow1['MntWines'].sum()
            widow_fruit1 = widow1 ['MntFruits'].sum()
            widow_meat1 = widow1 ['MntMeatProducts'].sum()
            widow_fish1 = widow1 ['MntFishProducts'].sum()
            widow_sweet1 = widow1 ['MntSweetProducts'].sum()
            widow_gold1 = widow1 ['MntGoldProds'].sum()
            widow_regular1 = widow1 ['MntRegularProds'].sum()
            data = {'MaritalStatus' : ['Widow', 'Widow', 'Widow', 'Widow', 'Widow', 'Widow', 'Widow'],
                    'ProductCategory' : ['Wine', 'Fruit', 'Meat', 'Fish', 'Sweets', 'Gold', 'Regular Products'],
                    'AmountSpent' : [widow_wine1, widow_fruit1, widow_meat1, widow_fish1, widow_sweet1, widow_gold1, widow_regular1]}
            widow1 = pd.DataFrame(data)

            # Combining all the dataframes
            frames = [divorced1, married1, single1, together1, widow1]
            result = pd.concat(frames)

            # Plotting the stacked Barchart
            fig5 = px.bar(result, x='MaritalStatus', y= 'AmountSpent', color = 'ProductCategory',
                        color_discrete_sequence = ['#8dbdff', '#ffb0d9', '#f7e4f8','#b3d4ff','#f7e4f8','#ffb0d9','#d6eaff'])

            # Remove gridlines
            fig5.update_layout(xaxis=dict(showgrid=False),
                          yaxis=dict(showgrid=False)
            )

            # Aligning title
            fig5.update_layout(
                title={
                    'text': "Spending by Marital Status",
                    'x':0.48,
                    'y':0.99,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            ### Horizontal bar chart for segment 1 ###
            # Interaction with the 5 campaigns

            # Step 1- Getting the values
            df1_total_cmp1 = cluster1['AcceptedCmp1'].sum()
            df1_total_cmp2 = cluster1['AcceptedCmp2'].sum()
            df1_total_cmp3 = cluster1['AcceptedCmp3'].sum()
            df1_total_cmp4 = cluster1['AcceptedCmp4'].sum()
            df1_total_cmp5 = cluster1['AcceptedCmp5'].sum()

            # Step2 - DataFrame
            data_1 = {'Segment' : [name1, name1, name1, name1, name1],
                    'Campaign' : ['Campaign1', 'Campaign2', 'Campaign3', 'Campaign4', 'Campaign5'],
                    'Count' : [df1_total_cmp1, df1_total_cmp2, df1_total_cmp3, df1_total_cmp4, df1_total_cmp5]}

            df_1 = pd.DataFrame(data_1)

            fig1 = go.Figure(go.Bar(
                                    x = data_1['Count'],
                                    y = data_1['Campaign'],
                                    marker_color = ['#ffb0d9', '#f9c9e2', '#8dbdff','#b3d4ff', '#d6eaff'],
                                    orientation = 'h'))
            # Remove gridlines
            fig1.update_layout(xaxis=dict(showgrid=False),
                          yaxis=dict(showgrid=False))

            # Aligning title
            fig1.update_layout(
                title={
                    'text': "Response to Previous Campaigns",
                    'x':0.48,
                    #'y':0.99,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            ### Scatterplot ###
            # Tenure and Spending
            # changing customer days from continuous to discrete
            cluster1['Tenure'] = round(cluster1 ['Customer_Days']/365)
            cluster1 ['Tenure'] = cluster1 ['Tenure'].astype('int64')

            years = cluster1 ['Tenure'].unique().tolist()

            count = cluster1 ['Tenure'].value_counts().tolist()

            # money spent by tenure = 6
            tenure6 = cluster1 [cluster1 ['Tenure'] == 6]
            monetary6 = mean(tenure6['Monetary'])

            # money spent by tenure = 7
            tenure7 = cluster1 [cluster1 ['Tenure'] == 7]
            monetary7 = mean(tenure7['Monetary'])

            # money spent by tenure = 8
            tenure8 = cluster1 [cluster1 ['Tenure'] == 8]
            monetary8 = mean(tenure8['Monetary'])

            data = {'Tenure':years,
                    'count':count,
                    'Monetary':[monetary8, monetary7, monetary6],
                    'segment':['Segment 1', 'Segment 1', 'Segment 1']}
            df_tenure = pd.DataFrame(data)

            fig3 = px.scatter(df_tenure, x="Tenure", y="Monetary",
            	         size="count", hover_name="segment", log_x=True, size_max=60,
                         color = 'segment', color_discrete_sequence = ["#ffb0d9"])
            # Remove gridlines
            fig3.update_layout(xaxis=dict(showgrid=False),
                          yaxis=dict(showgrid=False))

            # Aligning title
            fig3.update_layout(
                title={
                    'text': "Spending by Tenure",
                    'x':0.48,
                    #'y':0.99,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            # Plotting the charts
            col1, col2, col3 = st.columns(3)
            with col1:
                st.plotly_chart(fig1)
                st.plotly_chart(fig4)
            with col2:
                st.plotly_chart(fig2)
                st.plotly_chart(fig5)
            with col3:
                st.plotly_chart(fig3)
                st.plotly_chart(fig6)

        if option == 'Segment 2':
            ### Treemap for segment 2 ###
            # Step 1- Subsetting the df for Cluster 2
            cluster2 = df_X[df_X['Segment'] == 'Segment 2']

            # Step 2- Calculating the amount spent ($) for each product category
            wine2 = cluster2['MntWines'].sum()
            fruit2 = cluster2['MntFruits'].sum()
            meat2 = cluster2['MntMeatProducts'].sum()
            fish2 = cluster2['MntFishProducts'].sum()
            sweet2 = cluster2['MntSweetProducts'].sum()
            gold2 = cluster2['MntGoldProds'].sum()
            regular2 = cluster2['MntRegularProds'].sum()

            # Getting the name of the cluster
            name2 = cluster2.iloc[0,-1]

            # Step 3- df_2
            data = {'Segment' : [name2, name2, name2, name2, name2, name2, name2],
                   'ProductCategory' : ['Wine', 'Fruit', 'Meat', 'Fish', 'Sweets', 'Gold', 'Regular Products'],
                   'Monetary' : [wine2, fruit2, meat2, fish2, sweet2, gold2, regular2]}
            df_2 = pd.DataFrame(data)

            # Step 4- Plot Treemap
            fig2 = px.treemap(df_2,
                            names = 'ProductCategory',
                             values = 'Monetary',
                             parents = 'Segment',
                             color = 'ProductCategory',
                             color_discrete_sequence = ['#f9c9e2', '#d6eaff',
                             '#f7e4f8', '#8dbdff', '#8dbdff', '#ffb0d9', '#b3d4ff'])
            # Aligning title
            fig2.update_layout(
                title={
                    'text': "Spending by Product Category",
                    'x':0.48,
                    #'y':0.99,
                    'xanchor': 'center',
                    'yanchor': 'top'})

        ### Pie chart for segment 2 ###
            # Step 1- Subsetting the df for Cluster 2
            pie2 = cluster2[cluster2['Segment'] == 'Segment 2']

            # Step 2- Calculating the sum for each channel
            web2 = pie2['NumWebPurchases'].sum()
            store2 = pie2['NumStorePurchases'].sum()
            catalog2 = pie2['NumCatalogPurchases'].sum()

            # Getting the name of the cluster
            name2 = cluster2.iloc[0,-1]

            # Step 3- df_2
            data = {'Segment' : [name2, name2, name2],
                   'Channel' : ['Website', 'In-Store', 'Catalog'],
                   'Count' : [web2, store2, catalog2]}
            df_2 = pd.DataFrame(data)

            # Step 4- Plot pie
            fig6 = px.pie(df_2, values = 'Count', names = 'Channel', color = 'Channel',
            color_discrete_sequence = ['#8dbdff', '#b3d4ff', '#d6eaff'])

            # Aligning title
            fig6.update_layout(
                title={
                    'text': "Shopping Channels",
                    'y':0.99,
                    'x':0.49,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            ### Scatterplot for segment 2 ###
            # Relationship between web visits and deals purchased
            fig4 = px.scatter(cluster2, x='NumWebVisitsMonth', y='NumDealsPurchases', trendline="ols",
                                color_discrete_sequence = ['#ffb0d9'])

            # Remove gridlines
            fig4.update_layout(xaxis=dict(showgrid=False),
                            yaxis=dict(showgrid=False))
            # Remove gridlines
            fig4.update_layout(xaxis=dict(showgrid=False),
                            yaxis=dict(showgrid=False))

            # Aligning title
            fig4.update_layout(
                title={
                    'text': "Deals Purchased by Monthly Webiste Visits",
                    'x':0.48,
                    'y':0.99,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            ### Stacked Barchart for segment 2 ###
            # Amount spent on each product category by each marital status of cluster 2
            # divorced
            divorced2 = cluster2[cluster2['marital_Divorced'] == 1]
            divorced_wine2 = divorced2['MntWines'].sum()
            divorced_fruit2 = divorced2['MntFruits'].sum()
            divorced_meat2 = divorced2['MntMeatProducts'].sum()
            divorced_fish2 = divorced2['MntFishProducts'].sum()
            divorced_sweet2 = divorced2['MntSweetProducts'].sum()
            divorced_gold2 = divorced2['MntGoldProds'].sum()
            divorced_regular2 = divorced2['MntRegularProds'].sum()
            data = {'MaritalStatus' : ['Divorced', 'Divorced', 'Divorced', 'Divorced', 'Divorced', 'Divorced', 'Divorced'],
                    'ProductCategory' : ['Wine', 'Fruit', 'Meat', 'Fish', 'Sweets', 'Gold', 'Regular Products'],
                    'AmountSpent' : [divorced_wine2, divorced_fruit2, divorced_meat2, divorced_fish2, divorced_sweet2, divorced_gold2, divorced_regular2]}
            divorced2 = pd.DataFrame(data)

            # married
            married2 = cluster2[cluster2['marital_Married'] == 1]
            married_wine2 = married2['MntWines'].sum()
            married_fruit2 = married2['MntFruits'].sum()
            married_meat2 = married2['MntMeatProducts'].sum()
            married_fish2 = married2['MntFishProducts'].sum()
            married_sweet2 = married2['MntSweetProducts'].sum()
            married_gold2 = married2['MntGoldProds'].sum()
            married_regular2 = married2['MntRegularProds'].sum()
            data = {'MaritalStatus' : ['Married', 'Married', 'Married', 'Married', 'Married', 'Married', 'Married'],
                    'ProductCategory' : ['Wine', 'Fruit', 'Meat', 'Fish', 'Sweets', 'Gold', 'Regular Products'],
                    'AmountSpent' : [married_wine2, married_fruit2, married_meat2, married_fish2, married_sweet2, married_gold2, married_regular2]}
            married2 = pd.DataFrame(data)

            # single
            single2 = cluster2[cluster2['marital_Single'] == 1]
            single_wine2 = single2['MntWines'].sum()
            single_fruit2 = single2['MntFruits'].sum()
            single_meat2 = single2['MntMeatProducts'].sum()
            single_fish2 = single2['MntFishProducts'].sum()
            single_sweet2 = single2['MntSweetProducts'].sum()
            single_gold2 = single2['MntGoldProds'].sum()
            single_regular2 = single2['MntRegularProds'].sum()
            data = {'MaritalStatus' : ['Single', 'Single', 'Single', 'Single', 'Single', 'Single', 'Single'],
                    'ProductCategory' : ['Wine', 'Fruit', 'Meat', 'Fish', 'Sweets', 'Gold', 'Regular Products'],
                    'AmountSpent' : [single_wine2, single_fruit2, single_meat2, single_fish2, single_sweet2, single_gold2, single_regular2]}
            single2 = pd.DataFrame(data)

            # together
            together2 = cluster2[cluster2['marital_Together'] == 1]
            together_wine2 = together2['MntWines'].sum()
            together_fruit2 = together2['MntFruits'].sum()
            together_meat2 = together2['MntMeatProducts'].sum()
            together_fish2 = together2['MntFishProducts'].sum()
            together_sweet2 = together2['MntSweetProducts'].sum()
            together_gold2 = together2['MntGoldProds'].sum()
            together_regular2 = together2['MntRegularProds'].sum()
            data = {'MaritalStatus' : ['Together', 'Together', 'Together', 'Together', 'Together', 'Together', 'Together'],
                    'ProductCategory' : ['Wine', 'Fruit', 'Meat', 'Fish', 'Sweets', 'Gold', 'Regular Products'],
                    'AmountSpent' : [together_wine2, together_fruit2, together_meat2, together_fish2, together_sweet2, together_gold2, together_regular2]}
            together2 = pd.DataFrame(data)

            #widow
            widow2 = cluster2[cluster2['marital_Widow'] == 1]
            widow_wine2 = widow2['MntWines'].sum()
            widow_fruit2 = widow2['MntFruits'].sum()
            widow_meat2 = widow2['MntMeatProducts'].sum()
            widow_fish2 = widow2['MntFishProducts'].sum()
            widow_sweet2 = widow2['MntSweetProducts'].sum()
            widow_gold2 = widow2['MntGoldProds'].sum()
            widow_regular2 = widow2['MntRegularProds'].sum()
            data = {'MaritalStatus' : ['Widow', 'Widow', 'Widow', 'Widow', 'Widow', 'Widow', 'Widow'],
                    'ProductCategory' : ['Wine', 'Fruit', 'Meat', 'Fish', 'Sweets', 'Gold', 'Regular Products'],
                    'AmountSpent' : [widow_wine2, widow_fruit2, widow_meat2, widow_fish2, widow_sweet2, widow_gold2, widow_regular2]}
            widow2 = pd.DataFrame(data)

            # Combining all the dataframes
            frames = [divorced2, married2, single2, together2, widow2]
            result = pd.concat(frames)

            # Plotting the stacked Barchart
            fig5 = px.bar(result, x='MaritalStatus', y= 'AmountSpent', color = 'ProductCategory',
            color_discrete_sequence = ['#8dbdff', '#ffb0d9', '#f7e4f8','#b3d4ff','#f7e4f8','#ffb0d9','#d6eaff'])

            # Remove gridlines
            fig5.update_layout(xaxis=dict(showgrid=False),
                          yaxis=dict(showgrid=False)
            )
            # Aligning title
            fig5.update_layout(
                title={
                    'text': "Spending by Marital Status",
                    'x':0.48,
                    'y':0.99,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            ### Horizontal bar chart for segment 2 ###
            # Interaction with the 5 campaigns

            # Step 1- Getting the values
            df2_total_cmp1 = cluster2['AcceptedCmp1'].sum()
            df2_total_cmp2 = cluster2 ['AcceptedCmp2'].sum()
            df2_total_cmp3 = cluster2 ['AcceptedCmp3'].sum()
            df2_total_cmp4 = cluster2 ['AcceptedCmp4'].sum()
            df2_total_cmp5 = cluster2 ['AcceptedCmp5'].sum()

            # Step2 - DataFrame
            data_2 = {'Segment' : [name2, name2, name2, name2, name2],
                    'Campaign' : ['Campaign1', 'Campaign2', 'Campaign3', 'Campaign4', 'Campaign5'],
                    'Count' : [df2_total_cmp1, df2_total_cmp2, df2_total_cmp3, df2_total_cmp4, df2_total_cmp5]}

            df_2 = pd.DataFrame(data_2)

            fig1 = go.Figure(go.Bar(
                                    x = data_2['Count'],
                                    y = data_2['Campaign'],
                                    marker_color = ['#ffb0d9', '#f9c9e2',
                                    '#8dbdff','#b3d4ff', '#d6eaff'],
                                    orientation = 'h'))
            # Remove gridlines
            fig1.update_layout(xaxis=dict(showgrid=False),
                          yaxis=dict(showgrid=False))
            # Aligning title
            fig1.update_layout(
                title={
                    'text': "Response to Previous Campaigns",
                    'x':0.48,
                    #'y':0.99,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            ### Scatterplot ###
            # Tenure and Spending
            # changing customer days from continuous to discrete
            cluster2['Tenure'] = round(cluster2['Customer_Days']/365)
            cluster2['Tenure'] = cluster2['Tenure'].astype('int64')

            years = cluster2['Tenure'].unique().tolist()
            count = cluster2['Tenure'].value_counts().tolist()

            # money spent by tenure = 6
            tenure6 = cluster2[cluster2['Tenure'] == 6]
            monetary6 = mean(tenure6['Monetary'])

            # money spent by tenure = 7
            tenure7 = cluster2[cluster2['Tenure'] == 7]
            monetary7 = mean(tenure7['Monetary'])

            # money spent by tenure = 8
            tenure8 = cluster2[cluster2['Tenure'] == 8]
            monetary8 = mean(tenure8['Monetary'])

            data = {'Tenure':years,
                    'count':count,
                    'Monetary':[monetary8, monetary7, monetary6],
                    'segment':['Segment 2', 'Segment 2', 'Segment 2']}
            df_tenure = pd.DataFrame(data)

            fig3 = px.scatter(df_tenure, x="Tenure", y="Monetary",
            	         size="count", hover_name="segment", log_x=True, size_max=60,
                         color = 'segment', color_discrete_sequence = ["#ffb0d9"])
            # Remove gridlines
            fig3.update_layout(xaxis=dict(showgrid=False),
                          yaxis=dict(showgrid=False))
            # Aligning title
            fig3.update_layout(
                title={
                    'text': "Spending by Tenure",
                    'x':0.48,
                    #'y':0.99,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            # Plotting the charts
            col1, col2, col3 = st.columns(3)
            with col1:
                st.plotly_chart(fig1)
                st.plotly_chart(fig4)
            with col2:
                st.plotly_chart(fig2)
                st.plotly_chart(fig5)
            with col3:
                st.plotly_chart(fig3)
                st.plotly_chart(fig6)

    # Our dataset
    else:
        data = pd.read_csv('ifood_df.csv')

        # Adding 2 new columns
        data['Monetary'] = data['MntWines'] + data['MntFruits'] + data['MntMeatProducts'] + data['MntFishProducts'] + data['MntSweetProducts'] + data['MntGoldProds'] + data['MntRegularProds']
        data['Frequency'] = data['NumDealsPurchases'] + data['NumWebPurchases'] + data['NumCatalogPurchases'] + data['NumStorePurchases']

        ### Calculate Recency, Frequency, Monetary ###
        # Subsetting the dataframe to only include the features: 'Recency', 'Frequency', 'Monetary'
        df_RFM = data.copy()
        columns = ['Recency', 'Frequency', 'Monetary']
        df_RFM = df_RFM[columns]

        # Choose number of clusters
        K = 3
        model = KMeans(n_clusters = K, random_state = 1)
        model.fit(df_RFM)
        df_RFM['Cluster'] = model.labels_

        # Change the cluster columns data type into discrete values
        df_RFM['Segment'] = 'Segment ' + df_RFM['Cluster'].astype('str')

        # Calculate average values for each cluster and return size of each
        df_agg = df_RFM.groupby('Segment').agg({
                                                'Recency' : 'mean',
                                                'Frequency' : 'mean',
                                                'Monetary' : ['mean', 'count']}).round(0)

        df_agg.columns = df_agg.columns.droplevel()
        df_agg.columns = ['RecencyMean', 'FrequencyMean', 'MonetaryMean', 'Count']
        df_agg['Percent'] = round((df_agg['Count']/df_agg.Count.sum())*100,2)

        # Reset the index
        df_agg = df_agg.reset_index()

        # Adding cluster label to each row in original dataframe
        data['Cluster'] = model.labels_
        data['Segment'] = 'Segment ' + data['Cluster'].astype('str')


        ######## Plotting graphs for each segment on a different page ########
        # USE data
        if option == 'Segment 0':
            ### Treemap for segment 0 ###
            # Step 1- Subsetting the df for Cluster 0
            cluster0 = data[data['Segment'] == 'Segment 0']

            # Step 2- Calculating the amount spent ($) for each product category
            wine0 = cluster0['MntWines'].sum()
            fruit0 = cluster0['MntFruits'].sum()
            meat0 = cluster0['MntMeatProducts'].sum()
            fish0 = cluster0['MntFishProducts'].sum()
            sweet0 = cluster0['MntSweetProducts'].sum()
            gold0 = cluster0['MntGoldProds'].sum()
            regular0 = cluster0['MntRegularProds'].sum()

            # Getting the name of the cluster
            name0 = cluster0.iloc[0,-1]

            # Step 3- df_0
            data = {'Segment' : [name0, name0, name0, name0, name0, name0, name0],
                   'ProductCategory' : ['Wine', 'Fruit', 'Meat', 'Fish', 'Sweets', 'Gold', 'Regular Products'],
                   'Monetary' : [wine0, fruit0, meat0, fish0, sweet0, gold0, regular0]}
            df_0 = pd.DataFrame(data)

            # Step 4- Plot Treemap
            fig2 = px.treemap(df_0,
                             names = 'ProductCategory',
                             values = 'Monetary',
                             parents = 'Segment',
                             color = 'ProductCategory',
                             color_discrete_sequence = ['#f9c9e2', '#d6eaff',
                         '#f7e4f8', '#8dbdff', '#8dbdff', '#ffb0d9', '#b3d4ff'])

            # Aligning title
            fig2.update_layout(
                title={
                    'text': "Spending by Product Category",
                    'x':0.48,
                    #'y':0.99,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            ### Pie chart for segment 0 ###
            # Step 1- Subsetting the df for Cluster 0
            pie0 = cluster0[cluster0['Segment'] == 'Segment 0']

            # Step 2- Calculating the sum for each channel
            web0 = pie0['NumWebPurchases'].sum()
            store0 = pie0['NumStorePurchases'].sum()
            catalog0 = pie0['NumCatalogPurchases'].sum()

            # Getting the name of the cluster
            name0 = cluster0.iloc[0,-1]

            # Step 3- df_0
            data = {'Segment' : [name0, name0, name0],
                   'Channel' : ['Website', 'In-Store', 'Catalog'],
                   'Count' : [web0, store0, catalog0]}
            df_0 = pd.DataFrame(data)

            # Step 4- Plot pie
            fig6 = px.pie(df_0, values = 'Count', names = 'Channel', color = 'Channel',
                        color_discrete_sequence = ['#8dbdff', '#b3d4ff', '#d6eaff'])

            # Aligning title
            fig6.update_layout(
                title={
                    'text': "Shopping Channels",
                    'y':0.99,
                    'x':0.49,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            ### Scatterplot for segment 0 ###
            # Relationship between web visits and deals purchased
            fig4 = px.scatter(cluster0, x='NumWebVisitsMonth', y='NumDealsPurchases',
                            trendline="ols", color_discrete_sequence = ['#ffb0d9'])

            # Remove gridlines
            fig4.update_layout(xaxis=dict(showgrid=False),
                            yaxis=dict(showgrid=False))

            # Aligning title
            fig4.update_layout(
                title={
                    'text': "Deals Purchased by Monthly Webiste Visits",
                    'x':0.48,
                    'y':0.99,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            ### Stacked Barchart for segment 0 ###
            # Amount spent on each product category by each marital status of cluster 0
            # divorced
            divorced0 = cluster0[cluster0['marital_Divorced'] == 1]
            divorced_wine0 = divorced0['MntWines'].sum()
            divorced_fruit0 = divorced0 ['MntFruits'].sum()
            divorced_meat0 = divorced0 ['MntMeatProducts'].sum()
            divorced_fish0 = divorced0 ['MntFishProducts'].sum()
            divorced_sweet0 = divorced0 ['MntSweetProducts'].sum()
            divorced_gold0 = divorced0 ['MntGoldProds'].sum()
            divorced_regular0 = divorced0 ['MntRegularProds'].sum()
            data = {'MaritalStatus' : ['Divorced', 'Divorced', 'Divorced', 'Divorced', 'Divorced', 'Divorced', 'Divorced'],
                    'ProductCategory' : ['Wine', 'Fruit', 'Meat', 'Fish', 'Sweets', 'Gold', 'Regular Products'],
                    'AmountSpent' : [divorced_wine0, divorced_fruit0, divorced_meat0, divorced_fish0, divorced_sweet0, divorced_gold0, divorced_regular0]}
            divorced0 = pd.DataFrame(data)

            # married
            married0 = cluster0[cluster0['marital_Married'] == 1]
            married_wine0 = married0['MntWines'].sum()
            married_fruit0 = married0 ['MntFruits'].sum()
            married_meat0 = married0 ['MntMeatProducts'].sum()
            married_fish0 = married0 ['MntFishProducts'].sum()
            married_sweet0 = married0 ['MntSweetProducts'].sum()
            married_gold0 = married0 ['MntGoldProds'].sum()
            married_regular0 = married0 ['MntRegularProds'].sum()
            data = {'MaritalStatus' : ['Married', 'Married', 'Married', 'Married', 'Married', 'Married', 'Married'],
                    'ProductCategory' : ['Wine', 'Fruit', 'Meat', 'Fish', 'Sweets', 'Gold', 'Regular Products'],
                    'AmountSpent' : [married_wine0, married_fruit0, married_meat0, married_fish0, married_sweet0, married_gold0, married_regular0]}
            married0 = pd.DataFrame(data)

            # single
            single0 = cluster0[cluster0['marital_Single'] == 1]
            single_wine0 = single0['MntWines'].sum()
            single_fruit0 = single0 ['MntFruits'].sum()
            single_meat0 = single0 ['MntMeatProducts'].sum()
            single_fish0 = single0 ['MntFishProducts'].sum()
            single_sweet0 = single0 ['MntSweetProducts'].sum()
            single_gold0 = single0 ['MntGoldProds'].sum()
            single_regular0 = single0 ['MntRegularProds'].sum()
            data = {'MaritalStatus' : ['Single', 'Single', 'Single', 'Single', 'Single', 'Single', 'Single'],
                    'ProductCategory' : ['Wine', 'Fruit', 'Meat', 'Fish', 'Sweets', 'Gold', 'Regular Products'],
                    'AmountSpent' : [single_wine0, single_fruit0, single_meat0, single_fish0, single_sweet0, single_gold0, single_regular0]}
            single0 = pd.DataFrame(data)

            # together
            together0 = cluster0[cluster0['marital_Together'] == 1]
            together_wine0 = together0['MntWines'].sum()
            together_fruit0 = together0 ['MntFruits'].sum()
            together_meat0 = together0 ['MntMeatProducts'].sum()
            together_fish0 = together0 ['MntFishProducts'].sum()
            together_sweet0 = together0 ['MntSweetProducts'].sum()
            together_gold0 = together0 ['MntGoldProds'].sum()
            together_regular0 = together0 ['MntRegularProds'].sum()
            data = {'MaritalStatus' : ['Together', 'Together', 'Together', 'Together', 'Together', 'Together', 'Together'],
                    'ProductCategory' : ['Wine', 'Fruit', 'Meat', 'Fish', 'Sweets', 'Gold', 'Regular Products'],
                    'AmountSpent' : [together_wine0, together_fruit0, together_meat0, together_fish0, together_sweet0, together_gold0, together_regular0]}
            together0 = pd.DataFrame(data)

            #widow
            widow0 = cluster0[cluster0['marital_Widow'] == 1]
            widow_wine0 = widow0['MntWines'].sum()
            widow_fruit0 = widow0 ['MntFruits'].sum()
            widow_meat0 = widow0 ['MntMeatProducts'].sum()
            widow_fish0 = widow0 ['MntFishProducts'].sum()
            widow_sweet0 = widow0 ['MntSweetProducts'].sum()
            widow_gold0 = widow0 ['MntGoldProds'].sum()
            widow_regular0 = widow0 ['MntRegularProds'].sum()
            data = {'MaritalStatus' : ['Widow', 'Widow', 'Widow', 'Widow', 'Widow', 'Widow', 'Widow'],
                    'ProductCategory' : ['Wine', 'Fruit', 'Meat', 'Fish', 'Sweets', 'Gold', 'Regular Products'],
                    'AmountSpent' : [widow_wine0, widow_fruit0, widow_meat0, widow_fish0, widow_sweet0, widow_gold0, widow_regular0]}
            widow0 = pd.DataFrame(data)

            # Combining all the dataframes
            frames = [divorced0, married0, single0, together0, widow0]
            result = pd.concat(frames)

            # Plotting the stacked Barchart
            fig5 = px.bar(result, x='MaritalStatus', y= 'AmountSpent', color = 'ProductCategory',
                        color_discrete_sequence = ['#8dbdff', '#ffb0d9', '#f7e4f8','#b3d4ff','#f7e4f8','#ffb0d9','#d6eaff'])

            # Remove gridlines
            fig5.update_layout(xaxis=dict(showgrid=False),
                          yaxis=dict(showgrid=False)
            )

            # Aligning title
            fig5.update_layout(
                title={
                    'text': "Spending by Marital Status",
                    'x':0.48,
                    'y':0.99,
                    'xanchor': 'center',
                    'yanchor': 'top'})


            ### Horizontal bar chart for segment 0 ###
            # Interaction with the 5 campaigns

            # Step 1- Getting the values
            df0_total_cmp1 = cluster0['AcceptedCmp1'].sum()
            df0_total_cmp2 = cluster0['AcceptedCmp2'].sum()
            df0_total_cmp3 = cluster0['AcceptedCmp3'].sum()
            df0_total_cmp4 = cluster0['AcceptedCmp4'].sum()
            df0_total_cmp5 = cluster0['AcceptedCmp5'].sum()

            # Step2 - DataFrame
            data_0 = {'Segment' : [name0, name0, name0, name0, name0],
                    'Campaign' : ['Campaign1', 'Campaign2', 'Campaign3', 'Campaign4', 'Campaign5'],
                    'Count' : [df0_total_cmp1, df0_total_cmp2, df0_total_cmp3, df0_total_cmp4, df0_total_cmp5]}

            df_0 = pd.DataFrame(data_0)
            fig1 = go.Figure(go.Bar(
                                    x = data_0['Count'],
                                    y = data_0['Campaign'],
                                    marker_color = ['#ffb0d9', '#f9c9e2', '#8dbdff','#b3d4ff', '#d6eaff'],
                                    orientation = 'h'))
            # Remove gridlines
            fig1.update_layout(xaxis=dict(showgrid=False),
                          yaxis=dict(showgrid=False))
            # Aligning title
            fig1.update_layout(
                title={
                    'text': "Response to Previous Campaigns",
                    'x':0.48,
                    #'y':0.99,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            ### Scatterplot ###
            # Tenure and Spending
            # changing customer days from continuous to discrete
            cluster0 ['Tenure'] = round(cluster0['Customer_Days']/365)
            cluster0 ['Tenure'] = cluster0 ['Tenure'].astype('int64')

            years = cluster0 ['Tenure'].unique().tolist()
            count = cluster0 ['Tenure'].value_counts().tolist()

            # money spent by tenure = 6
            tenure6 = cluster0 [cluster0 ['Tenure'] == 6]
            monetary6 = mean(tenure6['Monetary'])

            # money spent by tenure = 7
            tenure7 = cluster0 [cluster0 ['Tenure'] == 7]
            monetary7 = mean(tenure7['Monetary'])

            # money spent by tenure = 8
            tenure8 = cluster0 [cluster0 ['Tenure'] == 8]
            monetary8 = mean(tenure8['Monetary'])

            data = {'Tenure':years,
                    'count':count,
                    'Monetary':[monetary8, monetary7, monetary6],
                    'segment':['Segment 0', 'Segment 0', 'Segment 0']}
            df_tenure = pd.DataFrame(data)

            fig3 = px.scatter(df_tenure, x="Tenure", y="Monetary",
            	         size="count", hover_name="segment", log_x=True, size_max=60,
                         color = 'segment', color_discrete_sequence = ["#ffb0d9"])
            # Remove gridlines
            fig3.update_layout(xaxis=dict(showgrid=False),
                          yaxis=dict(showgrid=False))
            # Aligning title
            fig3.update_layout(
                title={
                    'text': "Spending by Tenure",
                    'x':0.48,
                    #'y':0.99,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            # Plotting the charts
            col1, col2, col3 = st.columns(3)
            with col1:
                st.plotly_chart(fig1)
                st.plotly_chart(fig4)
            with col2:
                st.plotly_chart(fig2)
                st.plotly_chart(fig5)
            with col3:
                st.plotly_chart(fig3)
                st.plotly_chart(fig6)

        if option == 'Segment 1':
            ### Treemap for segment 1 ###
            # Step 1- Subsetting the df for Cluster 1
            cluster1 = data[data['Segment'] == 'Segment 1']

            # Step 2- Calculating the amount spent ($) for each product category
            wine1 = cluster1['MntWines'].sum()
            fruit1 = cluster1['MntFruits'].sum()
            meat1 = cluster1['MntMeatProducts'].sum()
            fish1 = cluster1['MntFishProducts'].sum()
            sweet1 = cluster1['MntSweetProducts'].sum()
            gold1 = cluster1['MntGoldProds'].sum()
            regular1 = cluster1['MntRegularProds'].sum()

            # Getting the name of the cluster
            name1 = cluster1.iloc[0,-1]

            # Step 3- df_1
            data = {'Segment' : [name1, name1, name1, name1, name1, name1, name1],
                   'ProductCategory' : ['Wine', 'Fruit', 'Meat', 'Fish', 'Sweets', 'Gold', 'Regular Products'],
                   'Monetary' : [wine1, fruit1, meat1, fish1, sweet1, gold1, regular1]}
            df_1 = pd.DataFrame(data)

            # Step 4- Plot Treemap
            fig2 = px.treemap(df_1,
                            names = 'ProductCategory',
                             values = 'Monetary',
                             parents = 'Segment',
                             color = 'ProductCategory',
                             color_discrete_sequence = ['#f9c9e2', '#d6eaff',
                             '#f7e4f8', '#8dbdff', '#8dbdff', '#ffb0d9', '#b3d4ff'])
            # Aligning title
            fig2.update_layout(
                title={
                    'text': "Spending by Product Category",
                    'x':0.48,
                    #'y':0.99,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            ### Pie chart for segment 1 ###
            # Step 1- Subsetting the df for Cluster 1
            pie1 = cluster1[cluster1['Segment'] == 'Segment 1']

            # Step 2- Calculating the sum for each channel
            web1 = pie1['NumWebPurchases'].sum()
            store1 = pie1['NumStorePurchases'].sum()
            catalog1 = pie1['NumCatalogPurchases'].sum()

            # Getting the name of the cluster
            name1 = cluster1.iloc[0,-1]

            # Step 3- df_1
            data = {'Segment' : [name1, name1, name1],
                   'Channel' : ['Website', 'In-Store', 'Catalog'],
                   'Count' : [web1, store1, catalog1]}
            df_1 = pd.DataFrame(data)

            # Step 4- Plot pie
            fig6 = px.pie(df_1, values = 'Count', names = 'Channel', color = 'Channel',
            color_discrete_sequence = ['#8dbdff', '#b3d4ff', '#d6eaff'])

            # Aligning title
            fig6.update_layout(
                title={
                    'text': "Shopping Channels",
                    'y':0.99,
                    'x':0.49,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            ### Scatterplot for segment 1 ###
            # Relationship between web visits and deals purchased
            fig4 = px.scatter(cluster1, x='NumWebVisitsMonth', y='NumDealsPurchases',
                                trendline="ols", color_discrete_sequence = ['#ffb0d9'])

            # Remove gridlines
            fig4.update_layout(xaxis=dict(showgrid=False),
                            yaxis=dict(showgrid=False))
            # Aligning title
            fig4.update_layout(
                title={
                    'text': "Deals Purchased by Monthly Webiste Visits",
                    'x':0.48,
                    'y':0.99,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            ### Stacked Barchart for segment 1 ###
            # Amount spent on each product category by each marital status of cluster 2
            # divorced
            divorced1 = cluster1[cluster1['marital_Divorced'] == 1]
            divorced_wine1 = divorced1['MntWines'].sum()
            divorced_fruit1 = divorced1 ['MntFruits'].sum()
            divorced_meat1 = divorced1 ['MntMeatProducts'].sum()
            divorced_fish1 = divorced1 ['MntFishProducts'].sum()
            divorced_sweet1 = divorced1 ['MntSweetProducts'].sum()
            divorced_gold1 = divorced1 ['MntGoldProds'].sum()
            divorced_regular1 = divorced1 ['MntRegularProds'].sum()
            data = {'MaritalStatus' : ['Divorced', 'Divorced', 'Divorced', 'Divorced', 'Divorced', 'Divorced', 'Divorced'],
                    'ProductCategory' : ['Wine', 'Fruit', 'Meat', 'Fish', 'Sweets', 'Gold', 'Regular Products'],
                    'AmountSpent' : [divorced_wine1, divorced_fruit1, divorced_meat1, divorced_fish1, divorced_sweet1, divorced_gold1, divorced_regular1]}
            divorced1 = pd.DataFrame(data)

            # married
            married1 = cluster1[cluster1['marital_Married'] == 1]
            married_wine1 = married1['MntWines'].sum()
            married_fruit1 = married1 ['MntFruits'].sum()
            married_meat1 = married1 ['MntMeatProducts'].sum()
            married_fish1 = married1 ['MntFishProducts'].sum()
            married_sweet1 = married1 ['MntSweetProducts'].sum()
            married_gold1 = married1 ['MntGoldProds'].sum()
            married_regular1 = married1 ['MntRegularProds'].sum()
            data = {'MaritalStatus' : ['Married', 'Married', 'Married', 'Married', 'Married', 'Married', 'Married'],
                    'ProductCategory' : ['Wine', 'Fruit', 'Meat', 'Fish', 'Sweets', 'Gold', 'Regular Products'],
                    'AmountSpent' : [married_wine1, married_fruit1, married_meat1, married_fish1, married_sweet1, married_gold1, married_regular1]}
            married1 = pd.DataFrame(data)

            # single
            single1 = cluster1[cluster1['marital_Single'] == 1]
            single_wine1 = single1['MntWines'].sum()
            single_fruit1 = single1 ['MntFruits'].sum()
            single_meat1 = single1 ['MntMeatProducts'].sum()
            single_fish1 = single1 ['MntFishProducts'].sum()
            single_sweet1 = single1 ['MntSweetProducts'].sum()
            single_gold1 = single1 ['MntGoldProds'].sum()
            single_regular1 = single1 ['MntRegularProds'].sum()
            data = {'MaritalStatus' : ['Single', 'Single', 'Single', 'Single', 'Single', 'Single', 'Single'],
                    'ProductCategory' : ['Wine', 'Fruit', 'Meat', 'Fish', 'Sweets', 'Gold', 'Regular Products'],
                    'AmountSpent' : [single_wine1, single_fruit1, single_meat1, single_fish1, single_sweet1, single_gold1, single_regular1]}
            single1 = pd.DataFrame(data)

            # together
            together1 = cluster1[cluster1['marital_Together'] == 1]
            together_wine1 = together1['MntWines'].sum()
            together_fruit1 = together1 ['MntFruits'].sum()
            together_meat1 = together1 ['MntMeatProducts'].sum()
            together_fish1 = together1 ['MntFishProducts'].sum()
            together_sweet1 = together1 ['MntSweetProducts'].sum()
            together_gold1 = together1 ['MntGoldProds'].sum()
            together_regular1 = together1 ['MntRegularProds'].sum()
            data = {'MaritalStatus' : ['Together', 'Together', 'Together', 'Together', 'Together', 'Together', 'Together'],
                    'ProductCategory' : ['Wine', 'Fruit', 'Meat', 'Fish', 'Sweets', 'Gold', 'Regular Products'],
                    'AmountSpent' : [together_wine1, together_fruit1, together_meat1, together_fish1, together_sweet1, together_gold1, together_regular1]}
            together1 = pd.DataFrame(data)

            #widow
            widow1 = cluster1[cluster1['marital_Widow'] == 1]
            widow_wine1 = widow1['MntWines'].sum()
            widow_fruit1 = widow1 ['MntFruits'].sum()
            widow_meat1 = widow1 ['MntMeatProducts'].sum()
            widow_fish1 = widow1 ['MntFishProducts'].sum()
            widow_sweet1 = widow1 ['MntSweetProducts'].sum()
            widow_gold1 = widow1 ['MntGoldProds'].sum()
            widow_regular1 = widow1 ['MntRegularProds'].sum()
            data = {'MaritalStatus' : ['Widow', 'Widow', 'Widow', 'Widow', 'Widow', 'Widow', 'Widow'],
                    'ProductCategory' : ['Wine', 'Fruit', 'Meat', 'Fish', 'Sweets', 'Gold', 'Regular Products'],
                    'AmountSpent' : [widow_wine1, widow_fruit1, widow_meat1, widow_fish1, widow_sweet1, widow_gold1, widow_regular1]}
            widow1 = pd.DataFrame(data)

            # Combining all the dataframes
            frames = [divorced1, married1, single1, together1, widow1]
            result = pd.concat(frames)

            # Plotting the stacked Barchart
            fig5 = px.bar(result, x='MaritalStatus', y= 'AmountSpent', color = 'ProductCategory',
            color_discrete_sequence = ['#8dbdff', '#ffb0d9', '#f7e4f8','#b3d4ff','#f7e4f8','#ffb0d9','#d6eaff']
)

            # Remove gridlines
            fig5.update_layout(xaxis=dict(showgrid=False),
                          yaxis=dict(showgrid=False)
            )
            # Aligning title
            fig5.update_layout(
                title={
                    'text': "Spending by Marital Status",
                    'x':0.48,
                    'y':0.99,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            ### Horizontal bar chart for segment 1 ###
            # Interaction with the 5 campaigns

            # Step 1- Getting the values
            df1_total_cmp1 = cluster1['AcceptedCmp1'].sum()
            df1_total_cmp2 = cluster1['AcceptedCmp2'].sum()
            df1_total_cmp3 = cluster1['AcceptedCmp3'].sum()
            df1_total_cmp4 = cluster1['AcceptedCmp4'].sum()
            df1_total_cmp5 = cluster1['AcceptedCmp5'].sum()

            # Step2 - DataFrame
            data_1 = {'Segment' : [name1, name1, name1, name1, name1],
                    'Campaign' : ['Campaign1', 'Campaign2', 'Campaign3', 'Campaign4', 'Campaign5'],
                    'Count' : [df1_total_cmp1, df1_total_cmp2, df1_total_cmp3, df1_total_cmp4, df1_total_cmp5]}

            df_1 = pd.DataFrame(data_1)

            fig1 = go.Figure(go.Bar(
                                    x = data_1['Count'],
                                    y = data_1['Campaign'],
                                    marker_color = ['#ffb0d9', '#f9c9e2', '#8dbdff','#b3d4ff', '#d6eaff'],
                                    orientation = 'h'))
            # Remove gridlines
            fig1.update_layout(xaxis=dict(showgrid=False),
                          yaxis=dict(showgrid=False))
            # Aligning title
            fig1.update_layout(
                title={
                    'text': "Response to Previous Campaigns",
                    'x':0.48,
                    #'y':0.99,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            ### Scatterplot ###
            # Tenure and Spending
            # changing customer days from continuous to discrete
            cluster1['Tenure'] = round(cluster1 ['Customer_Days']/365)
            cluster1 ['Tenure'] = cluster1 ['Tenure'].astype('int64')

            years = cluster1 ['Tenure'].unique().tolist()
            count = cluster1 ['Tenure'].value_counts().tolist()

            # money spent by tenure = 6
            tenure6 = cluster1 [cluster1 ['Tenure'] == 6]
            monetary6 = mean(tenure6['Monetary'])

            # money spent by tenure = 7
            tenure7 = cluster1 [cluster1 ['Tenure'] == 7]
            monetary7 = mean(tenure7['Monetary'])

            # money spent by tenure = 8
            tenure8 = cluster1 [cluster1 ['Tenure'] == 8]
            monetary8 = mean(tenure8['Monetary'])

            data = {'Tenure':years,
                    'count':count,
                    'Monetary':[monetary8, monetary7, monetary6],
                    'segment':['Segment 1', 'Segment 1', 'Segment 1']}
            df_tenure = pd.DataFrame(data)

            fig3 = px.scatter(df_tenure, x="Tenure", y="Monetary",
            	         size="count", hover_name="segment", log_x=True, size_max=60,
                        color = 'segment', color_discrete_sequence = ["#ffb0d9"])
            # Remove gridlines
            fig3.update_layout(xaxis=dict(showgrid=False),
                          yaxis=dict(showgrid=False))
            # Aligning title
            fig3.update_layout(
                title={
                    'text': "Spending by Tenure",
                    'x':0.48,
                    #'y':0.99,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            # Plotting the charts
            col1, col2, col3 = st.columns(3)
            with col1:
                st.plotly_chart(fig1)
                st.plotly_chart(fig4)
            with col2:
                st.plotly_chart(fig2)
                st.plotly_chart(fig5)
            with col3:
                st.plotly_chart(fig3)
                st.plotly_chart(fig6)

        if option == 'Segment 2':
            ### Treemap for segment 2 ###
            # Step 1- Subsetting the df for Cluster 2
            cluster2 = data[data['Segment'] == 'Segment 2']

            # Step 2- Calculating the amount spent ($) for each product category
            wine2 = cluster2['MntWines'].sum()
            fruit2 = cluster2['MntFruits'].sum()
            meat2 = cluster2['MntMeatProducts'].sum()
            fish2 = cluster2['MntFishProducts'].sum()
            sweet2 = cluster2['MntSweetProducts'].sum()
            gold2 = cluster2['MntGoldProds'].sum()
            regular2 = cluster2['MntRegularProds'].sum()

            # Getting the name of the cluster
            name2 = cluster2.iloc[0,-1]

            # Step 3- df_2
            data = {'Segment' : [name2, name2, name2, name2, name2, name2, name2],
                   'ProductCategory' : ['Wine', 'Fruit', 'Meat', 'Fish', 'Sweets', 'Gold', 'Regular Products'],
                   'Monetary' : [wine2, fruit2, meat2, fish2, sweet2, gold2, regular2]}
            df_2 = pd.DataFrame(data)

            # Step 4- Plot Treemap
            fig2 = px.treemap(df_2,
                            names = 'ProductCategory',
                             values = 'Monetary',
                             parents = 'Segment',
                             color = 'ProductCategory',
                             color_discrete_sequence = ['#f9c9e2', '#d6eaff', '#f7e4f8', '#8dbdff', '#8dbdff', '#ffb0d9', '#b3d4ff'])
            # Aligning title
            fig2.update_layout(
                title={
                    'text': "Spending by Product Category",
                    'x':0.48,
                    #'y':0.99,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            ### Pie chart for segment 2 ###
            # Step 1- Subsetting the df for Cluster 2
            pie2 = cluster2[cluster2['Segment'] == 'Segment 2']

            # Step 2- Calculating the sum for each channel
            web2 = pie2['NumWebPurchases'].sum()
            store2 = pie2['NumStorePurchases'].sum()
            catalog2 = pie2['NumCatalogPurchases'].sum()

            # Getting the name of the cluster
            name2 = cluster2.iloc[0,-1]

            # Step 3- df_2
            data = {'Segment' : [name2, name2, name2],
                   'Channel' : ['Website', 'In-Store', 'Catalog'],
                   'Count' : [web2, store2, catalog2]}
            df_2 = pd.DataFrame(data)

            # Step 4- Plot pie
            fig6 = px.pie(df_2, values = 'Count', names = 'Channel', color = 'Channel',
            color_discrete_sequence = ['#8dbdff', '#b3d4ff', '#d6eaff'])

            # Aligning title
            fig6.update_layout(
                title={
                    'text': "Shopping Channels",
                    'y':0.99,
                    'x':0.49,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            ### Scatterplot for segment 2 ###
            # Relationship between web visits and deals purchased
            fig4 = px.scatter(cluster2, x='NumWebVisitsMonth', y='NumDealsPurchases', trendline="ols",
            color_discrete_sequence = ['#ffb0d9'])

            # Remove gridlines
            fig4.update_layout(xaxis=dict(showgrid=False),
                            yaxis=dict(showgrid=False))
            # Aligning title
            fig4.update_layout(
                title={
                    'text': "Deals Purchased by Monthly Webiste Visits",
                    'x':0.48,
                    'y':0.99,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            ### Stacked Barchart for segment 2 ###
            # Amount spent on each product category by each marital status of cluster 2
            # divorced
            divorced2 = cluster2[cluster2['marital_Divorced'] == 1]
            divorced_wine2 = divorced2['MntWines'].sum()
            divorced_fruit2 = divorced2['MntFruits'].sum()
            divorced_meat2 = divorced2['MntMeatProducts'].sum()
            divorced_fish2 = divorced2['MntFishProducts'].sum()
            divorced_sweet2 = divorced2['MntSweetProducts'].sum()
            divorced_gold2 = divorced2['MntGoldProds'].sum()
            divorced_regular2 = divorced2['MntRegularProds'].sum()
            data = {'MaritalStatus' : ['Divorced', 'Divorced', 'Divorced', 'Divorced', 'Divorced', 'Divorced', 'Divorced'],
                    'ProductCategory' : ['Wine', 'Fruit', 'Meat', 'Fish', 'Sweets', 'Gold', 'Regular Products'],
                    'AmountSpent' : [divorced_wine2, divorced_fruit2, divorced_meat2, divorced_fish2, divorced_sweet2, divorced_gold2, divorced_regular2]}
            divorced2 = pd.DataFrame(data)

            # married
            married2 = cluster2[cluster2['marital_Married'] == 1]
            married_wine2 = married2['MntWines'].sum()
            married_fruit2 = married2['MntFruits'].sum()
            married_meat2 = married2['MntMeatProducts'].sum()
            married_fish2 = married2['MntFishProducts'].sum()
            married_sweet2 = married2['MntSweetProducts'].sum()
            married_gold2 = married2['MntGoldProds'].sum()
            married_regular2 = married2['MntRegularProds'].sum()
            data = {'MaritalStatus' : ['Married', 'Married', 'Married', 'Married', 'Married', 'Married', 'Married'],
                    'ProductCategory' : ['Wine', 'Fruit', 'Meat', 'Fish', 'Sweets', 'Gold', 'Regular Products'],
                    'AmountSpent' : [married_wine2, married_fruit2, married_meat2, married_fish2, married_sweet2, married_gold2, married_regular2]}
            married2 = pd.DataFrame(data)

            # single
            single2 = cluster2[cluster2['marital_Single'] == 1]
            single_wine2 = single2['MntWines'].sum()
            single_fruit2 = single2['MntFruits'].sum()
            single_meat2 = single2['MntMeatProducts'].sum()
            single_fish2 = single2['MntFishProducts'].sum()
            single_sweet2 = single2['MntSweetProducts'].sum()
            single_gold2 = single2['MntGoldProds'].sum()
            single_regular2 = single2['MntRegularProds'].sum()
            data = {'MaritalStatus' : ['Single', 'Single', 'Single', 'Single', 'Single', 'Single', 'Single'],
                    'ProductCategory' : ['Wine', 'Fruit', 'Meat', 'Fish', 'Sweets', 'Gold', 'Regular Products'],
                    'AmountSpent' : [single_wine2, single_fruit2, single_meat2, single_fish2, single_sweet2, single_gold2, single_regular2]}
            single2 = pd.DataFrame(data)

            # together
            together2 = cluster2[cluster2['marital_Together'] == 1]
            together_wine2 = together2['MntWines'].sum()
            together_fruit2 = together2['MntFruits'].sum()
            together_meat2 = together2['MntMeatProducts'].sum()
            together_fish2 = together2['MntFishProducts'].sum()
            together_sweet2 = together2['MntSweetProducts'].sum()
            together_gold2 = together2['MntGoldProds'].sum()
            together_regular2 = together2['MntRegularProds'].sum()
            data = {'MaritalStatus' : ['Together', 'Together', 'Together', 'Together', 'Together', 'Together', 'Together'],
                    'ProductCategory' : ['Wine', 'Fruit', 'Meat', 'Fish', 'Sweets', 'Gold', 'Regular Products'],
                    'AmountSpent' : [together_wine2, together_fruit2, together_meat2, together_fish2, together_sweet2, together_gold2, together_regular2]}
            together2 = pd.DataFrame(data)

            #widow
            widow2 = cluster2[cluster2['marital_Widow'] == 1]
            widow_wine2 = widow2['MntWines'].sum()
            widow_fruit2 = widow2['MntFruits'].sum()
            widow_meat2 = widow2['MntMeatProducts'].sum()
            widow_fish2 = widow2['MntFishProducts'].sum()
            widow_sweet2 = widow2['MntSweetProducts'].sum()
            widow_gold2 = widow2['MntGoldProds'].sum()
            widow_regular2 = widow2['MntRegularProds'].sum()
            data = {'MaritalStatus' : ['Widow', 'Widow', 'Widow', 'Widow', 'Widow', 'Widow', 'Widow'],
                    'ProductCategory' : ['Wine', 'Fruit', 'Meat', 'Fish', 'Sweets', 'Gold', 'Regular Products'],
                    'AmountSpent' : [widow_wine2, widow_fruit2, widow_meat2, widow_fish2, widow_sweet2, widow_gold2, widow_regular2]}
            widow2 = pd.DataFrame(data)

            # Combining all the dataframes
            frames = [divorced2, married2, single2, together2, widow2]
            result = pd.concat(frames)

            # Plotting the stacked Barchart
            fig5 = px.bar(result, x='MaritalStatus', y= 'AmountSpent', color = 'ProductCategory',
            color_discrete_sequence = ['#8dbdff', '#ffb0d9', '#f7e4f8','#b3d4ff','#f7e4f8','#ffb0d9','#d6eaff'])

            # Remove gridlines
            fig5.update_layout(xaxis=dict(showgrid=False),
                          yaxis=dict(showgrid=False)
            )
            # Aligning title
            fig5.update_layout(
                title={
                    'text': "Spending by Marital Status",
                    'x':0.48,
                    'y':0.99,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            ### Horizontal bar chart for segment 2 ###
            # Interaction with the 5 campaigns

            # Step 1- Getting the values
            df2_total_cmp1 = cluster2['AcceptedCmp1'].sum()
            df2_total_cmp2 = cluster2 ['AcceptedCmp2'].sum()
            df2_total_cmp3 = cluster2 ['AcceptedCmp3'].sum()
            df2_total_cmp4 = cluster2 ['AcceptedCmp4'].sum()
            df2_total_cmp5 = cluster2 ['AcceptedCmp5'].sum()

            # Step2 - DataFrame
            data_2 = {'Segment' : [name2, name2, name2, name2, name2],
                    'Campaign' : ['Campaign1', 'Campaign2', 'Campaign3', 'Campaign4', 'Campaign5'],
                    'Count' : [df2_total_cmp1, df2_total_cmp2, df2_total_cmp3, df2_total_cmp4, df2_total_cmp5]}

            df_2 = pd.DataFrame(data_2)

            fig1 = go.Figure(go.Bar(
                                    x = data_2['Count'],
                                    y = data_2['Campaign'],
                                    marker_color = ['#ffb0d9', '#f9c9e2', '#8dbdff','#b3d4ff', '#d6eaff'],
                                    orientation = 'h'))
            # Remove gridlines
            fig1.update_layout(xaxis=dict(showgrid=False),
                          yaxis=dict(showgrid=False))
            # Aligning title
            fig1.update_layout(
                title={
                    'text': "Response to Previous Campaigns",
                    'x':0.48,
                    #'y':0.99,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            ### Scatterplot ###
            # Tenure and Spending
            # changing customer days from continuous to discrete
            cluster2['Tenure'] = round(cluster2['Customer_Days']/365)
            cluster2['Tenure'] = cluster2['Tenure'].astype('int64')

            years = cluster2['Tenure'].unique().tolist()
            count = cluster2['Tenure'].value_counts().tolist()

            # money spent by tenure = 6
            tenure6 = cluster2[cluster2['Tenure'] == 6]
            monetary6 = mean(tenure6['Monetary'])

            # money spent by tenure = 7
            tenure7 = cluster2[cluster2['Tenure'] == 7]
            monetary7 = mean(tenure7['Monetary'])

            # money spent by tenure = 8
            tenure8 = cluster2[cluster2['Tenure'] == 8]
            monetary8 = mean(tenure8['Monetary'])

            data = {'Tenure':years,
                    'count':count,
                    'Monetary':[monetary8, monetary7, monetary6],
                    'segment':['Segment 2', 'Segment 2', 'Segment 2']}
            df_tenure = pd.DataFrame(data)

            fig3 = px.scatter(df_tenure, x="Tenure", y="Monetary",
            	         size="count", hover_name="segment", log_x=True, size_max=60,
                         color = 'segment', color_discrete_sequence = ["#ffb0d9"])
            # Remove gridlines
            fig3.update_layout(xaxis=dict(showgrid=False),
                          yaxis=dict(showgrid=False))
            # Aligning title
            fig3.update_layout(
                title={
                    'text': "Spending by Tenure",
                    'x':0.48,
                    #'y':0.99,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            # Plotting the charts
            col1, col2, col3 = st.columns(3)
            with col1:
                st.plotly_chart(fig1)
                st.plotly_chart(fig4)
            with col2:
                st.plotly_chart(fig2)
                st.plotly_chart(fig5)
            with col3:
                st.plotly_chart(fig3)
                st.plotly_chart(fig6)
#-----------------------------------------------------------------------------#
# Component 6: Segment Demographics
if selected == "Demographics":

    option = st.selectbox(
        'Select a customer segment',
        ('Segment 0', 'Segment 1', 'Segment 2'))

    # Dataset of the user
    if uploaded_file is not None:
        df_X = pd.read_csv(uploaded_file)

        # Adding 2 new columns
        df_X['Monetary'] = df_X['MntWines'] + df_X['MntFruits'] + df_X['MntMeatProducts'] + df_X['MntFishProducts'] + df_X['MntSweetProducts'] + df_X['MntGoldProds'] + df_X['MntRegularProds']
        df_X['Frequency'] = df_X['NumDealsPurchases'] + df_X['NumWebPurchases'] + df_X['NumCatalogPurchases'] + df_X['NumStorePurchases']

        ### Calculate Recency, Frequency, Monetary ###
        # Subsetting the dataframe to only include the features: 'Recency', 'Frequency', 'Monetary'
        df_RFM = df_X.copy()
        columns = ['Recency', 'Frequency', 'Monetary']
        df_RFM = df_RFM[columns]

        # Choose number of clusters
        K = 3
        model = KMeans(n_clusters = K, random_state = 1)
        model.fit(df_RFM)
        df_RFM['Cluster'] = model.labels_

        # Change the cluster columns data type into discrete values
        df_RFM['Segment'] = 'Segment ' + df_RFM['Cluster'].astype('str')

        # Calculate average values for each cluster and return size of each
        df_agg = df_RFM.groupby('Segment').agg({
                                                'Recency' : 'mean',
                                                'Frequency' : 'mean',
                                                'Monetary' : ['mean', 'count']}).round(0)

        df_agg.columns = df_agg.columns.droplevel()
        df_agg.columns = ['RecencyMean', 'FrequencyMean', 'MonetaryMean', 'Count']
        df_agg['Percent'] = round((df_agg['Count']/df_agg.Count.sum())*100,2)

        # Reset the index
        df_agg = df_agg.reset_index()

        # Adding cluster label to each row in original dataframe
        df_X['Cluster'] = model.labels_
        df_X['Segment'] = 'Segment ' + df_X['Cluster'].astype('str')

        # Plotting graphs for each segment on a different page
        # USE df_X

        if option == 'Segment 0':
            ### Histogram for segment 0 ###
            # Subsetting the df for Cluster 0
            cluster0 = df_X[df_X['Segment'] == 'Segment 0']
            # Plotting the histogram of age distribution in cluster0
            fig1 = px.histogram(cluster0, x = 'Income', nbins = 8, color_discrete_sequence = ['#8dbdff'])

            # Remove gridlines
            fig1.update_layout(xaxis=dict(showgrid=False),
                            yaxis=dict(showgrid=False))

            # Aligning title
            fig1.update_layout(
                title={
                    'text': "Income Distribution",
                    #'y':0.99,
                    'x':0.49,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            ### Barchart for segment 0 ###
            # Marital status
            divorced0 = cluster0['marital_Divorced'].sum()
            married0 = cluster0['marital_Married'].sum()
            single0 = cluster0['marital_Single'].sum()
            together0 = cluster0['marital_Together'].sum()
            widow0 = cluster0['marital_Widow'].sum()

            fig4 = go.Figure(go.Bar(
            x = [divorced0, married0, single0, together0, widow0],
            y = ['Divorced', 'Married', 'Single', 'Together', 'Widow'],
            marker_color = ['#ffb0d9', '#f9c9e2', '#8dbdff','#b3d4ff', '#d6eaff'],
            orientation = 'h'))

            # Remove gridlines
            fig4.update_layout(xaxis=dict(showgrid=False),
                            yaxis=dict(showgrid=False))
            # Aligning title
            fig4.update_layout(
                title={
                    'text': "Marital Status",
                    #'y':0.99,
                    'x':0.49,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            ### Pie chart for segment 0 ###
            # Education
            secondcycle0 = cluster0['education_2n Cycle'].sum()
            basic0 = cluster0['education_Basic'].sum()
            graduation0 = cluster0['education_Graduation'].sum()
            master0 = cluster0['education_Master'].sum()
            phd0 = cluster0['education_PhD'].sum()

            data = {'education' : ['Second Cycle', 'High School', 'Bachelors', 'Masters', 'PhD'],
                    'count' : [secondcycle0, basic0, graduation0, master0, phd0]}
            df_education = pd.DataFrame(data)

            # pie chart
            fig2 = px.pie(df_education, values = 'count', names = 'education', color = 'education',
            color_discrete_sequence = ["#8dbdff", "#b3d4ff", "#d6eaff", '#ffb0d9', '#f9c9e2', '#f7e4f8'])

            # Aligning title
            fig2.update_layout(
                title={
                    'text': "Education Achievement",
                    #'y':0.99,
                    'x':0.49,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            ### distplot for segment 0 ###
            # Age distribution
            # Plot the distribution of R
            x = cluster0['Age']
            hist_data = [x]
            group_labels = ['cluster0']

            colors = ['#ffb0d9']
            fig3 = ff.create_distplot(hist_data, group_labels, bin_size=8, colors = colors)
            fig3.update_xaxes(showgrid=False)
            fig3.update_yaxes(showgrid=False)

            # Aligning title
            fig3.update_layout(
                title={
                    'text': "Age Distribution",
                    #'y':0.99,
                    'x':0.49,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            ### Bar chart for segment 0 ###
            # Number of dependents
            # Number of customers with kids
            df0_kid = cluster0[cluster0['Kidhome'] != 0]
            kid0 = df0_kid['Kidhome'].count()

            # Number of customers with teenagers
            df0_teen = cluster0[cluster0['Teenhome'] != 0]
            teen0 = df0_teen['Kidhome'].count()

            # Number of customers with no kids
            nodependents0 = sum((cluster0['Kidhome'] == 0) & (cluster0['Teenhome'] == 0))

            data = {'dependents' : ['Kids', 'Teens', 'No Dependents'],
                    'count' : [kid0, teen0, nodependents0]}
            df_dependents = pd.DataFrame(data)

            # bar chart
            fig6 = px.bar(df_dependents, x = 'dependents', y = 'count', color = 'dependents',
            color_discrete_sequence = ['#ffb0d9', '#f9c9e2', '#8dbdff'])

            # Remove gridlines
            fig6.update_layout(xaxis=dict(showgrid=False),
                          yaxis=dict(showgrid=False))
            # Aligning title
            fig6.update_layout(
                title={
                    'text': "Number of Dependents",
                    #'y':0.99,
                    'x':0.48,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            # Tenure ie. Customer_Days
            # changing customer days from continuous to discrete
            cluster0['Tenure'] = round(cluster0 ['Customer_Days']/365)
            cluster0 ['Tenure'] = cluster0 ['Tenure'].astype('int64')

            years = cluster0 ['Tenure'].unique().tolist()
            count = cluster0 ['Tenure'].value_counts().tolist()

            data = {'Tenure':years,
                    'count':count,
                    'segment':['Segment 0', 'Segment 0', 'Segment 0']}
            df_tenure = pd.DataFrame(data)

            fig5 = px.scatter(df_tenure, x="Tenure", y="count",
            	         size="count", hover_name="segment", log_x=True, size_max=60,
                         color = 'segment', color_discrete_sequence = ['#b3d4ff'])
            # Remove gridlines
            fig5.update_layout(xaxis=dict(showgrid=False),
                          yaxis=dict(showgrid=False))

            # Aligning title
            fig5.update_layout(
                title={
                    'text': "Customer Tenure",
                    #'y':0.99,
                    'x':0.48,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            # Plotting the charts
            col1, col2, col3 = st.columns(3)
            with col1:
                st.plotly_chart(fig1)
                st.plotly_chart(fig4)
            with col2:
                st.plotly_chart(fig2)
                st.plotly_chart(fig5)
            with col3:
                st.plotly_chart(fig3)
                st.plotly_chart(fig6)

        if option == 'Segment 1':
            ### Histogram for segment 1 ###
            # Step 1- Subsetting the df for Cluster 1
            cluster1 = df_X[df_X['Segment'] == 'Segment 1']

            # Plotting the histogram of age distribution in cluster0
            fig1 = px.histogram(cluster1, x = 'Income', nbins = 8, color_discrete_sequence = ["#8dbdff"])

            # Remove gridlines
            fig1.update_layout(xaxis=dict(showgrid=False),
                            yaxis=dict(showgrid=False))

            # Aligning title
            fig1.update_layout(
                title={
                    'text': "Income Distribution",
                    #'y':0.99,
                    'x':0.49,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            ### Barchart for segment 1 ###
            # Marital status
            divorced1 = cluster1['marital_Divorced'].sum()
            married1 = cluster1['marital_Married'].sum()
            single1 = cluster1['marital_Single'].sum()
            together1 = cluster1['marital_Together'].sum()
            widow1 = cluster1['marital_Widow'].sum()

            fig4 = go.Figure(go.Bar(
            x = [divorced1, married1, single1, together1, widow1],
            y = ['Divorced', 'Married', 'Single', 'Together', 'Widow'],
            marker_color = ['#ffb0d9', '#f9c9e2', '#8dbdff','#b3d4ff', '#d6eaff'],
            orientation = 'h'))

            # Remove gridlines
            fig4.update_layout(xaxis=dict(showgrid=False),
                          yaxis=dict(showgrid=False)
            )

            # Aligning title
            fig4.update_layout(
                title={
                    'text': "Marital Status",
                    #'y':0.99,
                    'x':0.49,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            ### Pie chart for segment 1 ###
            # Education
            secondcycle1 = cluster1['education_2n Cycle'].sum()
            basic1 = cluster1 ['education_Basic'].sum()
            graduation1 = cluster1 ['education_Graduation'].sum()
            master1 = cluster1 ['education_Master'].sum()
            phd1 = cluster1 ['education_PhD'].sum()

            data = {'education' : ['Second Cycle', 'High School', 'Bachelors', 'Masters', 'PhD'],
                    'count' : [secondcycle1, basic1, graduation1, master1, phd1]}
            df_education = pd.DataFrame(data)

            # pie chart
            fig2 = px.pie(df_education, values = 'count', names = 'education', color = 'education',
            color_discrete_sequence = ["#8dbdff", "#b3d4ff", "#d6eaff", '#ffb0d9', '#f9c9e2', '#f7e4f8'])

            # Aligning title
            fig2.update_layout(
                title={
                    'text': "Education Achievement",
                    #'y':0.99,
                    'x':0.49,
                    'xanchor': 'center',
                    'yanchor': 'top'})

        ### distplot for segment 1 ###
            # Age distribution
            # Plot the distribution of R
            x = cluster1['Age']
            hist_data = [x]
            group_labels = ['cluster1']

            colors = ['#ffb0d9']
            fig3 = ff.create_distplot(hist_data, group_labels, bin_size=8, colors = colors)
            fig3.update_layout(title_text = 'Age Distribution')
            fig3.update_xaxes(showgrid=False)
            fig3.update_yaxes(showgrid=False)

            # Aligning title
            fig3.update_layout(
                title={
                    'text': "Age Distribution",
                    #'y':0.99,
                    'x':0.49,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            ### Bar chart for segment 1 ###
            # Number of dependents
            # Number of customers with kids
            df1_kid = cluster1[cluster1['Kidhome'] != 0]
            kid1 = df1_kid['Kidhome'].count()

            # Number of customers with teenagers
            df1_teen = cluster1[cluster1['Teenhome'] != 0]
            teen1 = df1_teen['Kidhome'].count()

            # Number of customers with no kids
            nodependents1 = sum((cluster1['Kidhome'] == 0) & (cluster1['Teenhome'] == 0))


            data = {'dependents' : ['Kids', 'Teens', 'No Dependents'],
                    'count' : [kid1, teen1, nodependents1]}
            df_dependents = pd.DataFrame(data)

            # bar chart
            fig6 = px.bar(df_dependents, x = 'dependents', y = 'count', color = 'dependents',
            color_discrete_sequence = ['#ffb0d9', '#f9c9e2', '#8dbdff'])

            # Remove gridlines
            fig6.update_layout(xaxis=dict(showgrid=False),
                          yaxis=dict(showgrid=False))
            # Aligning title
            fig6.update_layout(
                title={
                    'text': "Number of Dependents",
                    #'y':0.99,
                    'x':0.48,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            # Tenure ie. Customer_Days
            # changing customer days from continuous to discrete
            cluster1['Tenure'] = round(cluster1['Customer_Days']/365)
            cluster1['Tenure'] = cluster1['Tenure'].astype('int64')

            years = cluster1['Tenure'].unique().tolist()
            count = cluster1['Tenure'].value_counts().tolist()

            data = {'Tenure':years,
                    'count':count,
                    'segment':['Segment 1', 'Segment 1', 'Segment 1']}
            df_tenure = pd.DataFrame(data)

            fig5 = px.scatter(df_tenure, x="Tenure", y="count",
            	         size="count", hover_name="segment", log_x=True, size_max=60,
                         color = 'segment', color_discrete_sequence = ['#b3d4ff'])
            # Remove gridlines
            fig5.update_layout(xaxis=dict(showgrid=False),
                          yaxis=dict(showgrid=False))
            # Aligning title
            fig5.update_layout(
                title={
                    'text': "Customer Tenure",
                    #'y':0.99,
                    'x':0.48,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            # Plotting the charts
            col1, col2, col3 = st.columns(3)
            with col1:
                st.plotly_chart(fig1)
                st.plotly_chart(fig4)
            with col2:
                st.plotly_chart(fig2)
                st.plotly_chart(fig5)
            with col3:
                st.plotly_chart(fig3)
                st.plotly_chart(fig6)

        if option == 'Segment 2':

            ### Histogram for segment 2 ###
            # Subsetting the df for Cluster 2
            cluster2 = df_X[df_X['Segment'] == 'Segment 2']

            # Plotting the histogram of age distribution in cluster0
            fig1 = px.histogram(cluster2, x = 'Income', nbins = 8, color_discrete_sequence = ["#8dbdff"])

            # Remove gridlines
            fig1.update_layout(xaxis=dict(showgrid=False),
                            yaxis=dict(showgrid=False))

            # Aligning title
            fig1.update_layout(
                title={
                    'text': "Income Distribution",
                    'y':0.99,
                    'x':0.49,
                    'xanchor': 'center',
                    'yanchor': 'top'})
            # Aligning title
            fig1.update_layout(
                title={
                    'text': "Income Distribution",
                    #'y':0.99,
                    'x':0.49,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            ### Barchart for segment 2 ###
            # Marital status
            divorced2 = cluster2['marital_Divorced'].sum()
            married2 = cluster2['marital_Married'].sum()
            single2 = cluster2['marital_Single'].sum()
            together2 = cluster2['marital_Together'].sum()
            widow2 = cluster2['marital_Widow'].sum()

            fig4 = go.Figure(go.Bar(
            x = [divorced2, married2, single2, together2, widow2],
            y = ['Divorced', 'Married', 'Single', 'Together', 'Widow'],
            marker_color = ['#ffb0d9', '#f9c9e2', '#8dbdff','#b3d4ff', '#d6eaff'],
            orientation = 'h'))

            # Remove gridlines
            fig4.update_layout(xaxis=dict(showgrid=False),
                          yaxis=dict(showgrid=False)
            )
            # Aligning title
            fig4.update_layout(
                title={
                    'text': "Marital Status",
                    #'y':0.99,
                    'x':0.49,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            ### Pie chart for segment 2 ###
            # Education
            secondcycle2 = cluster2['education_2n Cycle'].sum()
            basic2 = cluster2['education_Basic'].sum()
            graduation2 = cluster2['education_Graduation'].sum()
            master2 = cluster2['education_Master'].sum()
            phd2 = cluster2['education_PhD'].sum()

            data = {'education' : ['Second Cycle', 'High School', 'Bachelors', 'Masters', 'PhD'],
                    'count' : [secondcycle2, basic2, graduation2, master2, phd2]}
            df_education = pd.DataFrame(data)

            # pie chart
            fig2 = px.pie(df_education, values = 'count', names = 'education', color = 'education',
            color_discrete_sequence = ["#8dbdff", "#b3d4ff", "#d6eaff", '#ffb0d9', '#f9c9e2', '#f7e4f8'])

            # Aligning title
            fig2.update_layout(
                title={
                    'text': "Education Achievement",
                    #'y':0.99,
                    'x':0.49,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            ### distplot for segment 2 ###
            # Age distribution
            # Plot the distribution of R
            x = cluster2['Age']
            hist_data = [x]
            group_labels = ['cluster2']

            colors = ['#ffb0d9']
            fig3 = ff.create_distplot(hist_data, group_labels, bin_size=8, colors = colors)
            fig3.update_layout(title_text = 'Age Distribution')
            fig3.update_xaxes(showgrid=False)
            fig3.update_yaxes(showgrid=False)

            # Aligning title
            fig3.update_layout(
                title={
                    'text': "Age Distribution",
                    #'y':0.99,
                    'x':0.49,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            ### Bar chart for segment 2 ###
            # Number of dependents
            # Number of customers with kids
            df2_kid = cluster2[cluster2['Kidhome'] != 0]
            kid2 = df2_kid['Kidhome'].count()

            # Number of customers with teenagers
            df2_teen = cluster2[cluster2['Teenhome'] != 0]
            teen2 = df2_teen['Kidhome'].count()

            # Number of customers with no kids
            nodependents2 = sum((cluster2['Kidhome'] == 0) & (cluster2['Teenhome'] == 0))

            data = {'dependents' : ['Kids', 'Teens', 'No Dependents'],
                    'count' : [kid2, teen2, nodependents2]}
            df_dependents = pd.DataFrame(data)

            # bar chart
            fig6 = px.bar(df_dependents, x = 'dependents', y = 'count', color = 'dependents',
            color_discrete_sequence = ['#ffb0d9', '#f9c9e2', '#8dbdff'])

            # Remove gridlines
            fig6.update_layout(xaxis=dict(showgrid=False),
                          yaxis=dict(showgrid=False))

            # Aligning title
            fig6.update_layout(
                title={
                    'text': "Number of Dependents",
                    #'y':0.99,
                    'x':0.48,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            # Tenure ie. Customer_Days
            # changing customer days from continuous to discrete
            cluster2['Tenure'] = round(cluster2['Customer_Days']/365)
            cluster2['Tenure'] = cluster2['Tenure'].astype('int64')

            years = cluster2['Tenure'].unique().tolist()
            count = cluster2['Tenure'].value_counts().tolist()

            data = {'Tenure':years,
                    'count':count,
                    'segment':['Segment 2', 'Segment 2', 'Segment 2']}
            df_tenure = pd.DataFrame(data)

            fig5 = px.scatter(df_tenure, x="Tenure", y="count",
            	         size="count", hover_name="segment", log_x=True, size_max=60,
                         color = 'segment', color_discrete_sequence = ['#b3d4ff'])
            # Remove gridlines
            fig5.update_layout(xaxis=dict(showgrid=False),
                          yaxis=dict(showgrid=False))
            # Aligning title
            fig5.update_layout(
                title={
                    'text': "Customer Tenure",
                    #'y':0.99,
                    'x':0.48,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            # Plotting the charts
            col1, col2, col3 = st.columns(3)
            with col1:
                st.plotly_chart(fig1)
                st.plotly_chart(fig4)
            with col2:
                st.plotly_chart(fig2)
                st.plotly_chart(fig5)
            with col3:
                st.plotly_chart(fig3)
                st.plotly_chart(fig6)

    # Our dataset
    else:
        data = pd.read_csv('ifood_df.csv')

        # Adding 2 new columns
        data['Monetary'] = data['MntWines'] + data['MntFruits'] + data['MntMeatProducts'] + data['MntFishProducts'] + data['MntSweetProducts'] + data['MntGoldProds'] + data['MntRegularProds']
        data['Frequency'] = data['NumDealsPurchases'] + data['NumWebPurchases'] + data['NumCatalogPurchases'] + data['NumStorePurchases']

        ### Calculate Recency, Frequency, Monetary ###
        # Subsetting the dataframe to only include the features: 'Recency', 'Frequency', 'Monetary'
        df_RFM = data.copy()
        columns = ['Recency', 'Frequency', 'Monetary']
        df_RFM = df_RFM[columns]

        # Choose number of clusters
        K = 3
        model = KMeans(n_clusters = K, random_state = 1)
        model.fit(df_RFM)
        df_RFM['Cluster'] = model.labels_

        # Change the cluster columns data type into discrete values
        df_RFM['Segment'] = 'Segment ' + df_RFM['Cluster'].astype('str')

        # Calculate average values for each cluster and return size of each
        df_agg = df_RFM.groupby('Segment').agg({
                                                'Recency' : 'mean',
                                                'Frequency' : 'mean',
                                                'Monetary' : ['mean', 'count']}).round(0)

        df_agg.columns = df_agg.columns.droplevel()
        df_agg.columns = ['RecencyMean', 'FrequencyMean', 'MonetaryMean', 'Count']
        df_agg['Percent'] = round((df_agg['Count']/df_agg.Count.sum())*100,2)

        # Reset the index
        df_agg = df_agg.reset_index()

        # Adding cluster label to each row in original dataframe
        data['Cluster'] = model.labels_
        data['Segment'] = 'Segment ' + data['Cluster'].astype('str')

        ######## Plotting graphs for each segment on a different page ########
        # USE data
        if option == 'Segment 0':
            ### Histogram for segment 0 ###
            # Step 1- Subsetting the df for Cluster 0
            cluster0 = data[data['Segment'] == 'Segment 0']

            # Plotting the histogram of age distribution in cluster0
            fig1 = px.histogram(cluster0, x = 'Income', nbins = 8, color_discrete_sequence = ['#8dbdff'])

            # Remove gridlines
            fig1.update_layout(xaxis=dict(showgrid=False),
                            yaxis=dict(showgrid=False))

            # Aligning title
            fig1.update_layout(
                title={
                    'text': "Income Distribution",
                    'y':0.99,
                    'x':0.49,
                    'xanchor': 'center',
                    'yanchor': 'top'})
            # Aligning title
            fig1.update_layout(
                title={
                    'text': "Income Distribution",
                    #'y':0.99,
                    'x':0.49,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            ### Barchart for segment 0 ###
            # Marital status
            divorced0 = cluster0['marital_Divorced'].sum()
            married0 = cluster0['marital_Married'].sum()
            single0 = cluster0['marital_Single'].sum()
            together0 = cluster0['marital_Together'].sum()
            widow0 = cluster0['marital_Widow'].sum()

            fig4 = go.Figure(go.Bar(
            x = [divorced0, married0, single0, together0, widow0],
            y = ['Divorced', 'Married', 'Single', 'Together', 'Widow'],
            marker_color = ['#ffb0d9', '#f9c9e2', '#8dbdff','#b3d4ff', '#d6eaff'],
            orientation = 'h'))

            # Remove gridlines
            fig4.update_layout(xaxis=dict(showgrid=False),
                          yaxis=dict(showgrid=False)
            )
            # Aligning title
            fig4.update_layout(
                title={
                    'text': "Marital Status",
                    #'y':0.99,
                    'x':0.49,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            ### Pie chart for segment 0 ###
            # Education
            secondcycle0 = cluster0['education_2n Cycle'].sum()
            basic0 = cluster0['education_Basic'].sum()
            graduation0 = cluster0['education_Graduation'].sum()
            master0 = cluster0['education_Master'].sum()
            phd0 = cluster0['education_PhD'].sum()

            data = {'education' : ['Second Cycle', 'High School', 'Bachelors', 'Masters', 'PhD'],
                    'count' : [secondcycle0, basic0, graduation0, master0, phd0]}
            df_education = pd.DataFrame(data)

            # pie chart
            fig2 = px.pie(df_education, values = 'count', names = 'education', color = 'education',
            color_discrete_sequence = ["#8dbdff", "#b3d4ff", "#d6eaff", '#ffb0d9', '#f9c9e2', '#f7e4f8'])

            # Aligning title
            fig2.update_layout(
                title={
                    'text': "Education Achievement",
                    #'y':0.99,
                    'x':0.49,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            ### distplot for segment 0 ###
            # Age distribution
            # Plot the distribution of R
            x = cluster0['Age']
            hist_data = [x]
            group_labels = ['cluster0']

            colors = ['#ffb0d9']
            fig3 = ff.create_distplot(hist_data, group_labels, bin_size=8, colors = colors)
            fig3.update_layout(title_text = 'Age Distribution')
            fig3.update_xaxes(showgrid=False)
            fig3.update_yaxes(showgrid=False)

            # Aligning title
            fig3.update_layout(
                title={
                    'text': "Age Distribution",
                    #'y':0.99,
                    'x':0.49,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            ### Bar chart for segment 0 ###
            # Number of dependents
            # Number of customers with kids
            df0_kid = cluster0[cluster0['Kidhome'] != 0]
            kid0 = df0_kid['Kidhome'].count()

            # Number of customers with teenagers
            df0_teen = cluster0[cluster0['Teenhome'] != 0]
            teen0 = df0_teen['Kidhome'].count()

            # Number of customers with no kids
            nodependents0 = sum((cluster0['Kidhome'] == 0) & (cluster0['Teenhome'] == 0))


            data = {'dependents' : ['Kids', 'Teens', 'No Dependents'],
                    'count' : [kid0, teen0, nodependents0]}
            df_dependents = pd.DataFrame(data)

            # bar chart
            fig6 = px.bar(df_dependents, x = 'dependents', y = 'count', color = 'dependents',
            color_discrete_sequence = ['#ffb0d9', '#f9c9e2', '#8dbdff'])

            # Remove gridlines
            fig6.update_layout(xaxis=dict(showgrid=False),
                          yaxis=dict(showgrid=False))
            # Aligning title
            fig6.update_layout(
                title={
                    'text': "Number of Dependents",
                    #'y':0.99,
                    'x':0.48,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            # Tenure ie. Customer_Days
            # changing customer days from continuous to discrete
            cluster0['Tenure'] = round(cluster0 ['Customer_Days']/365)
            cluster0 ['Tenure'] = cluster0 ['Tenure'].astype('int64')

            years = cluster0 ['Tenure'].unique().tolist()
            count = cluster0 ['Tenure'].value_counts().tolist()

            data = {'Tenure':years,
                    'count':count,
                    'segment':['Segment 0', 'Segment 0', 'Segment 0']}
            df_tenure = pd.DataFrame(data)

            fig5 = px.scatter(df_tenure, x="Tenure", y="count",
            	         size="count", hover_name="segment", log_x=True, size_max=60,
                         color = 'segment', color_discrete_sequence = ["#b3d4ff"])
            # Remove gridlines
            fig5.update_layout(xaxis=dict(showgrid=False),
                          yaxis=dict(showgrid=False))
            # Aligning title
            fig5.update_layout(
                title={
                    'text': "Customer Tenure",
                    #'y':0.99,
                    'x':0.48,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            # Plotting the charts
            col1, col2, col3 = st.columns(3)
            with col1:
                st.plotly_chart(fig1)
                st.plotly_chart(fig4)
            with col2:
                st.plotly_chart(fig2)
                st.plotly_chart(fig5)
            with col3:
                st.plotly_chart(fig3)
                st.plotly_chart(fig6)

        if option == 'Segment 1':
            ### Histogram for segment 1 ###
            # Subsetting the df for Cluster 1
            cluster1 = data[data['Segment'] == 'Segment 1']

            # Plotting the histogram of age distribution in cluster0
            fig1 = px.histogram(cluster1, x = 'Income', nbins = 8, color_discrete_sequence = ["#8dbdff"])

            # Remove gridlines
            fig1.update_layout(xaxis=dict(showgrid=False),
                            yaxis=dict(showgrid=False))

            # Aligning title
            fig1.update_layout(
                title={
                    'text': "Income Distribution",
                    'y':0.99,
                    'x':0.49,
                    'xanchor': 'center',
                    'yanchor': 'top'})
            # Aligning title
            fig1.update_layout(
                title={
                    'text': "Income Distribution",
                    #'y':0.99,
                    'x':0.49,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            ### Barchart for segment 1 ###
            # Marital status
            divorced1 = cluster1['marital_Divorced'].sum()
            married1 = cluster1['marital_Married'].sum()
            single1 = cluster1['marital_Single'].sum()
            together1 = cluster1['marital_Together'].sum()
            widow1 = cluster1['marital_Widow'].sum()

            fig4 = go.Figure(go.Bar(
            x = [divorced1, married1, single1, together1, widow1],
            y = ['Divorced', 'Married', 'Single', 'Together', 'Widow'],
            marker_color = ['#ffb0d9', '#f9c9e2', '#8dbdff','#b3d4ff', '#d6eaff'],
            orientation = 'h'))

            # Remove gridlines
            fig4.update_layout(xaxis=dict(showgrid=False),
                          yaxis=dict(showgrid=False)
            )
            # Aligning title
            fig4.update_layout(
                title={
                    'text': "Marital Status",
                    #'y':0.99,
                    'x':0.49,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            ### Pie chart for segment 1 ###
            # Education
            secondcycle1 = cluster1['education_2n Cycle'].sum()
            basic1 = cluster1 ['education_Basic'].sum()
            graduation1 = cluster1 ['education_Graduation'].sum()
            master1 = cluster1 ['education_Master'].sum()
            phd1 = cluster1 ['education_PhD'].sum()

            data = {'education' : ['Second Cycle', 'High School', 'Bachelors', 'Masters', 'PhD'],
                    'count' : [secondcycle1, basic1, graduation1, master1, phd1]}
            df_education = pd.DataFrame(data)

            # pie chart
            fig2 = px.pie(df_education, values = 'count', names = 'education', color = 'education',
            color_discrete_sequence = ["#8dbdff", "#b3d4ff", "#d6eaff", "#ffb0d9", "#f9c9e2", "#f7e4f8"])

            # Aligning title
            fig2.update_layout(
                title={
                    'text': "Education Achievement",
                    #'y':0.99,
                    'x':0.49,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            ### distplot for segment 1 ###
            # Age distribution
            # Plot the distribution of R
            x = cluster1['Age']
            hist_data = [x]
            group_labels = ['cluster1']

            colors = ['#ffb0d9']
            fig3 = ff.create_distplot(hist_data, group_labels, bin_size=8, colors = colors)
            fig3.update_layout(title_text = 'Age Distribution')
            fig3.update_xaxes(showgrid=False)
            fig3.update_yaxes(showgrid=False)

            # Aligning title
            fig3.update_layout(
                title={
                    'text': "Age Distribution",
                    #'y':0.99,
                    'x':0.49,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            ### Bar chart for segment 1 ###
            # Number of dependents
            # Number of customers with kids
            df1_kid = cluster1[cluster1['Kidhome'] != 0]
            kid1 = df1_kid['Kidhome'].count()

            # Number of customers with teenagers
            df1_teen = cluster1[cluster1['Teenhome'] != 0]
            teen1 = df1_teen['Kidhome'].count()

            # Number of customers with no kids
            nodependents1 = sum((cluster1['Kidhome'] == 0) & (cluster1['Teenhome'] == 0))

            data = {'dependents' : ['Kids', 'Teens', 'No Dependents'],
                    'count' : [kid1, teen1, nodependents1]}
            df_dependents = pd.DataFrame(data)

            # bar chart
            fig6 = px.bar(df_dependents, x = 'dependents', y = 'count', color = 'dependents',
            color_discrete_sequence = ['#ffb0d9', '#f9c9e2', '#8dbdff'])

            # Remove gridlines
            fig6.update_layout(xaxis=dict(showgrid=False),
                          yaxis=dict(showgrid=False))

            # Aligning title
            fig6.update_layout(
                title={
                    'text': "Number of Dependents",
                    #'y':0.99,
                    'x':0.48,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            # Tenure ie. Customer_Days
            # changing customer days from continuous to discrete
            cluster1['Tenure'] = round(cluster1['Customer_Days']/365)
            cluster1['Tenure'] = cluster1['Tenure'].astype('int64')

            years = cluster1['Tenure'].unique().tolist()
            count = cluster1['Tenure'].value_counts().tolist()

            data = {'Tenure':years,
                    'count':count,
                    'segment':['Segment 1', 'Segment 1', 'Segment 1']}
            df_tenure = pd.DataFrame(data)

            fig5 = px.scatter(df_tenure, x="Tenure", y="count",
            	         size="count", hover_name="segment", log_x=True, size_max=60,
                         color = 'segment', color_discrete_sequence = ['#b3d4ff'])
            # Remove gridlines
            fig5.update_layout(xaxis=dict(showgrid=False),
                          yaxis=dict(showgrid=False))

            # Aligning title
            fig5.update_layout(
                title={
                    'text': "Customer Tenure",
                    #'y':0.99,
                    'x':0.48,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            # Plotting the charts
            col1, col2, col3 = st.columns(3)
            with col1:
                st.plotly_chart(fig1)
                st.plotly_chart(fig4)
            with col2:
                st.plotly_chart(fig2)
                st.plotly_chart(fig5)
            with col3:
                st.plotly_chart(fig3)
                st.plotly_chart(fig6)

        if option == 'Segment 2':
            ### Histogram for segment 2 ###
            # Subsetting the df for Cluster 2
            cluster2 = data[data['Segment'] == 'Segment 2']

            # Plotting the histogram of age distribution in cluster0
            fig1 = px.histogram(cluster2, x = 'Income', nbins = 6, color_discrete_sequence = ["#8dbdff"])

            # Remove gridlines
            fig1.update_layout(xaxis=dict(showgrid=False),
                            yaxis=dict(showgrid=False))

            # Aligning title
            fig1.update_layout(
                title={
                    'text': "Income Distribution",
                    #'y':0.99,
                    'x':0.49,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            ### Barchart for segment 2 ###
            # Marital status
            divorced2 = cluster2['marital_Divorced'].sum()
            married2 = cluster2['marital_Married'].sum()
            single2 = cluster2['marital_Single'].sum()
            together2 = cluster2['marital_Together'].sum()
            widow2 = cluster2['marital_Widow'].sum()

            fig4 = go.Figure(go.Bar(
            x = [divorced2, married2, single2, together2, widow2],
            y = ['Divorced', 'Married', 'Single', 'Together', 'Widow'],
            marker_color = ['#ffb0d9', '#f9c9e2', '#8dbdff','#b3d4ff', '#d6eaff'],
            orientation = 'h'))

            # Remove gridlines
            fig4.update_layout(xaxis=dict(showgrid=False),
                          yaxis=dict(showgrid=False)
            )

            # Aligning title
            fig4.update_layout(
                title={
                    'text': "Marital Status",
                    #'y':0.99,
                    'x':0.49,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            ### Pie chart for segment 2 ###
            # Education
            secondcycle2 = cluster2['education_2n Cycle'].sum()
            basic2 = cluster2['education_Basic'].sum()
            graduation2 = cluster2['education_Graduation'].sum()
            master2 = cluster2['education_Master'].sum()
            phd2 = cluster2['education_PhD'].sum()

            data = {'education' : ['Second Cycle', 'High School', 'Bachelors', 'Masters', 'PhD'],
                    'count' : [secondcycle2, basic2, graduation2, master2, phd2]}
            df_education = pd.DataFrame(data)

            # pie chart
            fig2 = px.pie(df_education, values = 'count', names = 'education', color = 'education',
                            color_discrete_sequence = ["#8dbdff", "#b3d4ff", "#d6eaff", '#ffb0d9', '#f9c9e2', '#f7e4f8'])

            # Aligning title
            fig2.update_layout(
                title={
                    'text': "Education Achievement",
                    #'y':0.99,
                    'x':0.49,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            ### distplot for segment 2 ###
            # Age distribution
            # Plot the distribution of R
            x = cluster2['Age']
            hist_data = [x]
            group_labels = ['cluster2']

            colors = ['#ffb0d9']
            fig3 = ff.create_distplot(hist_data, group_labels, bin_size=8, colors = colors)
            fig3.update_layout(title_text = 'Age Distribution')
            fig3.update_xaxes(showgrid=False)
            fig3.update_yaxes(showgrid=False)

            # Aligning title
            fig3.update_layout(
                title={
                    'text': "Age Distribution",
                    #'y':0.99,
                    'x':0.49,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            ### Bar chart for segment 2 ###
            # Number of dependents
            # Number of customers with kids
            df2_kid = cluster2[cluster2['Kidhome'] != 0]
            kid2 = df2_kid['Kidhome'].count()

            # Number of customers with teenagers
            df2_teen = cluster2[cluster2['Teenhome'] != 0]
            teen2 = df2_teen['Kidhome'].count()

            # Number of customers with no kids
            nodependents2 = sum((cluster2['Kidhome'] == 0) & (cluster2['Teenhome'] == 0))

            data = {'dependents' : ['Kids', 'Teens', 'No Dependents'],
                    'count' : [kid2, teen2, nodependents2]}
            df_dependents = pd.DataFrame(data)

            # bar chart
            fig6 = px.bar(df_dependents, x = 'dependents', y = 'count', color = 'dependents',
            color_discrete_sequence = ['#ffb0d9', '#f9c9e2', '#8dbdff'])

            # Remove gridlines
            fig6.update_layout(xaxis=dict(showgrid=False),
                          yaxis=dict(showgrid=False))

            # Aligning title
            fig6.update_layout(
                title={
                    'text': "Number of Dependents",
                    #'y':0.99,
                    'x':0.48,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            # Tenure ie. Customer_Days
            # changing customer days from continuous to discrete
            cluster2['Tenure'] = round(cluster2['Customer_Days']/365)
            cluster2['Tenure'] = cluster2['Tenure'].astype('int64')

            years = cluster2['Tenure'].unique().tolist()
            count = cluster2['Tenure'].value_counts().tolist()

            data = {'Tenure':years,
                    'count':count,
                    'segment':['Segment 2', 'Segment 2', 'Segment 2']}
            df_tenure = pd.DataFrame(data)

            fig5 = px.scatter(df_tenure, x="Tenure", y="count",
            	         size="count", hover_name="segment", log_x=True, size_max=60,
                         color = 'segment', color_discrete_sequence = ["#b3d4ff"])
            # Remove gridlines
            fig5.update_layout(xaxis=dict(showgrid=False),
                          yaxis=dict(showgrid=False))

            # Aligning title
            fig5.update_layout(
                title={
                    'text': "Customer Tenure",
                    #'y':0.99,
                    'x':0.48,
                    'xanchor': 'center',
                    'yanchor': 'top'})

            # Plotting the charts
            col1, col2, col3 = st.columns(3)
            with col1:
                st.plotly_chart(fig1)
                st.plotly_chart(fig4)
            with col2:
                st.plotly_chart(fig2)
                st.plotly_chart(fig5)
            with col3:
                st.plotly_chart(fig3)
                st.plotly_chart(fig6)
