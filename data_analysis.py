"""An example of showing geographic data."""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
from PIL import Image

def data_analysis():
    st.title("Data Analysis - Job placements recommender for google ads")
    st.subheader("The goal is to build a recommender system that given a job title recommends N top placements to advertise the job")
    st.markdown("<br>", unsafe_allow_html=True)

    @st.cache()
    def load_data():
        data = pd.read_csv("./merged_dataset_3_types.csv", na_filter= False) 
        lowercase = lambda x: str(x).lower()
        data.rename(lowercase, axis="columns", inplace=True)
        # data = data.drop(columns=['avg. cost','currency code','german','international','foreign'])
        return data

    @st.cache()
    def first_image():
        im = Image.open('old_lady_example.jpeg')
        return im

    @st.cache()
    def second_image():
        im = Image.open('Graphic method difference.png')
        return im

    data = load_data()

    def top_sites(job_type,placement_type):
        numerical_att = ['avg. cpc','impr.','clicks','ctr','interactions','viewable impr.','non-viewable impr.','measurable impr.','non-measurable impr.']
        shared_sites = data[(data['type']==placement_type)&(data['job title']==job_type)].pivot_table(values = numerical_att,index='placement',aggfunc = ['sum','mean','size'])
        top = shared_sites.reindex(shared_sites.sort_values(by=('size',0),ascending=False).index).round(4)
        return top

    #-------------------------------------------------------------------#
    #----------------------------   Intro   ----------------------------#
    #-------------------------------------------------------------------#
    st.markdown("But first, lets state what exactly a *recommender system* is and some details from them.")
    st.subheader("Recommender Systems")
    st.write("We can see recommendation systems everywhere, and they are differentiators from competitors if done right.")
    st.image(first_image(),use_column_width=True)
    st.write("There are two main groups of recommendation systems, the collaborative one and the content based. Collaborative methods are based entirely on past observations of how the user interacted with the content (user-item interactions matrix). Content based methods aim to build a model, based on the available information from the users and items, that explain the observed user-item interactions.")
    st.write("**Collaborative algorithms**")
    st.write("The main Idea in collaborative methods is to detect similar users or items and estimate proximities. Is divided in memory based and model based. They rely on user interactions, so the quality of the recommendation grows with more samples.")
    st.write("Memory based algorithms  just look for neighbors (give the most popular movies back that share characteristics with the one that was just watched), while model based algorithms explains how the behavior of user-item is made and predicts following this model.")
    st.markdown("""> * **User-user:** It is based on the search of similar users in terms of interactions with items in the user-item interaction matrix. This method is more personalized.
    > * **Item-item:** It is based on the search of similar items in terms of user-item interactions. The advantage of the item-based approach is that item similarity is more stable and can be efficiently pre-computed.
    > * **Matrix factorisation:** It decomposes the user-item interaction matrix into a reconstructed interaction matrix (dot product of the user matrix and the transposed matrix) plus a reconstruction error (matrix). Saves data file size.""")
    st.image(second_image(),use_column_width=True)

    st.write("**Content based methods**")
    st.write("A drawback from collaborative systems is that they suffer a “cold star” because they don’t have user-item interaction to start with, while content based can infer something based on the user or item features. Just with users/items that show a new feature set, this drawback is present.")
    st.write("Content based methods define a model for user-item interactions where users and/or items representations are given. We also have feedback data that could be likes or ratings (explicit) or the interaction rate like if they skipped the video, close the ad (implicit).")
    st.markdown("""> * **Item-centered:** When based on user features. A model by item based on users features trying to answer the question “what is the probability for each user to like this item?
    > * **User-centered:** When based on the item features. A model by user based on items features that tries to answer the question “what is the probability for this user to like each item?”""")
    st.markdown("""<br>""",unsafe_allow_html=True)


    #-------------------------------------------------------------------#
    #----------------------------   Step 1  ----------------------------#
    #-------------------------------------------------------------------#
    st.subheader("Data analysys")
    st.markdown("""**Step 1** We downloaded performance data from google ads for previus campaigns. The job title selected \nwere *Application Engineer* and *Bus driver*. Here is a small take of the downloaded dataset:""")

    st.write(data[100:800])

    st.write("")
    st.write("")
    st.write("""We found a total of """, data.shape[0], " placements in the dataset. Placements are classified into 3 categories: *youtube channel*, *site* and *mobile application*. The distribution of the data by job title is about 60/40 which...")


    patches, texts, autotexts = plt.pie(data['job title'].value_counts(), explode=(0.05, 0.05,0.05), colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99'],labels=['Busfahrer','Applications Engineer','Nurse'], pctdistance=0.85,autopct='%1.1f%%')

    for autotext in autotexts:
        autotext.set_color('dimgray')
    st.pyplot()

    #-------------------------------------------------------------------#
    #----------------------------   Step 2  ----------------------------#
    #-------------------------------------------------------------------#
    st.markdown("""**Step 2** Lets find out more about placements! Lets display the top 100 placement types for the different job titles and see if can observe some patterns""")

    jobtype = st.selectbox(
        'What job title do you want to explore',
        data["job title"].unique() )

    placement_type = st.selectbox(
        'What type of placement do you want to know more',
        data["type"].unique() )

    st.subheader(f"Top 100 *{placement_type}s* for *{jobtype}*")
    st.write(top_sites(jobtype,placement_type)[0:100])

    st.write("""Some observations:""")
    st.markdown("""> * Youtube channel names like BibisBeautyPalace shows that we are showing the ads on kids content. 
    > * More information about the placement is needed in order to discover patterns: category of the mobile application (is it an app or a game?).
    > * Eventually figuring out if the site/mobile/youtube is female or male oriented to help the algorithm match better (we saw *Germanys next top model* in the top youtube channels for application engineers which is questionable)
    """)

    #-------------------------------------------------------------------#
    #----------------------------   Step 3  ----------------------------#
    #-------------------------------------------------------------------#
    st.write("")
    st.write("")

    st.markdown("""**Step 3** Now lets look at the distributions of the attributes. Are they concentrated in a region? Is the distribution alike for different categories? Should we observe just certain sample of the placements?""")
    numeric_att = data.columns.drop(['placement','placement url','type','campaign','campaign type','ad group','job title']) #domain origin
    categorical_att = ['type','job title','domain origin','campaign type']
    att_usable = ['avg. cpc','impr.','clicks','ctr','interactions','viewable impr.','measurable impr.']

    att = st.selectbox(
        'Which attribute would you like to view?',
        numeric_att
    )

    grouping = st.selectbox(
        'Which category do you want to group by?',
        categorical_att
    )

    quantile = st.slider(
        'Select range of att data to show. (Will plot data that ranges from the lower dot quantile until the higher dot quantile',
        0,100,(0,80),step=1,format='%d%%'
    )
    st.subheader(f"Histogram of *{att}* from *{quantile[0]}*% until *{quantile[1]}*% of the total placements")
    fig = px.histogram(data[(data[att]<=data[att].quantile(quantile[1]/100))&(data[att]>=data[att].quantile(quantile[0]/100))],x=att,color=grouping)
    st.plotly_chart(fig)

    st.write("""Some observations:
    > * Some distributions are realy condensated in a small range with some placements with anomral behavior. The range selector is used to chunk cutting the n% of data at the beggining and also the data after the n% of the points. 
    > * Some attributes seem to have 2 decreases in the lower ranges (Impr., Clicks, etc.).
    > * Impressions also seem to behave alike no matter what job type the ad is for.
    """)

    #-------------------------------------------------------------------#
    #----------------------------   Step 4  ----------------------------#
    #-------------------------------------------------------------------#
    st.write("")
    st.write("")

    st.markdown("""**Step 4** Now we will take a look at the numerical attributes of the dataset. In particular *Impressions*, *Clicks*, *CTR*, *Interactions*, *Interactions rate*, *Viewable impressions* and *non-viewable impressions*.""")
    st.markdown(""" The goal is to find outliers and eventually patterns and features. Let's start with the distribution of these attributes:""")

    column1 = st.selectbox(
        'Select the attribute for the X axis',
        numeric_att )

    column2 = st.selectbox(
        'Select the attribute for the Y axis',
        numeric_att, index = 5 )

    color_type = st.selectbox(
        'Group the results by placement type or job title',
        ["job title", "type"], index = 0 )

    fig = px.scatter(data, x=column1, y=column2, color=color_type,title=f"Here we have {column2} against {column1} grouped by {color_type}.")
    st.plotly_chart(fig)

    st.markdown("Some observations:")
    st.markdown("""> * Points that seem far from the bigger dot concentration may be outliers that we'll need to decide wether or not to keep them.
    > * Some attributes seem to have high correlation and may be redundant, also something we should evaluate
    """)

    #st.write(data[(data['type']==placement_type)&(data['job title']==job_type)]).sort_values(by=[column2],ascending = False)


    #-------------------------------------------------------------------#
    #----------------------------   Step 5  ----------------------------#
    #-------------------------------------------------------------------#

    st.write("")
    st.write("")

    st.markdown("""**Step 5** Now we will take a look at some categorical attributes that we have """)
    st.markdown(""" The goal is to find patterns or behaviors from groups when analysed separatedly:""")

    st.markdown(""" First lets take a look at the placements grouped by these attributes""")

    col = st.selectbox(
        'Select the categorical attribute you want to explore',
        (categorical_att), index = 1
    )

    hue = st.selectbox(
        'Select grouping category.',
        categorical_att, index = 0
    )

    st.subheader(f"Count of *{col}s* in each *{hue}* category")
    sns.countplot(data[col],hue=hue,data=data)
    st.pyplot()

    st.write("""Some observations:
    > * The placements type distribution seems to be similar for both job categories.
    > * Currently just categorizing by domain URL, more information could be retrieved from the YouTube channels.
    > * Busdrivers rely more on foreign sites than applications engineers maybe because of the procedence of the target population.
    > * Still need to transfer this plot to plotly (note for developers)
    """)


    #-------------------------------------------------------------------#
    #----------------------------   Step 6  ----------------------------#
    #-------------------------------------------------------------------#

    st.write("")
    st.write("")

    st.markdown("""**Step 6** Here we are going to visualize correlations. """)
    st.markdown(""" The objective here is to determine which attributes are already described by other attributes to reduce the feature set:""")
    #df_cleaned.fillna(0,inplace=True)
    att_corr = st.multiselect(
        label = 'Which variable would you like to plot in the correlation matrix?',
        options = numeric_att.to_list(), 
        default = att_usable
    )

    corr = data[att_corr].corr()
    sns.heatmap(corr, 
            xticklabels=corr.columns,
            yticklabels=corr.columns,
            cmap=sns.color_palette("Blues"),
            annot=True)
    st.subheader(f"Correlation matrix of the selected features")
    st.pyplot()
    st.write("""Some observations:
    > * Non viewable impressions have negative correlation with CTR.
    > * Clicks and Interactions are the same.
    > * Cost is more correlated with Clicks than Impr.
    > * Most of the Impr. related metrics are and should be correlated.
    > * Pending to transfer to pyplot and have features to plot grabbed from a selection pane. (Developer note)
    """)

    #-------------------------------------------------------------------#
    #----------------------------  Playbox  ----------------------------#
    #-------------------------------------------------------------------#

    st.write("")
    st.write("")

    st.markdown("""**Playbox** Just to plot everything against everything. The goal is to explore:""")
    col1 = st.selectbox(
        'Select an attribute for the X axis',
        data.columns.unique(),
        index=10)

    col2 = st.selectbox(
        'Select an attribute for the Y axis',
        data.columns.unique(),
        index=14)

    color_type = st.selectbox(
        'Group the results by placement type or job title',
        ["job title", "type","domain origin"] )

    st.subheader(f"Here we have {col2} against {col1} grouped by {color_type}.")
    fig = px.scatter(data, x=col1, y=col2, color=color_type)
    st.plotly_chart(fig)
    st.markdown(""" Again, have some fun or satisfy your curiosity.""")
    place_type = st.selectbox(
        'Choose platform',
        ["YouTube channel", "Mobile application",'Site'] 
    )

    job_title = st.selectbox(
        'Choose job type',
        ["Busfahrer", "Applications Engineer","Nurse"] 
    )

    attribute = st.selectbox(
        'Select attribute to sort data',
        numeric_att )
    st.subheader("")
    st.subheader(f"Here is a list of placements orderered by *{att}* from highest to lowest. (Only show top 600 results)")
    st.write(data[(data['type']==place_type)&(data['job title']==job_title)].sort_values(by=[attribute],ascending = False)[:600])

    print('data_analysis.py succesfully executed')
    return