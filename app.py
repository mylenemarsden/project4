# Import dependencies
import pandas as pd
from dash import Dash, Input, Output, dcc, html
import requests
import plotly.express as px
import seaborn as sns
import plotly.graph_objects as go
from plotly.data import tips
import plotly.figure_factory as ff
import numpy as np
import sys
import base64       
import pandas as pd

# Start with loading all necessary libraries
import numpy as np
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from textblob import TextBlob   
import matplotlib.pyplot as plt
from dash_holoniq_wordcloud import DashWordcloud

import io
from io import BytesIO

# Making an HTTP request to the api
api_url = "http://127.0.0.1:5000/api/v1/resources/wine/all"
response = requests.get(api_url)

# Reading the response into the dataframe
data = pd.read_json(response.json())

sentiment=["Positive","Negative","Neutral"]

country = list(filter(None, data['country'].sort_values().unique() )) 
## create function to subjectivity
def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity
## create function to get polarity
def getPolarity(text):
    return TextBlob(text).sentiment.polarity

# Create function to see if a score means the sentiment is negative, neutral, or positive
def getAnalysis(score):
    if score<0:
        return 'Negative'
    elif score==0:
        return 'Neutral'
    else:
        return 'Positive'
    
# Applying the functions on the dataframe
data['Subjectivity']=data['description'].apply(getSubjectivity)
data['Polarity']=data['description'].apply(getPolarity)
data['Analysis'] = data['Polarity'].apply(getAnalysis)

data.reset_index()

external_stylesheets = [
    {
        "href": (
            "https://fonts.googleapis.com/css2?"
            "family=Lato:wght@400;700&display=swap"
        ),
        "rel": "stylesheet",
    },
]
app = Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Sentiment Analysis"

# Creating the wordcloud
def generate_wordcloud_data(data):
    text = " ".join(review for review in data.description)
    wordcloud = WordCloud(stopwords = STOPWORDS,
                        collocations=True).generate(text)
    text_dictionary = wordcloud.process_text(text)
    # Sort the dictionary
    word_freq={k: v for k, v in sorted(text_dictionary.items(),reverse=True, key=lambda item: item[1])}
    # Use words_ to print relative word frequencies
    rel_freq=wordcloud.words_

    N = 100
    sorted_dictionary = dict(sorted(word_freq.items(), key = lambda x: x[1], reverse = True)[:N])
    results=list(map(list, sorted_dictionary.items()))
    return results

# Creating a list of positive words
word_cloud_items=generate_wordcloud_data(data.loc[data["Analysis"]=="Positive"])

# Getting the top 10 countries
summary = data.groupby('country').agg({'Analysis':'size', 'Polarity':'mean'}).rename(columns={'Analysis':'count','mean':'mean_sent'}).reset_index()
summary_sanitized_data=summary[summary['count'].ge(1)].sort_values(by=['count'],  ascending=False).head(10)
# Creating the bar figure for the top 10 highest wine consumer per country
fig_bar_line_count = go.Figure(
    data=go.Bar(
        x=summary_sanitized_data["country"].values,
        y=summary_sanitized_data["count"].values,
        name="Top 10 highest wine consumer per Country",
        marker=dict(color="#1e81b0"),
        )
    )
# Creating the bar figure for the top 10 highest wine positive review per country
summary_sanitized_data=summary[summary['count'].ge(1)].sort_values(by=['Polarity'],  ascending=False).head(10)
fig_bar_line_polarity = go.Figure(
    data=go.Bar(
        x=summary_sanitized_data["country"].values,
        y=summary_sanitized_data["Polarity"].values,
        name="Top 10 highest wine positive review per Country",
        marker=dict(color="#1e81b0"),
        )
    )
print(summary_sanitized_data)

# Defines the layout on the page
app.layout = html.Div(
    children=[
        html.Div(
            children=[
                html.H1(
                    children="Sentiment Analysis", className="header-title"
                ),
                html.P(
                    children=(
                        "Analyze the wine worldwide from" 
                        " 1000 random samples from Kaggle"
                    ),
                    className="header-description",
                ),
            ],
            className="header",
        ),

        # Creating the filter
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.Div(children="Country", className="menu-title"),
                        dcc.Dropdown(
                            id="country-filter",
                            options=[
                                {"label": company, "value": company}
                                for company in country
                            ],
                            value="All",
                            clearable=False,
                            className="dropdown",
                        ),
                    ],
                ),
            ],
            className="menu",
        ),

        html.Div(
            children=[

                html.H3('Sentiment per Country'),
                html.Div(
                    children=dcc.Graph(
                        id="price-chart-figure",
                        config={"displayModeBar": False},
                    ),
                    className="card",
                ),

                html.H3('Subjectivity per Polarity'),
                html.Div(
                    children=dcc.Graph(
                        id="scatter-plot-figure",
                        config={"displayModeBar": False},
                    ),
                     className="card",
                ),

                html.H3('Top 10 highest wine consumer per Country'),
                html.Div(   
                    children=dcc.Graph(
                        id="top-ten-table-figure",
                        config={"displayModeBar": False},
                        figure=fig_bar_line_count,
                    ),
                     className="card",
                ),

                html.H3('Top 10 highest positive review per Country'),
                html.Div(   
                    children=dcc.Graph(
                        id="top-ten-table-figure",
                        config={"displayModeBar": False},   
                        figure=fig_bar_line_polarity,
                    ),
                     className="card",
                ),

                html.H3('Wordcloud of Sentiments'),
                html.Div([  
                 html.Div([
                    DashWordcloud(
                        list=generate_wordcloud_data(data.loc[data["Analysis"]=="Positive"]),
                        width=1024, height=450,
                        gridSize=16,
                        shuffle=False,
                        rotateRatio=0.5,    
                        shrinkToFit=True,
                        shape='circle',
                        hover=True,
                        )
                    ]),
                ],className="card",),
            ],
            className="wrapper",
        ),
    ]
)


@app.callback(
    Output("price-chart-figure", "figure"), 
    Output("scatter-plot-figure", "figure"),

    Input("country-filter", "value"),
)
    

# Create function to update the figures according to the country filter chosen by the viewer
def update_charts(country):
    global word_cloud_items
    print("workyear: " + str(sentiment))
    print("country: " + str(country))

    if str(country) == "All":
        filtered_data=data
    else:
        filtered_data = data.query("country == @country")

    word_cloud_items= generate_wordcloud_data(filtered_data)

    summary = filtered_data.groupby('Analysis').agg({'Analysis':'size', 'Polarity':'mean'}).rename(columns={'Analysis':'count','mean':'mean_sent'}).reset_index()

    fig_bar_line = go.Figure(
    data=go.Bar(
        x=summary["Analysis"].values,
        y=summary["count"].values,
        name="Count Per Sentiment",
        marker=dict(color="#1e81b0"),
        )
    )

    fig_scatter_plot = px.scatter(filtered_data, x="Polarity", y="Subjectivity", color="Analysis",
                 size='Subjectivity', hover_data=['Polarity'])

    return fig_bar_line, fig_scatter_plot

if __name__ == "__main__":
    app.run_server(debug=True)