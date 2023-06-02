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
    # Import dependencies
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
api_url = "http://127.0.0.1:5000/api/v1/resources/wine/all"
response = requests.get(api_url)

data = pd.read_json(response.json())

sentiment=["Positive","Negative","Neutral"]


country = list(filter(None, data['country'].sort_values().unique() )) 
## create function to subjectivity
def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity
## create function to get polarity
def getPolarity(text):
    return TextBlob(text).sentiment.polarity

def getAnalysis(score):
    if score<0:
        return 'Negative'
    elif score==0:
        return 'Neutral'
    else:
        return 'Positive'
    
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

def generate_wordcloud_data(data):
    text = " ".join(review for review in data.description)
    wordcloud = WordCloud(stopwords = STOPWORDS,
                        collocations=True).generate(text)
    text_dictionary = wordcloud.process_text(text)
    # sort the dictionary
    word_freq={k: v for k, v in sorted(text_dictionary.items(),reverse=True, key=lambda item: item[1])}
    #use words_ to print relative word frequencies
    rel_freq=wordcloud.words_

    N = 100
    sorted_dictionary = dict(sorted(word_freq.items(), key = lambda x: x[1], reverse = True)[:N])
    results=list(map(list, sorted_dictionary.items()))
    return results

word_cloud_items=generate_wordcloud_data(data.loc[data["Analysis"]=="Positive"])

summary = data.groupby('country').agg({'Analysis':'size', 'Polarity':'mean'}).rename(columns={'Analysis':'count','mean':'mean_sent'}).reset_index()
summary_sanitized_data=summary[summary['count'].ge(1)].sort_values(by=['count'],  ascending=False).head(10)

fig_bar_line_count = go.Figure(
    data=go.Bar(
        x=summary_sanitized_data["country"].values,
        y=summary_sanitized_data["count"].values,
        name="Top 10 highest wine consumer per Country",
        marker=dict(color="#1e81b0"),
        )
    )

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
    # Output("top-ten-chart-figure", "figure"),
    # Output("top-ten-table-figure", "figure"),

    Input("country-filter", "value"),
 
)

    

    
def update_charts(country):
    global word_cloud_items
    print("workyear: " + str(sentiment))
    print("country: " + str(country))

    if str(country) == "All":
        filtered_data=data
    else:
        filtered_data = data.query("country == @country")

    word_cloud_items= generate_wordcloud_data(filtered_data)

    print(filtered_data)

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
    
    #fig_bar_line = px.bar(analysis, x='Analysis', y='count')
    # # print(filtered_data)
    # text = " ".join(review for review in filtered_data.description)

    # job_cloud= filtered_data.groupby(["description"])["description"].count().reset_index(name="count")
    # print("job_cloud")
    # print(job_cloud)
    # print(newGraph)
    # stopwords = set(STOPWORDS)
    # stopwords.update(["drink", "now", "wine", "flavor", "flavors"])
    # wordcloud = WordCloud(stopwords=stopwords,height=500, width=500, background_color="white", contour_color='white', colormap="magma").generate(newGraph)
    # buf = io.BytesIO() # in-memory files
    # plt.figure()
    # plt.imshow(wordcloud, interpolation="bilinear")
    # plt.axis("off")
    # plt.savefig(buf, format = "png", dpi=600, bbox_inches = 'tight', pad_inches = 0) # save to the above file object
    # data = base64.b64encode(buf.getbuffer()).decode("utf8") # encode to html elements
    # plt.close()
    # fig_pie_chart = px.pie(filtered_data, values='salary_in_usd', names='work_type', title='Salary per Work Year', color_discrete_sequence=px.colors.sequential.RdBu_r)

    # #eab676
    # remote_ratio_plot = go.Figure()
    # remote_ratio_plot.add_trace(go.Box(y=filtered_data['salary_in_usd'][(filtered_data['remote_ratio'] == 0)],  fillcolor="#042f66",marker=dict(color='#042f66'), quartilemethod="linear", name="No Remote Work (<20%)" ))
    # remote_ratio_plot.add_trace(go.Box(y=filtered_data['salary_in_usd'][(filtered_data['remote_ratio'] == 50)], fillcolor="#eab676",marker=dict(color='#eab676'), quartilemethod="inclusive", name="Partially remote (~50%)"))
    # remote_ratio_plot.add_trace(go.Box(y=filtered_data['salary_in_usd'][(filtered_data['remote_ratio'] == 100)], fillcolor="#e28743",marker=dict(color='#e28743'), quartilemethod="exclusive", name="Fully remote (>80%)"))
    # remote_ratio_plot.update_traces(boxpoints='all', jitter=0)
    # remote_ratio_plot.update_layout(title=dict(text="Media Salary per Work Type"))

    # summary = filtered_data.groupby('job_title').agg({'job_title':'size', 'salary_in_usd':'mean'}).rename(columns={'job_title':'count','mean':'mean_sent'}).reset_index()
    # sanitized_data=summary.sort_values(by=['salary_in_usd'],  ascending=False).head(10)

    # count=sanitized_data["count"].values
    # job_title=sanitized_data["job_title"].values
    # salary_in_usd=sanitized_data["salary_in_usd"].values

    # sanitized_data.rename(columns = {'count':'Count', 'job_title':'Job Description','salary_in_usd':'Salary'}, inplace = True)
    # fig_table_data=ff.create_table(sanitized_data.round(2), height_constant=60, colorscale=px.colors.sequential.Blues_r,)

    # fig_bar_line = go.Figure(
    # data=go.Bar(
    #     x=job_title,
    #     y=count,
    #     name="Count of Jobs",
    #     marker=dict(color="#1e81b0"),
    #     )
    # )

    # fig_bar_line.add_trace(
    # go.Scatter(
    #     x=job_title,
    #     y=salary_in_usd,
    #     yaxis="y2",
    #     name="Average Salary",
    #     marker=dict(color="crimson"),
    #     )
    # )

    # fig_bar_line.update_layout(
    # legend=dict(orientation="v"),
    # yaxis=dict(
    #     title=dict(text="Count of Jobs"),
    #     side="left",
    # ),
    # yaxis2=dict(
    #     title=dict(text="Average Salary"),
    #     side="right",
    #     overlaying="y",
    #     tickmode="sync",
    #     ),
    #     title=dict(text="Top 10 Jobs")
    # )
    # fig_pie_chart=[]
    # remote_ratio_plot=[],
    # fig_bar_line=[],
    # fig_table_data=[]
    # return fig_pie_chart,remote_ratio_plot,fig_bar_line,fig_table_data
    # text = " ".join(review for review in filtered_data.description)
    # # Create stopword list:
    # stopwords = set(STOPWORDS)
    # stopwords.update(["drink", "now", "wine", "flavor", "flavors"])

    # Generate a word cloud image
    # wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)

    # Display the generated image:
    # the matplotlib way:
    # plt.imshow(wordcloud, interpolation='bilinear')
    # fig_pie_chart=plt.axis("off")

    # newGraph =  [value[0]:value[1]]
    #print(newGraph)
    # wordcloud = WordCloud(height=500, width=500, background_color="white", contour_color='white', colormap="magma").generate(text)
    # buf = io.BytesIO() # in-memory files
    # plt.figure()
    # plt.imshow(wordcloud, interpolation="bilinear")
    # plt.axis("off")
    # plt.savefig(buf, format = "png", dpi=600, bbox_inches = 'tight', pad_inches = 0) # save to the above file object
    # data_ = base64.b64encode(buf.getbuffer()).decode("utf8") # encode to html elements
    # plt.close()
    
    # stopwords = set(STOPWORDS)
    # stopwords.update(["drink", "now", "wine", "flavor", "flavors"])

    # # Generate a word cloud image
    # wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
    # buf = io.BytesIO() # in-memory files
    # # Display the generated image:
    # # the matplotlib way:
    # plt.imshow(wordcloud, interpolation='bilinear')
    # plt.axis("off")
    # plt.savefig(buf, format = "png", dpi=600, bbox_inches = 'tight', pad_inches = 0) # save to the above file object
    # data_ = base64.b64encode(buf.getbuffer()).decode("utf8") # encode to html elements
    # plt.close()

    # security_data=text
    #return "data:image/png;base64,{}".format(data_)
    return fig_bar_line, fig_scatter_plot

# def plot_wordcloud(data):
#     d = {a: x for a, x in data.values}
#     wc = WordCloud(background_color='black', width=480, height=360)
#     wc.fit_words(d)
#     return wc.to_image()

# @app.callback(dd.Output('image_wc', 'src'), [dd.Input('image_wc', 'id')])
# def make_image(b):
#     img = BytesIO()
#     plot_wordcloud(data=dfm).save(img, format='PNG')
#     return 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())



if __name__ == "__main__":
    app.run_server(debug=True)