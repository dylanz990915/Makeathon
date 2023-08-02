import nltk
nltk.download('stopwords')
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
from nltk.corpus import stopwords
import plotly.express as px
from collections import Counter
from nltk import ngrams
from wordcloud import WordCloud, STOPWORDS
from sklearn.cluster import KMeans
import base64
import io
import matplotlib.pyplot as plt

class PlotPaper():
    def __init__(self) -> None:
        self.filter_year = None
        try:
            self.all_stopwords = stopwords.words('english') + ['[sep]', 'results', 'show', 'using', 'two', 'different', 'also','new','control', 'find','study','use','however,','approach', 'method','used','may',  'due', 'using']
        except:
            nltk.download('stopwords')
            self.all_stopwords = stopwords.words('english') + ['[sep]', 'results', 'show', 'using', 'two', 'different', 'also','new','control', 'find','study','use','however,','approach', 'method','used','may',  'due', 'using']

        st.session_state.bt_plot = False
        
        self.df = PlotPaper.init_data()

        if 'prev_key'  not in st.session_state:
            st.session_state.prev_key = None
        if 'disp_wc'  not in st.session_state:
            st.session_state.disp_wc = None
        if 'search_res'  not in st.session_state:
            st.session_state.search_res = []
        if 'container_data' not in st.session_state:
            st.session_state.container_data = {}

        if 'display_df' not in st.session_state:
            st.session_state.display_df = display_df = pd.DataFrame()

        if 'word_cloud_data' not in st.session_state:
            st.session_state.word_cloud_data = None


        self.years = self.df.join_year.unique()
        self.positions = self.df['Job Title'].unique().tolist()
        self.selected_position = None  
        self.orgs = self.df.Org.unique().tolist()
        
        self.col1, self.col2, self.col3 = st.columns([7, 2, 2]) 

        with self.col3:
            self.selected_position = st.selectbox("Filter by Position", ["All"] + self.positions)
            self.selected_org = st.selectbox("Filter by Org", ["All"] + self.orgs)
            if self.selected_position != "All" or self.selected_org !="All":
                self.df = self.position_filter(self.selected_position,self.selected_org)
            
                

        with self.col2:
            st.session_state.key = st.text_input("Enter the keyword", placeholder = "Search")
            st.caption("Use max two words")
            st.session_state.bt = st.button("Search")
            st.caption("Word Cloud of Pinpoyee Characteristics!")

            if len(st.session_state.key.split()) > 2 and st.session_state.bt :
                st.error("Entered more than two words. Please enter a valid input.")
            
            else:
                if st.session_state.bt or st.session_state.key or st.session_state.bt:
                        st.session_state.search_res = self.search(st.session_state.key, mode = 'search')
            
            if len(st.session_state.search_res)>0 and (st.button("Export Search Results")):
                self.export_data(st.session_state.search_res, file_name="search_results.csv")
                        
            

        with self.col1:
            if not st.session_state.bt and not st.session_state.key:
                self.tab1, self.tab2 = st.tabs(["Pinterest Tenure!","Pintastic-Clusters"])
                
                with self.tab1:
                    if not st.session_state.bt and not st.session_state.key:
                        self.year_filter_graph()
                    if len(st.session_state.display_df)>0 and st.button("Export Tenure Data"):
                        self.export_data(st.session_state.display_df, file_name="tenure_data.csv")
                with self.tab2:
                    if not st.session_state.bt and not st.session_state.key:
                        self.KMeans_slider()
                    if len(st.session_state.display_df)>0 and st.button("Export Cluster Data"):
                        self.export_data(st.session_state.display_df, file_name="cluster_data.csv")

            

    def position_filter(self, position, org):
        self.df=pd.read_csv('data_6.csv')
        if position == "All" and org == 'All':
            return self.df
        elif position == "All" and org != 'All':
            return self.df[self.df['Org'] == org]
        elif position != "All" and org == 'All':
            return self.df[self.df['Job Title'] == position]
        else:
            return self.df[(self.df['Job Title'] == position) & (self.df['Org'] == org)]

    def display_WordCloudImage(self, word_cloud_data):
        if 'word_cloud_image' not in st.session_state:
            st.session_state.word_cloud_image = word_cloud_data.to_image()
        else:
            st.session_state.word_cloud_image = word_cloud_data.to_image()
        
        # Display the image stored in the session variable
        st.image(st.session_state.word_cloud_image, use_column_width=True)

    def plot_wc(self,display_df):
        if len(display_df):
            with self.col2:
                # Check if there's an existing word cloud image displayed
                if 'word_cloud_display' in st.session_state:
                    # Clear the previous image if it exists
                    st.session_state.word_cloud_display.empty()

                # Generate and cache the word cloud data
                word_cloud_data = self.generate_WordCloudData(display_df)
                # Store the word cloud data in session state
                st.session_state.word_cloud_data = word_cloud_data

                # Display the word cloud image
                st.session_state.word_cloud_display = st.empty()
                st.session_state.word_cloud_display.image(word_cloud_data.to_image())

                
               
                
                
                 

    @staticmethod
    def clear_word_cloud_data():
        st.session_state.word_cloud_data = None
    
    @staticmethod
    @st.cache_data(persist=True)
    def init_data():
        df = pd.read_csv('data_6.csv')
        return df

    @staticmethod
    @st.cache_data(persist=True)
    def generate_WordCloudData(grams):
        full_text = ""
        for i in grams:
            full_text += f"{i} "

        stopword_set = set(STOPWORDS)
        cloud_no_stopword = WordCloud(
            background_color='white', stopwords=stopword_set,
            colormap='ocean', width=300, height=350, repeat=True
        ).generate(full_text)

        return cloud_no_stopword
    
    @staticmethod
    def get_download_link(file_name):
        with open(file_name, "rb") as file:
            contents = file.read()
        b64 = base64.b64encode(contents).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{file_name}">Click here to download</a>'
        return href

    def export_data(self, df, file_name="exported_data.csv"):
        if len(df)==0:
            st.warning("No data to export.")
            return

        df.to_csv(file_name, index=False)
        st.success(f"Data exported successfully. Click the link below to download:")
        st.markdown(self.get_download_link(file_name), unsafe_allow_html=True)
    
    
    
    def init_slider(self,fig_slider,type):
        

        if type == "years":
            for i in range(2010,2022):
                df_filter = self.df.loc[self.df['join_year'] < i]
                fig_slider.add_trace(go.Scatter(visible = False, x = df_filter['emb1'], y = df_filter['emb2'],  mode = 'markers', marker_color = df_filter['color_code'], opacity = 1, text = df_filter['Job Title'], customdata=df_filter['Name'],  hovertemplate = 'Self Intro: %{text} <br>'  + 'Name: %{customdata}<extra></extra>'))
            fig_slider.add_trace(go.Scatter(visible = False, x = df_filter['emb1'], y = df_filter['emb2'],  mode = 'markers', marker_color = df_filter['color_code'], opacity = 0.1, text = df_filter['Job Title'], customdata=df_filter['Name'],  hovertemplate = 'Self Intro: %{text} <br>' +  'Name: %{customdata}<extra></extra>'))

            # Set custom axis range to center the points
      
            fig_slider.data[-2].visible = True
            fig_slider.data[-1].visible = True
            return fig_slider

        elif type =="kmeans":
            df_filter_km = self.df[['emb1', 'emb2','Self Introduction','Name','Job Title']]
            x_km = df_filter_km[['emb1', 'emb2']]
            for i_km in range(2,8, 2):
                
                kmeans = KMeans(n_clusters=i_km, random_state=42).fit(x_km)
                fig_slider.add_trace(go.Scatter(visible = False, x = df_filter_km['emb1'], y = df_filter_km['emb2'],  mode = 'markers', marker=dict(color=kmeans.labels_), opacity = 1, text = df_filter_km['Job Title'], customdata=df_filter_km['Name'],  hovertemplate = 'Self Intro: %{text} <br>' +'Name: %{customdata}<extra></extra>'))
            fig_slider.add_trace(go.Scatter(visible = False, x = df_filter_km['emb1'], y = df_filter_km['emb2'],  mode = 'markers', marker=dict(color=kmeans.labels_), opacity = 0.1, text = df_filter_km['Job Title'], customdata=df_filter_km['Name'],  hovertemplate = 'Self Intro: %{text} <br>' +'Name: %{customdata}<extra></extra>'))
    
            fig_slider.data[-2].visible = True
            fig_slider.data[-1].visible = True
            
            return fig_slider
    
    
    def fig_trace_update(self,fig, paper_len = 0):
            fig.update_traces(marker_size=4 )
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            fig.update_xaxes(range=[0, 6])
            fig.update_yaxes(range=[-4, 4])

            fig.update_layout(
                    showlegend=False,                    
                    autosize=True,
                    )
            
            if paper_len == 0:
                fig.update_layout(template = "plotly_white")


            return fig

    def main_viz(self):
        
        fig_main = go.Figure()
        fig_main.add_trace(go.Scatter(x = self.df['emb1'], y = self.df['emb2'],  mode = 'markers', marker_color = self.df['color_code'], opacity = 1, text = self.df['Job Title'], customdata=self.df['Name'], hovertemplate = 'Self Intro: %{text} <br>'  + 'Name: %{customdata}<extra></extra>'))
        
        fig_main = self.fig_trace_update(fig_main)

   
        selected_data = plotly_events(
            fig_main,
            select_event= True
        )
        display_df, filter_data = self.get_ngrams(selected_data,self.df)
        st.session_state.display_df = display_df


    def search(self,key, mode):
        
        if mode == 'search':
            if len(key.split()) > 1:
                paper_idx = []
                for idx,(i,a) in enumerate(zip(self.position_filter(self.selected_position,self.selected_org)['Self Introduction'],self.position_filter(self.selected_position,self.selected_org)['Name']) ):
                    sing_split = i.split()
                    a = a[1:-1].replace("'","")
                    a = a[1:-1].replace(".","")
                    auth_token = a.split(",")
                    
                    pair_text = [f'{sing_split[i]} {sing_split[i+1]}' for i in range(len(sing_split) -1 )]

                    
                    if key in pair_text or key in auth_token:
                        paper_idx.append(idx)
                paper_idx = list(paper_idx)

                if len(paper_idx) == 0:
                    st.error("No match Found")
                    # return

            else:
                paper_idx = []
                for idx,(i,a) in enumerate(zip(self.position_filter(self.selected_position,self.selected_org)['Self Introduction'],self.position_filter(self.selected_position,self.selected_org)['Name']) ):
                
                    if key in i.split() or key in a:
                        paper_idx.append(idx)
                paper_idx = paper_idx
                if len(paper_idx) == 0:
                    st.error("No match Found")
        # self.df=self.position_filter(self.selected_position)
        self.df=self.position_filter(self.selected_position,self.selected_org)
        print(paper_idx)
        print(len(self.df))
        filter_data_search = self.position_filter(self.selected_position,self.selected_org).iloc[paper_idx]
        print(len(filter_data_search))
        st.session_state.bt_plot = True
           
        
        
        trace_1 = go.Scatter( x = filter_data_search['emb1'], y = filter_data_search['emb2'],  mode = 'markers', marker_color = filter_data_search['color_code'], opacity = 1, text = filter_data_search['Job Title'], customdata=filter_data_search['Name'], hovertemplate = 'Self Intro: %{text} <br>'  + 'Name: %{customdata}<extra></extra>')
        
        if len(paper_idx):
            trace_2 = go.Scatter(visible = True, x = self.position_filter(self.selected_position,self.selected_org)['emb1'], y = self.position_filter(self.selected_position,self.selected_org)['emb2'],  mode = 'markers', marker_color = self.df['color_code'], opacity = 0.1, text = self.df['Job Title'], customdata=self.df['Name'], hoverinfo = 'none')
        else:
            trace_2 = go.Scatter(visible = True, x = self.position_filter(self.selected_position,self.selected_org)['emb1'], y = self.position_filter(self.selected_position,self.selected_org)['emb2'],  mode = 'markers', marker_color = self.df['color_code'], opacity = 1, text = self.df['Job Title'], customdata=self.df['Name'], hoverinfo = 'none')
        
        
        fig_search = go.Figure(data = [trace_1, trace_2])

        fig_search = self.fig_trace_update(fig_search, paper_len = len(paper_idx))

        with self.col1:  
             selected_data_search = plotly_events(
                fig_search,
                select_event= True
            )
 
        display_df, filter_data = self.get_ngrams(selected_data_search,self.df)
        st.session_state.display_df = display_df
        
        return filter_data_search
       


    def get_ngrams(self,selected_data, filter_df):
        selected_paper = [el['pointIndex'] for el in selected_data]
        filter_data = filter_df.filter(items = selected_paper, axis = 0)

        filter_title = " ".join([x for x in filter_data['Self Introduction']])
        tokens_without_sw = [word.lower() for word in filter_title.split() if not word.lower() in self.all_stopwords]
        bigram_count = Counter(ngrams(tokens_without_sw, 2))
        unigram_count = Counter(ngrams(tokens_without_sw, 1))
        filter_data.reset_index(drop = True, inplace=True)
        display_data = []
        for i in unigram_count.most_common(25):
            display_data.append(i[0][0])
        for i in bigram_count.most_common(25):
            display_data.append(f'{i[0][0]} {i[0][1]}')
    
        self.plot_wc(display_data)
        PlotPaper.clear_word_cloud_data()
        return display_data,filter_data[['Self Introduction','Name']]
    
    # @st.cache(allow_output_mutation=True)
    # def get_WordCloud(self,grams):
    #     full_text = ""
    #     for i in grams:
    #         full_text += f"{i} "

    #     stopword_set = set(stopwords.words('english') + list(STOPWORDS))
    #     cloud_no_stopword = WordCloud(background_color='white', stopwords=stopword_set, colormap='ocean',
    #                                                 width=300, height=350, repeat=True).generate(full_text)
        

    #     image = cloud_no_stopword.to_image()
    #     # word_list = []
    #     # position_list = []
    #     # color_list = []

    #     # for (word, freq), fontsize, position, orientation, color in cloud_no_stopword.layout_:
    #     #     word_list.append(word)
    #     #     position_list.append(position)
    #     #     color_list.append(color)

    #     # x=[]
    #     # y=[]
    #     # for i in position_list:
    #     #     x.append(i[0])
    #     #     y.append(i[1])



    #     # trace = go.Scatter(x=x,
    #     #                     y=y,
    #     #                     hoverinfo='text',
    #     #                     mode = "text",
    #     #                     hovertext=['{0}'.format(w) for w in word_list],
    #     #                     text=word_list,
    #     #                     )
    #     # layout = go.Layout({'xaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False},
    #     #             'yaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False}},
    #     #             margin=go.layout.Margin(
    #     #                                 l=0, #left margin
    #     #                                 r=0, #right margin
    #     #                                 b=0, #bottom margin
    #     #                                 t=0  #top margin
    #     #                             ))

    #     # fig = go.Figure(data=[trace], layout = layout)

    #     # fig.add_layout_image(
    #     #     dict(
    #     #         source=image,
    #     #         xref="x",
    #     #         yref="y",
    #     #         x=-15,
    #     #         y=300,
    #     #         sizex=340,
    #     #         sizey=305,
    #     #         sizing="stretch",
    #     #         opacity=1,
    #     #         layer="above",
    #     #     )
    #     # )

    #     # fig.update_xaxes(visible=False)
    #     # fig.update_yaxes(visible=False)
    #     # w, h = image.size

    #     # fig.update_layout(
           

    #     #     xaxis_range=[0, w],
    #     #     yaxis_range=[0, h],
    #     # )

        

    #     return image
    
    def year_filter_graph(self):
        fig_slider = go.Figure()
        fig_slider = self.init_slider( fig_slider,"years")
        if self.selected_position and self.selected_position != "All":
            filter_data_position = self.df.filter(items=fig_slider.data[-1].customdata, axis=0)
            fig_slider.data[-1].x = filter_data_position['emb1']
            fig_slider.data[-1].y = filter_data_position['emb2']
        steps = []
        for i in range(len(fig_slider.data) - 1):
            step = dict(
                method="update",
                args=[{"visible": [False] * len(fig_slider.data)},
                    ], 
                label=str(2010 + i)# layout attribute
            )
            step["args"][0]["visible"][i] = True
            step["args"][0]["visible"][-1] = True# Toggle i'th trace to "visible"
            steps.append(step)

        sliders = [dict(
            active=len(fig_slider.data) - 2,
            currentvalue={"prefix": "Until Year: "},
            pad={"t": 50},
            steps=steps,
            visible = True
        )]

        fig_slider.update_layout(
            sliders=sliders,
            autosize = True,
        )

        fig_slider = self.fig_trace_update(fig_slider)

        # st.plotly_chart(fig_slider)
        with self.col1:
            with self.tab1:
                st.caption("🌟 Uncover the fascinating passions and hobbies of our incredible team members! 🎯 Slide through the years and explore the diverse and delightful interests that make our Pinployees shine! 🌈 ")
               

               
                selected_data = plotly_events(
                fig_slider,
                select_event= True)



        filter_data, display_df = self.get_ngrams(selected_data,self.df)
        st.session_state.display_df = display_df


    def KMeans_slider(self):

        fig_slider_km = go.Figure()
        fig_slider_km = self.init_slider(fig_slider_km,"kmeans")

        steps = []
        # print(fig_slider_km.data)
        for i in range(1, len(fig_slider_km.data)):
            step = dict(
                method="update",
                args=[{"visible": [False] * len(fig_slider_km.data)},
                    ], 
                label=str(2*i)# layout attribute
            )
            step["args"][0]["visible"][i - 1] = True
            step["args"][0]["visible"][-1] = True# Toggle i'th trace to "visible"
            steps.append(step)
        sliders = [dict(
            active=i -1,
            currentvalue={"prefix": "With n ="},
            pad={"t": 50},
            steps=steps,
            visible = True
        )]

        fig_slider_km.update_layout(
            sliders=sliders
        )

        fig_slider_km = self.fig_trace_update(fig_slider_km)

        # st.plotly_chart(fig_slider_km)
        with self.col1:
            with self.tab2:
                st.caption("🚀 Slide and discover cosmic clusters of shared interests! 🔮 Explore the stellar connections between our amazing team members! 🌌")
                
                selected_data = plotly_events(
                fig_slider_km,
                select_event= True
                )


     
        filter_data, display_df = self.get_ngrams(selected_data,self.df)
        st.session_state.display_df = display_df


  