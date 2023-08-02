import streamlit as st
from PIL import Image
from visualize import PlotPaper
from PIL import Image
import argparse


class homepage():
    def __init__(self) -> None:
                
            
        if "select_option" not in st.session_state:
                st.session_state.select_option = "Homepage"


 
            
            
        if st.session_state.select_option == "Homepage":
            homepage.Init_Home()
        


    @staticmethod
    def Init_Home():
        st.markdown(
            """
            <style>
            .title {
                font-size: 36px;
                color: #336699;
                padding-bottom: 20px;
            }
            .description {
                font-size: 18px;
                color: #666666;
                padding-bottom: 20px;
            }
            .intro-text {
                font-size: 20px;
                color: #444444;
                padding-bottom: 20px;
            }
            .highlight {
                font-weight: bold;
                color: #336699;
            }
            .output-description {
                font-size: 24px;
                color: #336699;
                padding-top: 40px;
                padding-bottom: 20px;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        col1, col2 = st.columns([3,1])
        
        
        # with col2:
        #     if st.button("Ignite Your Research Journey!"):
        #         # Store session state to indicate that active learning page should be shown
        #         st.experimental_set_query_params(page="active_learning")
        # Add introduction

        PlotPaper()

        #
