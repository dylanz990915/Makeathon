import streamlit as st

from home import homepage
# st.set_page_config(layout="wide", page_title="ExploreX")
if "enter_home" not in st.session_state:
    st.session_state.enter_home = False
        
def main():

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
    

    if st.session_state.enter_home:
        homepage()

    
    else:
        page_bg_img = '''
<style>
body {
    background-image: url("https://example.com/background_image.jpg");
    background-size: cover;
}
h1 {
    color: #ff6200;
    font-size: 36px;
}
p {
    font-size: 18px;
    color: #333333;
    line-height: 1.6;
}
</style>
'''
        col1, col2 , col3 = st.columns([1,3,1])
       
            

        st.markdown(page_bg_img, unsafe_allow_html=True)
    
        # Page title and header with appealing colors
        st.title("Introducing Funday - Your Workday Adventure Starts Here! 📌")
        st.header("Hey there, awesome Pinployees of Pinterest, right here on Workday! 🌟")
        
        # Add spacing for a more visually appealing layout
        # st.write("\n")
        
        # Add some introductory text with a larger font size and colored text
        st.write("<p style='font-size: 22px; color: #ffffff;'>Are you ready to level up your Workday experience and dive into a world of excitement? Get ready for Funday, the fantastic dashboard built right within Workday to connect, celebrate, and have a blast!</p>", unsafe_allow_html=True)
        
        # Continue with the rest of the introduction using formatted text
        st.markdown("<p>At Funday, we believe that work and fun can go hand in hand, like the perfect Pinterest board! 🌈 So let's turn your everyday tasks into delightful adventures, right here in your Workday universe!</p>", unsafe_allow_html=True)
        st.markdown("<p>Here's the inside scoop: We've gathered your fabulous introductions, each filled with your Pinterest passion, and transformed them into something magical! 🎨 Imagine your unique Pinterest interests transformed into colorful points on an interactive map, right here on Workday!</p>", unsafe_allow_html=True)
        
        # Add an appealing image (if available)
        # st.image("funday_image.jpg", use_column_width=True)
    
    # Continue with the rest of the introduction
        st.markdown("<p>Picture this: You're a shining Pinployee point, and all around you are your incredible co-workers, each with their own Pinterest flair! 🌟 Together, we're creating a web of connections, discovering new friends, and bonding over shared hobbies, all within the Workday ecosystem!</p>", unsafe_allow_html=True)
        st.markdown("<p>But wait, there's more fun to explore! Funday brings you the ultimate feature - 'Pin-tastic Clusters'! 🌟 We've created groups based on your Pinterest interests, making it easier than ever to find your tribe! Whether you're in the 'Travel Dreamers,' the 'Foodie Fanatics,' or the 'Art & Craft Crew,' you'll find like-minded pals who brighten your Workday journey!</p>", unsafe_allow_html=True)
        st.markdown("<p>Funday is as easy to navigate as pinning your favorite ideas! 🎯 Zoom in to explore the tiniest details of your pals' interests, or zoom out to appreciate the grand canvas of our vibrant Pinterest community! Hover over points to uncover delightful surprises about your co-workers and ignite meaningful conversations!</p>", unsafe_allow_html=True)
        st.markdown("<p>Don't worry, Pinployees, your privacy is our top priority! 🤫 Funday ensures your data is safe and secure, so you can focus on connecting and having a blast!</p>", unsafe_allow_html=True)
        st.markdown("<p>So, are you ready to infuse your Workday with the spirit of Funday? 🎉 Join the Funday extravaganza right here on Workday and let the excitement begin!</p>", unsafe_allow_html=True)
        st.markdown("<p>Let's make every workday a fabulous Funday together! 🎡🌟🎉 Welcome to the Funday family, where work and play collide in the most awesome way!</p>", unsafe_allow_html=True)
        

        if st.button('Try Funday now!', on_click= homepage):
            st.session_state.enter_home = True
            st.experimental_rerun()
            # insert your logic to start exploring

if __name__ == "__main__":
    main()
