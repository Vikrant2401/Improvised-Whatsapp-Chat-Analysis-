import streamlit as st
import pandas as pd
import preprocessor
import helper
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="WhatsApp Chat Analyzer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #424242;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-text {
        font-size: 1.2rem;
    }
    .stat-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0px;
    }
</style>
""", unsafe_allow_html=True)

# App title with custom styling
st.sidebar.markdown('<h1 style="color:#1E88E5;">WhatsApp Chat Analyzer</h1>', unsafe_allow_html=True)

# File upload with instructions
st.sidebar.markdown("### Upload your WhatsApp chat export file")
st.sidebar.markdown("_Export your chat from WhatsApp (without media)_")

uploaded_file = st.sidebar.file_uploader("Choose a file", type=["txt"])

if uploaded_file is not None:
    try:
        # Display a spinner while processing
        with st.spinner('Processing your chat data...'):
            bytes_data = uploaded_file.getvalue()
            data = bytes_data.decode("utf-8")
            df = preprocessor.preprocess(data)

        if df.empty:
            st.error("No valid data found in the file!")
        else:
            # Fetch unique users
            user_list = df['user'].unique().tolist()
            user_list.sort()
            user_list.insert(0, "Overall")

            # User selection with additional information
            st.sidebar.markdown("### Select User")
            selected_user = st.sidebar.selectbox(
                "Choose a specific user or analyze all messages:",
                user_list
            )

            # Analysis button
            if st.sidebar.button("Show Analysis", key="analyze_button", use_container_width=True):
                st.markdown('<h1 class="main-header">WhatsApp Chat Analysis</h1>', unsafe_allow_html=True)

                # Basic chat info
                st.markdown(
                    f"**Chat Period:** {df['date'].min().strftime('%B %d, %Y')} to {df['date'].max().strftime('%B %d, %Y')}")
                st.markdown(f"**Total participants:** {len(df['user'].unique())}")

                # Fetch statistics
                num_messages, num_words, num_media, num_links = helper.fetch_stats(selected_user, df)

                # Top statistics in a nicer layout with columns
                st.markdown('<h2 class="sub-header">Top Statistics</h2>', unsafe_allow_html=True)

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.markdown('<div class="stat-box">', unsafe_allow_html=True)
                    st.markdown("### Messages")
                    st.markdown(f"<h2 style='text-align: center; color: #1E88E5;'>{num_messages:,}</h2>",
                                unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                with col2:
                    st.markdown('<div class="stat-box">', unsafe_allow_html=True)
                    st.markdown("### Words")
                    st.markdown(f"<h2 style='text-align: center; color: #43A047;'>{num_words:,}</h2>",
                                unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                with col3:
                    st.markdown('<div class="stat-box">', unsafe_allow_html=True)
                    st.markdown("### Media")
                    st.markdown(f"<h2 style='text-align: center; color: #FB8C00;'>{num_media:,}</h2>",
                                unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                with col4:
                    st.markdown('<div class="stat-box">', unsafe_allow_html=True)
                    st.markdown("### Links")
                    st.markdown(f"<h2 style='text-align: center; color: #E53935;'>{num_links:,}</h2>",
                                unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                # Sentiment Analysis (only for "Overall")
                if selected_user == "Overall":
                    st.markdown('<h2 class="sub-header">Sentiment Analysis</h2>', unsafe_allow_html=True)
                    sentiment_df = helper.analyze_sentiment(selected_user, df)
                    sentiment_summary = helper.sentiment_summary(selected_user, df)

                    # Check if sentiment_summary is empty
                    if not sentiment_summary.empty:
                        # Display sentiment summary in a nicer way
                        col1, col2 = st.columns([1, 2])

                        with col1:
                            st.dataframe(
                                sentiment_summary.reset_index().rename(
                                    columns={'index': 'Sentiment', 'sentiment_category': 'Count'}
                                )
                            )

                        with col2:
                            # Plot sentiment distribution
                            fig, ax = plt.subplots(figsize=(10, 6))
                            colors = {'Positive': '#43A047', 'Negative': '#E53935', 'Neutral': '#1E88E5'}
                            ax.bar(
                                sentiment_summary.index,
                                sentiment_summary.values,
                                color=[colors.get(x, '#1E88E5') for x in sentiment_summary.index]
                            )
                            plt.title("Sentiment Distribution", fontsize=18, fontweight='bold')
                            plt.xlabel("Sentiment", fontsize=14)
                            plt.ylabel("Count", fontsize=14)
                            plt.xticks(rotation=0, fontsize=12)
                            plt.grid(axis='y', linestyle='--', alpha=0.7)
                            st.pyplot(fig)
                    else:
                        st.info("No sentiment data available.")

                # Monthly timeline with improved styling
                st.markdown('<h2 class="sub-header">Monthly Timeline</h2>', unsafe_allow_html=True)
                monthly_df = helper.monthly_timeline(selected_user, df)
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(monthly_df['month'], monthly_df['message_count'], marker='o', linewidth=3, markersize=8,
                        color='#1E88E5')
                ax.fill_between(monthly_df['month'], monthly_df['message_count'], alpha=0.2, color='#1E88E5')
                plt.title("Message Frequency Over Time", fontsize=18, fontweight='bold')
                plt.xlabel("Month", fontsize=14)
                plt.ylabel("Message Count", fontsize=14)
                plt.xticks(rotation=45, fontsize=10)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                fig.tight_layout()
                st.pyplot(fig)

                # Activity analysis in tabs
                st.markdown('<h2 class="sub-header">Activity Analysis</h2>', unsafe_allow_html=True)

                tab1, tab2, tab3 = st.tabs(["Day Analysis", "Month Analysis", "Time Heatmap"])

                with tab1:
                    # Most Busy Day
                    busy_day = helper.most_busy_day(selected_user, df)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.bar(busy_day.index, busy_day.values, color='#43A047')
                    plt.title("Messages by Day of Week", fontsize=18, fontweight='bold')
                    plt.xlabel("Day", fontsize=14)
                    plt.ylabel("Message Count", fontsize=14)
                    plt.xticks(rotation=0, fontsize=12)
                    plt.grid(axis='y', linestyle='--', alpha=0.7)
                    st.pyplot(fig)

                with tab2:
                    # Most Busy Month
                    busy_month = helper.most_busy_month(selected_user, df)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.bar(busy_month.index, busy_month.values, color='#FB8C00')
                    plt.title("Messages by Month", fontsize=18, fontweight='bold')
                    plt.xlabel("Month", fontsize=14)
                    plt.ylabel("Message Count", fontsize=14)
                    plt.xticks(rotation=0, fontsize=12)
                    plt.grid(axis='y', linestyle='--', alpha=0.7)
                    st.pyplot(fig)

                with tab3:
                    # Activity heatmap
                    st.markdown("### Hourly Activity Patterns")
                    st.markdown("This heatmap shows when users are most active throughout the week")
                    heatmap_data = helper.activity_heatmap(selected_user, df)
                    st.pyplot(heatmap_data)

                # Content analysis section
                st.markdown('<h2 class="sub-header">Content Analysis</h2>', unsafe_allow_html=True)

                # WordCloud
                st.markdown("### Word Cloud")
                st.markdown("Most frequently used words in the conversation")
                wordcloud = helper.create_wordcloud(selected_user, df)
                fig, ax = plt.subplots(figsize=(14, 10))  # Increased figure size
                ax.imshow(wordcloud, interpolation='bilinear')
                plt.axis("off")
                st.pyplot(fig)

                # Most common words
                st.markdown("### Most Common Words")
                common_words_df = helper.most_common_words(selected_user, df)

                col1, col2 = st.columns([1, 2])

                with col1:
                    st.dataframe(common_words_df)

                with col2:
                    # Visualize the common words
                    fig, ax = plt.subplots(figsize=(12, 8))
                    ax.barh(common_words_df['Word'][:15], common_words_df['Frequency'][:15], color='#5E35B1')
                    plt.title("Most Frequently Used Words", fontsize=18, fontweight='bold')
                    plt.xlabel("Frequency", fontsize=14)
                    plt.ylabel("Word", fontsize=14)
                    ax.invert_yaxis()  # Invert y-axis to show most common word at the top
                    plt.grid(axis='x', linestyle='--', alpha=0.7)
                    st.pyplot(fig)

                # Emoji Analysis
                st.markdown("### Emoji Analysis")
                emoji_df = helper.emoji_analysis(selected_user, df)

                if not emoji_df.empty:
                    col1, col2 = st.columns([1, 2])

                    with col1:
                        st.dataframe(emoji_df.head(10))

                    with col2:
                        # Plot emoji frequency
                        fig, ax = plt.subplots(figsize=(12, 8))
                        ax.bar(
                            emoji_df['Emoji'][:10],
                            emoji_df['Frequency'][:10],
                            color=plt.cm.Set3.colors[:10]
                        )
                        plt.title("Top 10 Emojis Used", fontsize=18, fontweight='bold')
                        plt.xlabel("Emoji", fontsize=14)
                        plt.ylabel("Frequency", fontsize=14)
                        plt.xticks(fontsize=20)  # Larger emoji font
                        plt.grid(axis='y', linestyle='--', alpha=0.7)
                        st.pyplot(fig)
                else:
                    st.info("No emojis found in the chat.")

                # Most active users (only for Overall analysis)
                if selected_user == "Overall":
                    st.markdown('<h2 class="sub-header">User Analysis</h2>', unsafe_allow_html=True)
                    st.markdown("### Most Active Users")
                    active_users_df, active_users_chart = helper.most_active_users(df)

                    col1, col2 = st.columns([1, 2])

                    with col1:
                        st.dataframe(active_users_df)

                    with col2:
                        st.pyplot(active_users_chart)

                # Network Analysis (only for "Overall" analysis)
                if selected_user == "Overall":
                    st.markdown('<h2 class="sub-header">Network Analysis</h2>', unsafe_allow_html=True)
                    st.markdown("### User Interaction Network")
                    st.markdown("""
                    This visualization shows who talks to whom in the chat. 
                    - Larger nodes represent more active users
                    - Thicker lines represent more frequent interactions
                    - Use the physics button to adjust the layout
                    """)

                    # Create and display the interaction graph with expanded height
                    G, network_file = helper.create_interaction_graph(df)
                    st.components.v1.html(open(network_file, 'r').read(), height=800)  # Increased height

    except Exception as e:
        st.error(f"Error parsing file: {str(e)}")
        st.info("Make sure you've uploaded a valid WhatsApp chat export file (.txt format)")
else:
    # Display welcome message and instructions when no file is uploaded
    st.markdown('<h1 class="main-header">WhatsApp Chat Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="info-text">Upload your WhatsApp chat export file to get started!</p>',
                unsafe_allow_html=True)

    st.markdown("""
    ### How to export your WhatsApp chat:
    1. Open the chat you want to analyze
    2. Tap on the three dots (menu) → More → Export chat
    3. Choose "Without Media"
    4. Share the exported .txt file
    5. Upload it using the file uploader in the sidebar

    ### This analyzer will help you discover:
    - Message statistics and trends
    - Activity patterns by day and time
    - Most commonly used words and emojis
    - User interaction patterns
    - Sentiment analysis
    """)
