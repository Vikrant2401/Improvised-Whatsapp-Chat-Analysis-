import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from nltk.corpus import stopwords
import nltk
import emoji
from textblob import TextBlob
import networkx as nx
from pyvis.network import Network

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')

# Custom list of Marathi and Hindi stopwords
marathi_stopwords = {
    "ka", "ho", "zala", "pn", "ky", "la", "tr", "nhi", "na", "ok", "ata", "ch",
    "ahe", "ha", "ani", "mg", "hota", "tu", "nay", "kay", "mala", "te", "pan",
    "cha", "ki", "tya", "nahi", "to", "he", "mi", "majha", "tuzha", "amcha",
    "tumcha", "mag", "kahi", "karan", "tar", "ti", "tula", "br", "madhe", "mdhe", "null", "😂", "🥳", "💐", "🌞", "🙏", "🎊"
}

hindi_stopwords = {
    "hai", "haan", "nahi", "kyu", "kyun", "ho", "gaya", "kya", "kaise", "kar", "raha",
    "rhe", "rhi", "kr", "tha", "thi", "hun", "bhi", "par", "pr", "ab", "aur", "se",
    "sab", "wo", "woh", "jo", "uska", "unka", "mera", "tera", "hamara", "inka",
    "unka", "kuch", "toh", "lekin", "magar", "sirf", "bus", "kyunki", "jab", "tab",
    "fir", "phir", "ki", "ke", "yeh", "vo", "koi", "isse", "usse", "jise", "jaise", "🎂", "😂", "🥳", "💐", "🌞", "🙏", "🎊"
}

custom_stopwords = marathi_stopwords.union(hindi_stopwords)


# Function to fetch statistics
def fetch_stats(selected_user, df):
    if selected_user != "Overall":
        df = df[df['user'] == selected_user]

    num_messages = df.shape[0]
    num_words = df['message'].apply(lambda x: len(str(x).split())).sum()
    num_media = df[df['message'] == "<Media omitted>"].shape[0]
    num_links = df['message'].str.contains(r'http[s]?://', regex=True).sum()

    return num_messages, num_words, num_media, num_links


# Function to generate the monthly timeline
def monthly_timeline(selected_user, df):
    if selected_user != "Overall":
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month']).count()['message'].reset_index()
    timeline['month'] = timeline['month'].astype(str) + "-" + timeline['year'].astype(str)

    return timeline[['month', 'message']].rename(columns={'message': 'message_count'})


# Function to create the activity heatmap
def activity_heatmap(selected_user, df):
    if selected_user != "Overall":
        df = df[df['user'] == selected_user]

    activity = df.pivot_table(index="day_name", columns="hour", values="message", aggfunc="count").fillna(0)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(activity, cmap="YlOrRd", ax=ax)  # Revert to red-orange-yellow color scheme
    plt.title("Activity Heatmap", fontsize=16)
    plt.xlabel("Hour", fontsize=12)
    plt.ylabel("Day", fontsize=12)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

    return fig


# Function to create the word cloud
def create_wordcloud(selected_user, df):
    stop_words = set(stopwords.words('english')).union(custom_stopwords)

    if selected_user != "Overall":
        df = df[df['user'] == selected_user]

    # Filter out media messages
    filtered_messages = \
        df[~df['message'].str.contains("Media omitted|omitted media", case=False, na=False, regex=True)]['message']

    # Generate WordCloud while removing stopwords
    text = " ".join(filtered_messages)
    wc = WordCloud(
        width=800, height=500,  # Increased dimensions for larger word cloud
        min_font_size=10,
        max_font_size=100,  # Added max font size for better visibility
        stopwords=stop_words,
        background_color='white',
        colormap='viridis',
        max_words=150,  # Increased max words
        random_state=42,  # For reproducibility
        contour_width=1,  # Added contour for better definition
        contour_color='steelblue'
    ).generate(text)

    return wc


# Function to get most common words
def most_common_words(selected_user, df):
    stop_words = set(stopwords.words('english')).union(custom_stopwords)

    if selected_user != "Overall":
        df = df[df['user'] == selected_user]

    words = []
    for msg in df['message']:
        for word in msg.lower().split():
            if word not in stop_words and word.isalpha():
                words.append(word)

    common_words_df = pd.DataFrame(Counter(words).most_common(20), columns=["Word", "Frequency"])
    return common_words_df


# Function to find most active users
def most_active_users(df):
    user_counts = df['user'].value_counts().head()
    user_df = user_counts.reset_index().rename(columns={'index': 'User', 'user': 'Message Count'})

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(user_counts.index, user_counts.values, color='purple')
    plt.title("Most Active Users", fontsize=16)
    plt.xlabel("User", fontsize=12)
    plt.ylabel("Message Count", fontsize=12)
    plt.xticks(rotation=45)

    return user_df, fig


# Function to analyze emoji usage
def emoji_analysis(selected_user, df):
    if selected_user != "Overall":
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if c in emoji.EMOJI_DATA])

    emoji_counter = Counter(emojis)
    emoji_df = pd.DataFrame(emoji_counter.most_common(len(emoji_counter)), columns=["Emoji", "Frequency"])

    return emoji_df


# Function to find most busy day
def most_busy_day(selected_user, df):
    if selected_user != "Overall":
        df = df[df['user'] == selected_user]

    busy_day = df['day_name'].value_counts()
    return busy_day


# Function to find most busy month
def most_busy_month(selected_user, df):
    if selected_user != "Overall":
        df = df[df['user'] == selected_user]

    busy_month = df['month'].value_counts()
    return busy_month


# Function to analyze sentiment (only for "Overall")
def analyze_sentiment(selected_user, df):
    if selected_user != "Overall":
        # Return an empty DataFrame for individual users
        df['sentiment'] = []
        df['sentiment_category'] = []
        return df

    # Calculate sentiment polarity for each message
    df['sentiment'] = df['message'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

    # Categorize sentiment
    df['sentiment_category'] = df['sentiment'].apply(
        lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral')
    )

    return df


# Function to get sentiment summary (only for "Overall")
def sentiment_summary(selected_user, df):
    if selected_user != "Overall":
        # Return an empty Series for individual users
        return pd.Series(dtype=float)

    # Calculate sentiment distribution
    sentiment_distribution = df['sentiment_category'].value_counts()
    return sentiment_distribution


# Improved function to create an interaction graph
def create_interaction_graph(df, min_interactions=2):
    """
    Creates an interactive network graph showing chat interactions between users.

    Parameters:
    - df: DataFrame containing the chat data
    - min_interactions: Minimum number of interactions required to show a connection

    Returns:
    - G: NetworkX graph object
    - html_file: Path to the saved network visualization HTML file
    """
    # Create a directed graph
    G = nx.DiGraph()

    # Count interactions between users
    interactions = {}

    # Iterate through messages to find replies
    for i in range(1, len(df)):
        if df.loc[i, 'user'] != df.loc[i - 1, 'user']:
            sender = df.loc[i - 1, 'user']
            receiver = df.loc[i, 'user']

            if (sender, receiver) in interactions:
                interactions[(sender, receiver)] += 1
            else:
                interactions[(sender, receiver)] = 1

    # Filter interactions to remove noise
    filtered_interactions = {k: v for k, v in interactions.items() if v >= min_interactions}

    # Add edges with weights
    for (sender, receiver), weight in filtered_interactions.items():
        G.add_edge(sender, receiver, weight=weight, title=f"{weight} interactions")

    # Calculate node importance metrics
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)

    # Set up color palette based on message frequency
    user_message_counts = df['user'].value_counts().to_dict()
    max_count = max(user_message_counts.values())

    # Create the network visualization with improved aesthetics
    net = Network(notebook=False, height="800px", width="100%", directed=True,
                  bgcolor="#f8f9fa", font_color="#333333")

    # Add nodes with more meaningful visual attributes
    for node in G.nodes():
        # Calculate node size based on message frequency
        msg_count = user_message_counts.get(node, 0)
        size = 20 + (msg_count / max_count * 60)

        # Calculate node color based on centrality (more central = darker color)
        centrality_score = degree_centrality[node]
        # Convert centrality to hex color (blue spectrum, darker = more central)
        color_value = max(0, min(255, int(255 * (1 - centrality_score))))
        node_color = f"#{color_value:02x}{color_value:02x}ff"

        # Add node with helpful tooltip
        net.add_node(
            node,
            label=node,
            size=size,
            color=node_color,
            title=f"User: {node}\nMessages: {msg_count}\nCentrality: {centrality_score:.2f}"
        )

    # Add edges with improved styling
    for sender, receiver, data in G.edges(data=True):
        weight = data['weight']
        # Scale width more visibly
        width = 1 + (weight * 0.8)
        # Darker edges indicate stronger connections
        edge_color = f"rgba(70, 130, 180, {min(0.9, 0.3 + weight / 20)})"

        net.add_edge(
            sender,
            receiver,
            width=width,
            color=edge_color,
            title=f"{weight} interactions",
            arrows={'to': {'enabled': True, 'scaleFactor': 0.5}}
        )

    # Improve layout algorithm parameters for better readability
    net.barnes_hut(
        gravity=-3000,  # Reduced gravity for less crowding
        central_gravity=0.2,  # Adjusted for better centering
        spring_length=200,  # Shorter springs for tighter layout
        spring_strength=0.08,  # Stronger springs for more structure
        damping=0.85  # Reduced damping for smoother layout
    )

    # Add a legend as custom HTML
    legend_html = """
    <div style="position: absolute; top: 10px; right: 10px; padding: 10px; background-color: rgba(255,255,255,0.8); 
                border-radius: 5px; border: 1px solid #ddd; font-family: Arial; z-index: 999;">
        <h3 style="margin-top: 0;">Network Legend</h3>
        <p><b>Node Size:</b> Number of messages sent</p>
        <p><b>Node Color:</b> Darker blue = More central in conversations</p>
        <p><b>Edge Width:</b> Number of interactions</p>
        <p><b>Edge Direction:</b> Direction of conversation flow</p>
        <p><small>Hover over nodes/edges for details</small></p>
    </div>
    """

    # Add controls and interactive features
    net.show_buttons(filter_=['physics', 'nodes', 'edges'])

    # Save the network with the legend included
    html_file = "chat_network.html"
    net.save_graph(html_file)

    # Add the legend to the saved HTML file
    with open(html_file, 'r', encoding='utf-8') as file:
        html_content = file.read()

    # Insert legend before the closing body tag
    modified_html = html_content.replace('</body>', f'{legend_html}</body>')

    with open(html_file, 'w', encoding='utf-8') as file:
        file.write(modified_html)

    print(f"Network visualization saved as {html_file}")
    return G, html_file


# Function to create a simplified conversation flow diagram
def create_conversation_flow(df, top_n=5):
    """
    Creates a simplified conversation flow diagram showing interaction patterns
    between the most active users.

    Parameters:
    - df: DataFrame containing the chat data
    - top_n: Number of top users to include in the diagram

    Returns:
    - html_file: Path to the saved diagram HTML file
    """
    # Get the most active users
    top_users = df['user'].value_counts().head(top_n).index.tolist()

    # Filter data to only include top users
    filtered_df = df[df['user'].isin(top_users)].copy()

    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes for each top user
    for user in top_users:
        G.add_node(user)

    # Count message exchanges between top users
    exchanges = {}

    # Track conversations
    for i in range(1, len(filtered_df)):
        current_user = filtered_df.iloc[i]['user']
        previous_user = filtered_df.iloc[i - 1]['user']

        if current_user != previous_user and current_user in top_users and previous_user in top_users:
            pair = (previous_user, current_user)
            if pair in exchanges:
                exchanges[pair] += 1
            else:
                exchanges[pair] = 1

    # Calculate message counts per user
    msg_counts = filtered_df['user'].value_counts().to_dict()

    # Add weighted edges
    for (source, target), weight in exchanges.items():
        G.add_edge(source, target, weight=weight)

    # Create network visualization
    net = Network(height="600px", width="100%", directed=True, bgcolor="#f0f0f0")

    # Color palette for nodes
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#d35400"]

    # Add nodes with custom styling
    for i, user in enumerate(top_users):
        color = colors[i % len(colors)]
        size = 15 + (msg_counts[user] / max(msg_counts.values()) * 30)
        net.add_node(user, label=user, size=size, color=color,
                     title=f"User: {user}\nTotal Messages: {msg_counts[user]}")

    # Add edges with custom styling
    for (source, target, weight) in G.edges(data='weight'):
        thickness = 1 + weight / 5
        net.add_edge(source, target, value=weight, width=thickness,
                     title=f"{source} → {target}: {weight} exchanges",
                     arrows={"to": {"enabled": True}})

    # Apply physics settings for better layout
    net.barnes_hut(spring_length=200, spring_strength=0.05, damping=0.9, central_gravity=0.1)

    # Add title and legend
    html_title = """
    <div style="text-align:center; margin-bottom:20px;">
        <h2>Conversation Flow Between Top Users</h2>
        <p>Shows who responds to whom in the chat</p>
    </div>
    """

    # Save to HTML
    html_file = "conversation_flow.html"
    net.save_graph(html_file)

    # Add title to HTML file
    with open(html_file, 'r', encoding='utf-8') as file:
        content = file.read()

    # Insert title
    body_start = content.find('<body')
    body_content_start = content.find('>', body_start) + 1
    modified_content = content[:body_content_start] + html_title + content[body_content_start:]

    with open(html_file, 'w', encoding='utf-8') as file:
        file.write(modified_content)

    print(f"Conversation flow diagram saved as {html_file}")
    return html_file


# Function to visualize user relationships based on conversation patterns
def visualize_user_relationships(df, min_exchanges=5):
    """
    Creates a network visualization showing relationships between users
    based on their conversation patterns, with community detection.

    Parameters:
    - df: DataFrame containing the chat data
    - min_exchanges: Minimum number of exchanges required to consider a relationship

    Returns:
    - html_file: Path to the saved visualization HTML file
    """
    # Create a directed graph
    G = nx.Graph()  # Undirected for community detection

    # Track all users
    all_users = df['user'].unique()
    for user in all_users:
        G.add_node(user)

    # Count interactions between users
    interactions = {}
    for i in range(1, len(df)):
        if df.iloc[i]['user'] != df.iloc[i - 1]['user']:
            user1 = df.iloc[i - 1]['user']
            user2 = df.iloc[i]['user']

            # Sort users to create undirected edge
            if user1 > user2:
                user1, user2 = user2, user1

            pair = (user1, user2)
            if pair in interactions:
                interactions[pair] += 1
            else:
                interactions[pair] = 1

    # Add edges for significant interactions
    for (user1, user2), count in interactions.items():
        if count >= min_exchanges:
            G.add_edge(user1, user2, weight=count)

    # Detect communities
    try:
        communities = nx.community.greedy_modularity_communities(G)

        # Map users to their community
        community_map = {}
        for i, community in enumerate(communities):
            for user in community:
                community_map[user] = i
    except:
        # If community detection fails, assign everyone to community 0
        community_map = {user: 0 for user in all_users}

    # Color palette for communities
    community_colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#d35400", "#c0392b"]

    # Calculate message counts
    msg_counts = df['user'].value_counts().to_dict()
    max_msgs = max(msg_counts.values())

    # Create the visualization
    net = Network(height="800px", width="100%", bgcolor="#f5f5f5", font_color="#333333")

    # Add nodes with community colors
    for user in G.nodes():
        comm_id = community_map.get(user, 0)
        color = community_colors[comm_id % len(community_colors)]
        size = 10 + (msg_counts.get(user, 0) / max_msgs * 40)

        net.add_node(user, label=user, size=size, color=color,
                     title=f"User: {user}\nMessages: {msg_counts.get(user, 0)}\nGroup: {comm_id + 1}")

    # Add edges
    for user1, user2, data in G.edges(data=True):
        weight = data['weight']
        width = 0.5 + (weight / 10)
        opacity = min(0.9, 0.2 + (weight / 50))

        net.add_edge(user1, user2, width=width, title=f"{weight} exchanges",
                     color=f"rgba(120, 120, 120, {opacity})")

    # Apply force-directed layout
    net.barnes_hut(spring_length=200, spring_strength=0.05, central_gravity=0.1)

    # Add interactive controls
    net.show_buttons(filter_=['physics'])

    # Add legend for communities
    legend_items = []
    for i, _ in enumerate(set(community_map.values())):
        color = community_colors[i % len(community_colors)]
        legend_items.append(
            f'<div><span style="display:inline-block; width:15px; height:15px; background-color:{color}; margin-right:5px;"></span>Group {i + 1}</div>')

    legend_html = f"""
    <div style="position:absolute; top:10px; left:10px; background-color:rgba(255,255,255,0.8); 
                padding:10px; border-radius:5px; border:1px solid #ddd; z-index:999;">
        <h3 style="margin-top:0;">User Groups</h3>
        {"".join(legend_items)}
        <hr style="margin:10px 0;">
        <div><small>Node size = number of messages</small></div>
        <div><small>Edge thickness = frequency of interaction</small></div>
        <div><small>Colors indicate conversation groups</small></div>
    </div>
    """

    # Save the network
    html_file = "user_relationships.html"
    net.save_graph(html_file)

    # Add legend to the HTML file
    with open(html_file, 'r', encoding='utf-8') as file:
        content = file.read()

    modified_content = content.replace('</body>', f'{legend_html}</body>')

    with open(html_file, 'w', encoding='utf-8') as file:
        file.write(modified_content)

    print(f"User relationship visualization saved as {html_file}")
    return html_file


# Function to create a time-based interaction analysis
def create_temporal_network(df, time_window='M'):
    """
    Creates an animation showing how the conversation network evolves over time.

    Parameters:
    - df: DataFrame containing the chat data
    - time_window: Time window for aggregation ('D' for daily, 'W' for weekly, 'M' for monthly)

    Returns:
    - html_file: Path to the saved visualization HTML file
    """
    # Ensure datetime column exists and is properly formatted
    if 'date' not in df.columns:
        print("Error: DataFrame must contain a 'date' column with datetime information")
        return None

    # Make a copy to avoid modifying the original
    df_copy = df.copy()

    # Convert date to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df_copy['date']):
        df_copy['date'] = pd.to_datetime(df_copy['date'])

    # Create time periods based on the specified window
    df_copy['period'] = df_copy['date'].dt.to_period(time_window)

    # Get all unique periods
    periods = df_copy['period'].unique()
    periods.sort()

    # Create a network for each time period
    networks = []

    for period in periods:
        period_df = df_copy[df_copy['period'] == period]

        # Skip periods with too few messages
        if len(period_df) < 10:
            continue

        # Create a graph for this period
        G = nx.Graph()

        # Count interactions
        interactions = {}
        for i in range(1, len(period_df)):
            if period_df.iloc[i]['user'] != period_df.iloc[i - 1]['user']:
                user1 = period_df.iloc[i - 1]['user']
                user2 = period_df.iloc[i]['user']

                # Create undirected edge
                if user1 > user2:
                    user1, user2 = user2, user1

                pair = (user1, user2)
                if pair in interactions:
                    interactions[pair] += 1
                else:
                    interactions[pair] = 1

        # Add significant edges (more than 1 interaction)
        for (user1, user2), count in interactions.items():
            if count > 1:
                G.add_edge(user1, user2, weight=count)

        # Calculate message counts for this period
        msg_counts = period_df['user'].value_counts().to_dict()

        # Store the graph and associated data
        networks.append({
            'period': period.strftime('%Y-%m'),
            'graph': G,
            'msg_counts': msg_counts
        })

    # Create HTML for timeline slider
    slider_html = """
    <div style="text-align:center; margin:20px 0;">
        <input type="range" min="0" max="{max}" value="0" id="timeSlider" style="width:80%;">
        <div id="periodLabel" style="margin-top:10px; font-weight:bold;">{start_period}</div>
    </div>

    <script>
    var networks = {network_data};
    var currentNetwork = 0;
    var slider = document.getElementById('timeSlider');
    var label = document.getElementById('periodLabel');

    slider.max = networks.length - 1;

    slider.oninput = function() {{
        currentNetwork = parseInt(this.value);
        label.innerHTML = networks[currentNetwork].period;

        // Update network visualization
        // This part will be implemented with pyvis
    }};
    </script>
    """

    # Since we can't create a true animation with static HTML,
    # let's create a comprehensive view of the entire network with time indicators

    # Create a combined network
    combined_net = Network(height="800px", width="100%", bgcolor="#f5f5f5", directed=False)

    # Prepare data for combined visualization
    all_users = set()
    all_interactions = {}
    all_msg_counts = {}

    # Collect all users and their interactions across all periods
    for network_data in networks:
        period = network_data['period']
        G = network_data['graph']
        msg_counts = network_data['msg_counts']

        # Update all users
        all_users.update(G.nodes())

        # Update message counts per user
        for user, count in msg_counts.items():
            if user in all_msg_counts:
                all_msg_counts[user] += count
            else:
                all_msg_counts[user] = count

        # Update interactions with time information
        for user1, user2, data in G.edges(data=True):
            edge = (user1, user2) if user1 < user2 else (user2, user1)
            weight = data['weight']

            if edge in all_interactions:
                all_interactions[edge]['weight'] += weight
                all_interactions[edge]['periods'].append(period)
            else:
                all_interactions[edge] = {
                    'weight': weight,
                    'periods': [period]
                }

    # Calculate max message count for sizing
    max_msg_count = max(all_msg_counts.values()) if all_msg_counts else 1

    # Add all nodes to combined network
    for user in all_users:
        size = 15 + (all_msg_counts.get(user, 0) / max_msg_count * 40)
        combined_net.add_node(user, label=user, size=size,
                              title=f"User: {user}\nTotal Messages: {all_msg_counts.get(user, 0)}")

    # Add all edges with time information
    for (user1, user2), data in all_interactions.items():
        weight = data['weight']
        periods = ", ".join(data['periods'])

        width = 1 + (weight / 10)
        combined_net.add_edge(user1, user2, width=width,
                              title=f"Interactions: {weight}\nActive during: {periods}")

    # Apply layout
    combined_net.barnes_hut(spring_length=200, spring_strength=0.05)

    # Add controls
    combined_net.show_buttons(filter_=['physics'])

    # Create description for the visualization
    description_html = """
    <div style="text-align:center; margin:20px 0;">
        <h2>Conversation Network Over Time</h2>
        <p>This visualization shows the entire conversation network across all time periods.</p>
        <p>Hover over connections to see when users interacted with each other.</p>
    </div>
    """

    # Save the network
    html_file = "temporal_network.html"
    combined_net.save_graph(html_file)

    # Add description to the HTML file
    with open(html_file, 'r', encoding='utf-8') as file:
        content = file.read()

    # Insert description after body tag
    body_start = content.find('<body')
    body_content_start = content.find('>', body_start) + 1
    modified_content = content[:body_content_start] + description_html + content[body_content_start:]

    with open(html_file, 'w', encoding='utf-8') as file:
        file.write(modified_content)

    print(f"Temporal network visualization saved as {html_file}")
    return html_file
