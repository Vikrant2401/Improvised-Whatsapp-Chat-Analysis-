import re
import pandas as pd

def preprocess(data):
    # Regex pattern to match WhatsApp chat timestamps
    pattern = r'\d{1,2}/\d{1,2}/\d{2},\s\d{1,2}:\d{2}[\s\u202F]?[APap][Mm]'

    # Split messages and dates
    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)

    # Handle mismatched lengths
    min_length = min(len(messages), len(dates))
    messages = messages[:min_length]
    dates = dates[:min_length]

    # Create DataFrame
    df = pd.DataFrame({'user_message': messages, 'message_date': dates})

    # Clean non-breaking spaces and convert to datetime
    df['message_date'] = df['message_date'].astype(str).str.replace('\u202F', ' ', regex=False)
    df['message_date'] = pd.to_datetime(df['message_date'], format='%m/%d/%y, %I:%M %p', errors='coerce')

    # Drop rows with invalid dates
    df = df.dropna(subset=['message_date'])

    # Rename column
    df.rename(columns={'message_date': 'date'}, inplace=True)

    # Extract users and messages
    users, messages = [], []
    for message in df['user_message']:
        entry = re.split(r'([\w\W]+?):\s', message, maxsplit=1)
        if len(entry) > 1:
            users.append(entry[1])
            messages.append(entry[2])
        else:
            users.append('group_notification')
            messages.append(entry[0])

    df['user'] = users
    df['message'] = messages
    df.drop(columns=['user_message'], inplace=True)

    # Extract datetime features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    return df
