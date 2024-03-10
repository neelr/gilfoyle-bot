import random
import sqlite3
import json
import pandas as pd
import re  # For regex matching
import os  # For file path manipulation
from tqdm import tqdm  # For progress bar


def extract_messages(db_path):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)

    # Create a SQL query to extract relevant message data
    query = """
    SELECT
        message.text as sent_text,
        message.date as sent_date,
        message.is_from_me,
        chat.chat_identifier
    FROM
        message
        JOIN chat_message_join ON message.ROWID = chat_message_join.message_id
        JOIN chat ON chat_message_join.chat_id = chat.ROWID
    WHERE
        message.text IS NOT NULL
    ORDER BY
        chat.chat_identifier, message.date
    """

    # Execute the query and fetch the results
    df = pd.read_sql_query(query, conn)

    # Close the database connection
    conn.close()

    return df


def is_ignored_message(text):
    # Define a regex pattern for messages to ignore
    ignore_patterns = [
        r'^Loved “.*”$',
        r'^Liked “.*”$'
    ]
    return any(re.match(pattern, text) for pattern in ignore_patterns)


def format_for_training_old(df):
    pairs = []
    prompt_buffer = []
    completion_buffer = []
    current_chat_id = None
    is_previous_from_me = None
    prompt = ''

    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        # Skip ignored messages
        if is_ignored_message(row['sent_text']):
            continue

        chat_id = row['chat_identifier']

        # Initialize or check if we're still in the same chat
        if current_chat_id is None or current_chat_id == chat_id:
            # If sender changes within the same chat, or if it's the first message of the chat
            if is_previous_from_me is not None and row['is_from_me'] != is_previous_from_me:
                # If changing from other to me, finalize prompt
                if row['is_from_me']:
                    if prompt_buffer:  # Ensure there's a prompt
                        prompt = ' '.join(prompt_buffer)
                        prompt_buffer = []  # Reset prompt buffer
                    else:
                        prompt = ''
                    # Start new completion
                    completion_buffer.append(row['sent_text'])
                else:
                    # If changing from me to other, finalize completion and start new prompt
                    if completion_buffer:  # Ensure there's a completion
                        completion = ' '.join(completion_buffer)
                        if prompt:  # Only add pair if there's a prompt
                            pairs.append(
                                {'prompt': prompt, 'completion': completion})
                        completion_buffer = []  # Reset completion buffer
                    prompt_buffer.append(row['sent_text'])  # Start new prompt
            else:
                # Continue buffering messages from the same sender
                if row['is_from_me']:
                    completion_buffer.append(row['sent_text'])
                else:
                    prompt_buffer.append(row['sent_text'])
        else:
            # Finalize the last message(s) when chat changes
            if prompt_buffer or completion_buffer:
                completion = ' '.join(completion_buffer)
                prompt = ' '.join(prompt_buffer)
                if prompt or completion:  # Add pair if either is non-empty
                    pairs.append({'prompt': prompt, 'completion': completion})
                prompt_buffer = []
                completion_buffer = []

            # Reset for a new chat
            current_chat_id = chat_id
            if row['is_from_me']:
                completion_buffer.append(row['sent_text'])
            else:
                prompt_buffer.append(row['sent_text'])

        is_previous_from_me = row['is_from_me']

    # Handle any remaining messages in buffer after the loop
    if prompt_buffer or completion_buffer:
        completion = ' '.join(completion_buffer)
        prompt = ' '.join(prompt_buffer)
        if prompt or completion:  # Ensure we don't add empty pairs
            pairs.append({'prompt': prompt, 'completion': completion})

    return pairs


def format_for_training(df, window_size=10):
    data = []
    chat_ids = df['chat_identifier'].unique()
    for chat_id in tqdm(chat_ids, desc="Processing chats"):
        chat_messages = df[df['chat_identifier'] == chat_id]
        for i in range(window_size, len(chat_messages)):
            # Check if the current message is from me
            if chat_messages.iloc[i]['is_from_me'] == 1 and not is_ignored_message(chat_messages.iloc[i]['sent_text']):
                # Construct the prompt from the preceding window_size messages
                prompt_messages = chat_messages.iloc[i-window_size:i]
                prompt = '\n'.join([msg for msg in prompt_messages['sent_text']
                                    if not is_ignored_message(msg)])
                completion = chat_messages.iloc[i]['sent_text']

                # Add the formatted data to the list
                data.append({'prompt': prompt, 'completion': completion})
    return data


def save_to_jsonl(pairs, output_file):
    with open(output_file, 'w') as f:
        for pair in pairs:
            json_record = json.dumps(pair, ensure_ascii=False)
            f.write(json_record + '\n')


def split_save_train_test_jsonl(input_file, train_ratio=0.8, seed=42):
    """
    Splits a .jsonl file into training and testing datasets and saves them.

    Parameters:
    - input_file: Path to the input .jsonl file.
    - train_ratio: Ratio of the dataset to be used as training data.
    - seed: Random seed for reproducibility.

    Outputs:
    - Saves two .jsonl files, one for training and one for testing.
    """
    # Ensure the train ratio is between 0 and 1
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")

    # Set the seed for reproducibility
    random.seed(seed)

    # Read the input .jsonl file
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Shuffle the lines to ensure random distribution
    random.shuffle(lines)

    # Calculate the split index
    split_idx = int(len(lines) * train_ratio)

    # Split the data
    train_data = lines[:split_idx]
    test_data = lines[split_idx:]

    # Save the training data
    with open('train.jsonl', 'w', encoding='utf-8') as train_file:
        for item in train_data:
            train_file.write(item)

    # Save the testing data
    with open('test.jsonl', 'w', encoding='utf-8') as test_file:
        for item in test_data:
            test_file.write(item)

    print(
        f"Data split into {len(train_data)} training and {len(test_data)} testing entries.")


def convert_jsonl_format(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            # Parse the existing JSON line
            data = json.loads(line)
            prompt = data.get('prompt', '')
            completion = data.get('completion', '')

            # Format according to the new specification
            formatted_text = f'<s>[INST]{prompt}[/INST]{completion}</s>'

            # Write the new format to the output file
            json_record = json.dumps(
                {"text": formatted_text}, ensure_ascii=False)
            outfile.write(json_record + '\n')


def export():
    db_path = '~/Library/Messages/chat.db'
    output_file = 'train.jsonl'

    # Adjust the path for the current operating system's file path conventions
    db_path = os.path.expanduser(db_path)

    print("Extracting messages...")
    df = extract_messages(db_path)

    print("Formatting data for training...")
    pairs = format_for_training(df)

    print(f"Saving to {output_file}...")
    save_to_jsonl(pairs, output_file)

    print("Converting to new format...")
    convert_jsonl_format('train.jsonl', 'train_new_format.jsonl')

    print("Splitting into training and testing datasets...")
    split_save_train_test_jsonl(
        'train_new_format.jsonl', train_ratio=0.8, seed=42)

    print("Done.")


if __name__ == "__main__":
    export()
