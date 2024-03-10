import math
import os
import mlx.core as mx
import mlx.nn as nn
import sqlite3
import subprocess
from mlx_lm import load, generate
from utils import linear_to_lora_layers

def send_imessage(number, text):
    """
    Sends an iMessage to the specified number with the given text.

    Parameters:
    - number: The phone number or email address to send the message to. Must be a string.
    - text: The message text to send. Must be a string.

    Returns:
    - None
    """
    # AppleScript command to send an iMessage
    applescript_command = f'''tell application "Messages"
        send "{text}" to buddy "{number}" of (service 1 whose service type is iMessage)
    end tell'''

    # Execute the AppleScript command
    try:
        subprocess.run(["osascript", "-e", applescript_command], check=True)
        print(f"Message sent to {number}: {text}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to send message to {number}: {e}")

    import sqlite3
import os
import time

# Path to the chat.db database
DATABASE_PATH = "~/Library/Messages/chat.db"
DATABASE_PATH = os.path.expanduser(DATABASE_PATH)  # Expands the ~ to the full path

def get_last_message_id(cursor):
    """Get the ID of the last message in the database."""
    cursor.execute("SELECT MAX(ROWID) FROM message")
    return cursor.fetchone()[0]

def fetch_new_messages(cursor, last_id):
    """Fetch new messages since the last known message ID."""
    cursor.execute("SELECT ROWID, handle_id, text FROM message WHERE ROWID > ? ORDER BY ROWID ASC", (last_id,))
    return cursor.fetchall()

def get_sender(handle_id, cursor):
    """Get the sender's phone number or email from the handle_id."""
    cursor.execute("SELECT id FROM handle WHERE ROWID = ?", (handle_id,))
    result = cursor.fetchone()
    return result[0] if result else "Unknown"

def fetch_past_messages(cursor, handle_id, limit=10):
    """Fetch the past 'limit' messages from a specific sender."""
    cursor.execute("""
        SELECT text FROM message
        WHERE handle_id = ? AND text IS NOT NULL AND cache_roomnames IS NULL
        ORDER BY ROWID DESC
        LIMIT ?
    """, (handle_id, limit))
    return [item[0] for item in cursor.fetchall()][::-1]


def main():
    model, tokenizer = load("mlx_model")
    def gen_text(inp):
        return generate(
            model=model,
            tokenizer=tokenizer,
            temp=0.8,
            max_tokens=500,
            prompt=f"<s>[INST]{inp}[/INST]",
        )
    # Convert linear layers to lora layers and unfreeze in the process
    linear_to_lora_layers(model, 4)
    model.load_weights("adapters.npz", strict=False)
    model.eval()

    # Connect to the database
    conn = sqlite3.connect(os.path.expanduser("~/Library/Messages/chat.db"))
    # Connect to the database
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    # Get the last message ID at the start
    last_id = get_last_message_id(cursor)

    try:
        print("Listening for new messages...")
        while True:
            # Fetch any new messages
            new_messages = fetch_new_messages(cursor, last_id)
            if new_messages:
                print(f"New messages: {new_messages}")
                for msg in new_messages:
                    rowid, handle_id, text = msg
                    if text is not None:
                        sender = get_sender(handle_id, cursor)
                        past_messages = fetch_past_messages(cursor, handle_id)
                        response_text = gen_text("\n".join(past_messages))  # Assuming gen_text can handle multiple messages
                        print(past_messages)
                        send_imessage(sender, response_text)
                        print(f"Responded to {sender} based on the last 10 messages.")
                    last_id = rowid  # Update the last seen message ID

            # Wait for 5 seconds before checking again
            time.sleep(5)
    except KeyboardInterrupt:
        print("Stopping the script.")
    finally:
        conn.close()

if __name__ == "__main__":
    main()