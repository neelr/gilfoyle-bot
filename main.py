import export
import train
import server
import sys

def main():
    print('Starting up CLI...')
    args = sys.argv[1:]

    if len(args) < 1:
        print('Usage: main.py <command>')
        print("""
        Available commands:
        - export: Export your chat.db database to test, train, and train_new_format jsonl files to finetune on
        - train: Train a new model on the exported jsonl files (all locally using LoRA)
        - server: Start the server to send and receive messages

        - all: Run all commands in order
        """)
        sys.exit(1)
    
    command = args[0]

    if command == 'export':
        print('Exporting...')
        export.main()
    elif command == 'train':
        print('Training...')
        train.main()
    elif command == 'server':
        print('Starting server...')
        server.main()
    elif command == 'all':
        export.main()
        train.main()
        server.main()

    print('Exiting...')

if __name__ == '__main__':
    main()
