import export
import train
import server
import sys
import argparse

parser = argparse.ArgumentParser(
    prog='Gilfoyle CLI',
    description='automatically respond to imeessages with a trained model!')

parser.add_argument(
    '--export', help='Export your chat.db database to test, train, and train_new_format jsonl files to finetune on', action='store_true')
parser.add_argument(
    '--train', help='Train a new model on the exported jsonl files (all locally using LoRA)', action='store_true')
parser.add_argument(
    '--server', help='Start the server to send and receive messages', action='store_true')
parser.add_argument('--all', help='Run all commands in order', action='store_true')
parser.add_argument('-i', '--iterations', type=int, default=1000,
                    help='Number of iterations to train the model for')


def main():
    print('Starting up CLI...')
    args = parser.parse_args()

    if args.export:
        print('Exporting...')
        export.main()
    elif args.train:
        print('Training...')
        train.main(args.iterations)
    elif args.server:
        print('Starting server...')
        server.main()
    elif args.all:
        export.main()
        train.main(args.iterations)
        server.main()
    else:
        parser.print_help()

    print('Exiting...')


if __name__ == '__main__':
    main()
