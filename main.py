import argparse
from utils import processor_main


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help="Your model static param dict file path.")
    parser.add_argument('--mp3_data', type=str, required=True, help="Your mp3 file path or the song id")
    args = parser.parse_args()

    genre_name, max_idx, _ = processor_main(params_pth=args.model_path,
                                            mp3_data=args.mp3_data,
                                            csv_path='./data/v2/track_lists.csv')
    print(f'{genre_name=}\n{max_idx=}')


if __name__ == '__main__':
    main()
