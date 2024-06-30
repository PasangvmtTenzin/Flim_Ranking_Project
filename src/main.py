import argparse
from src.analysis import analyze_data
from src.model import train_model, evaluate_model
from src.utils import load_data, setup_logging, handle_error

def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="Analyze film ranking data.")
    parser.add_argument('action', choices=['analyze', 'train', 'evaluate'], help="Action to perform")
    parser.add_argument('--data', type=str, required=True, help="Path to the data file")
    parser.add_argument('--model', type=str, default='model.pkl', help="Path to the model file")
    parser.add_argument('--start-year', type=int, help="Start year for analysis")
    parser.add_argument('--end-year', type=int, help="End year for analysis")
    args = parser.parse_args()

    try:
        data = load_data(args.data)

        if args.action == 'analyze':
            analyze_data(data, args.start_year, args.end_year)
        elif args.action == 'train':
            train_model(data)
        elif args.action == 'evaluate':
            evaluate_model(args.model, data)
        else:
            print("Invalid action. Choose 'analyze', 'train', or 'evaluate'.")
    except Exception as e:
        handle_error(e)

if __name__ == "__main__":
    main()
