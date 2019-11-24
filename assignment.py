from model import Model
from preprocessing import load_data
import sys

def train(model):
    pass

def test(model):
    pass

def main():
    small_dataset = False
    if len(sys.argv) != 2 or sys.argv[1] not in {"BIG", "SMALL"}:
        print("USAGE: python assignment.py <Dataset>")
        print("<Dataset>: [BIG/SMALL]")
        exit()

    print('=== Bach LSTM Generator (nwee, jlhota) ===')
    m = Model()

    if sys.argv[1] == "BIG":
        data = load_data('./data/jsbach.net/midi/', all_data=True)
    elif sys.argv[1] == "SMALL":
        data = load_data('./data/jsbach.net/midi/', all_data=False)

    print('=== Training ===')
    train(m)

    print('=== Testing ===')
    test(m)

if __name__ == '__main__':
    main()
