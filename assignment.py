from model import Model
from preprocessing import load_data

def train(model):
    pass

def test(model):
    pass

def main():
    print('=== Bach LSTM Generator (nwee, jlhota) ===')
    m = Model()

    data = load_data('./data/jsbach.net/midi/')

    print('=== Training ===')
    train(m)

    print('=== Testing ===')
    test(m)

if __name__ == '__main__':
	main()
