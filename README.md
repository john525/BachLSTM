# BachLSTM (CS147 Final Project, jlhota & nwee)

A deep learning model to generate music sequences based on a corpus of Bach compositions represented as MIDI files.

Most music generation models have used an approach combining the LSTM with the Restricted Boltzmann Machine, but this project attempts to replicate a paper that uses a purely-deep-learning approach.

Our project is based on Alleng Huang and Raymond Wu's paper: https://cs224d.stanford.edu/reports/allenh.pdf

However, since their dataset from MuseData has been taken offline, we are using midi files from jsbach.net to train our network.
To download these files, cd to the 'data' directory and run
```
chmod +x get_data.sh
./get_data.sh
```

Then to run the project, do
```
python -m venv env
source env/bin/activate
pip install -r requirements.txt
python assignment.py
```
