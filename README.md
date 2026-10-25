# Sentiment classification with SimpleRNN, LSTM and GRU (RLDL project)
Word embeddings + sequential models, implemented from scratch in NumPy (forward pass, back-propagation through time,
Adam, gradient clipping), with a numerical gradient check.

Files: data.py (synthetic review generator, tokenizer, vocabulary, padding, IMDb CSV loader), nn.py (models + gradient check),
run_experiment.py (all experiments, figures), predict.py (demo), results.json (measured results), figures/ (plots).

Run:
  pip install -r requirements.txt
  python3 nn.py                        # gradient check
  python3 run_experiment.py            # full experiment, about 10 minutes on one CPU core
  python3 predict.py "the movie was not very boring"     # demo

Real IMDb data (NOT tested in the build environment, which had no internet):
  python3 run_experiment.py --csv "IMDB Dataset.csv" --limit 5000 --maxlen 200 --out results_imdb.json
(expects columns `review` and `sentiment`; NumPy training is slow, so use a subset)
