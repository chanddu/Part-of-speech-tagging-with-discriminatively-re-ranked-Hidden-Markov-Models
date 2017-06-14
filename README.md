# Part-of-speech-tagging-with-discriminatively-re-ranked-Hidden-Markov-Models
Implemented POS tagging by combining a standard HMM tagger separately with a Maximum Entropy classifier designed to re-rank the k-best tag sequences produced by HMM – achieved better results than VITERBI (decoding algorithm)

Disclaimer: <br>
This programs takes approximately 4 to 5 hours to run: <br>
The soft copy of the report is attached incase you want to see the results <br>

Two programs have to be executed: <br>
The original algorithm: Viterbi <br>
The algorithm implemented in Project: Reranked HMM <br>

To execute: <br>
python hmm_beam_search.py <br>

Dependencies: <br>
python3 <br>
nltk <br>
megam <br>
numpy <br>
