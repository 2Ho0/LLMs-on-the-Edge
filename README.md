# LLMs on the Edge

This repository contains all the [data](dataset) used, all the [results](results) obtained and the corresponding source code 
for the experiments conducted as part of the “LLMs on the Edge” paper. 

## Code
The sourcecode to log the data during our experiments in located in the [code](code) folder.
The main file is [ask.py](code/ask.py). It is responsible for logging the answers and the time values.
The authors used ChatGPT (version GPT-4o) to assist with writing the code for the figures in the paper.

## Dataset

The [dataset](dataset) used is a subset of randomly selected 
questions and answers from the Question Answering Dataset [SQuAD2.0](https://rajpurkar.github.io/SQuAD-explorer/) which was created by [Stanford University](https://www.stanford.edu/).
This folder also contains the evaluation criteria for quality evaluation performed by the authors and by GPT-4o.

## Results

All results of the quality, energy and latency evaluations are stored in [results](results) for every model we tested on the edge device.


