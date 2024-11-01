# Documentation

This repository contains all the [data](dataset) used, all the [results](results) obtained and the corresponding source code 
for the experiments conducted as part of the scientific paper [LLMs on the Edge: Quality, Latency, and Energy Efficiency](https://www.doi.org/10.18420/inf2024_104) published at KIU-2024 (5th Workshop on "KI in der Umweltinformatik"). 

## Code
The sourcecode to log the data during our experiments is located in the [code](code) folder.
The main file is [ask.py](code/ask.py). It is responsible for logging the answers and the time values.


## Dataset

The [dataset](dataset) used is a subset of randomly selected 
questions and answers from the Question Answering Dataset [SQuAD2.0](https://rajpurkar.github.io/SQuAD-explorer/) which was created by [Stanford University](https://www.stanford.edu/).
This folder also contains the evaluation criteria for quality evaluation performed by the authors and by GPT-4o.

## Results

All results of the quality, energy and latency evaluations are stored in [results](results) for every model we tested on the edge device.

## Acknowledgements
ChatGPT (version GPT-4o) assisted us in creating the figures for this paper.

## Attribution
If you use parts of this repository in your work, please cite as follows:

>                 Bast, Sebastian; Begic Fazlic, Lejla; Naumann, Stefan; Dartmann, Guido (2024): LLMs on the Edge: Quality, Latency, and Energy Efficiency. INFORMATIK 2024. DOI: [10.18420/inf2024_104](https://www.doi.org/10.18420/inf2024_104). Bonn: Gesellschaft für Informatik e.V.. PISSN: 1617-5468. ISBN: 978-3-88579-746-3. pp. 1183-1192. 5. Workshop "KI in der Umweltinformatik" (KIU-2024). Wiesbaden. 24.-26. September 2024
