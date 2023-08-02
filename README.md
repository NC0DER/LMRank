[![Python-Versions](https://img.shields.io/badge/python-3.7_|_3.8_|_3.9_|_3.10-blue.svg)]()
[![Software-License](https://img.shields.io/badge/License-Apache--2.0-green)](https://github.com/NC0DER/LMRank/blob/main/LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rzjfnkKQ7EDdEFaLDhJspw7WdcKh9ipV?usp=sharing)
[![License: CC BY 4.0](https://licensebuttons.net/l/by/4.0/80x15.png)](https://creativecommons.org/licenses/by/4.0/)

# LMRank

This repository hosts code for the paper:

* [LMRank: Utilizing Pre-Trained Language Models and Dependency Parsing for Keyphrase Extraction](https://ieeexplore.ieee.org/document/10179894)

## About
LMRank is a keyphrase extraction approach, that builds on recent advancements in the fields of Keyphrase Extraction and Deep learning. Specifically, it utilizes dependency parsing, a technique which forms more coherent candidate keyphrases, as well as highly accurate `sentence-transformers` models to semantically compare the candidate keyphrases with the text and extract the most relevant ones. 

If you have any practical or research questions take a quick look at the [FAQ](https://github.com/NC0DER/LMRank/wiki/Frequently-Asked-Questions-(FAQ)). As shown in the FAQ, LMRank currently supports 14 languages including English, Greek and others.

## Installation
* Run `pip install git+https://github.com/NC0DER/LMRank/`

## Example
```python
from LMRank.model import LMRank

text = """
      Machine learning (ML) is a field of inquiry devoted to understanding and building 
      methods that 'learn', that is, methods that leverage data to improve performance 
      on some set of tasks.[1]  It is seen as a part of artificial intelligence. Machine 
      learning algorithms build a model based on sample data, known as training data, 
      in order to make predictions or decisions without being explicitly programmed 
      to do so.[2] Machine learning algorithms are used in a wide variety of 
      applications, such as in medicine, email filtering, speech recognition, agriculture, 
      and computer vision, where it is difficult or unfeasible to develop conventional 
      algorithms to perform the needed tasks.[3][4] A subset of machine learning is closely 
      related to computational statistics, which focuses on making predictions using computers.
 """
model = LMRank()
results = model.extract_keyphrases(text, language_code = 'en', top_n = 10)

print(results)
```

## Results

```python
[('machine learning', 0.012630817497539287),
('training data', 0.012126753048877012),
('data mining', 0.01074470174604561),
('inferences', 0.010638922139139086),
('neural networks', 0.010636195886216913),
('statistical learning', 0.010432026486056019),
('artificial intelligence', 0.010388909274319064),
('algorithms', 0.009517552425412449),
('unsupervised learning', 0.009076004950492262),
('predictive analytics', 0.008616772762298034)]
```

To see a list of supported languages and their codes, see the [FAQ](https://github.com/NC0DER/LMRank/wiki/Frequently-Asked-Questions-(FAQ)).

## Contributors
* Nikolaos Giarelis (giarelis@ceid.upatras.gr)
* Nikos Karacapilidis (karacap@upatras.gr)
