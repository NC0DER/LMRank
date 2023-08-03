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
```
pip install git+https://github.com/NC0DER/LMRank/
```

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
[('conventional algorithms', 0.03220074744562463),
 ('machine learning', 0.0320379078219184),
 ('training data', 0.02651275416153127),
 ('artificial intelligence', 0.023564133570545886),
 ('computational statistics', 0.018363250279455255),
 ('speech recognition', 0.017827318362436336),
 ('computer vision', 0.017721180700768415),
 ('data', 0.01647833767159313),
 ('sample data', 0.014187748325602852),
 ('predictions', 0.014133139194664955)]
```

To see a list of supported languages and their codes, see the [FAQ](https://github.com/NC0DER/LMRank/wiki/Frequently-Asked-Questions-(FAQ)).

## Citation
Please use the following BibTeX entry to cite `LMRank` if you use it in your research work:

```bibtex
@article{giarelis2023lmrank,
  title={LMRank: Utilizing pre-trained language models and dependency parsing for keyphrase extraction},
  author={Giarelis, Nikolaos and Karacapilidis, Nikos},
  journal={IEEE Access},
  year={2023},
  publisher={IEEE}
}
```

## Contributors
* Nikolaos Giarelis (giarelis@ceid.upatras.gr)
* Nikos Karacapilidis (karacap@upatras.gr)
