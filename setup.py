from setuptools import setup

with open('README.md', 'r', encoding = 'utf-8') as f:
    long_description = f.read()

setup(
    name = 'lmrank',
    packages = ['LMRank'],
    version = '2.0',
    author = 'Nikolaos Giarelis',
    author_email = 'giarelis@ceid.upatras.gr',
    description = 'LMRank is a keyphrase extraction approach that utilizes spaCy and state-of-the-art sentence transformersperforms keyword extraction with state-of-the-art sentence transformer models.',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    url = 'https://github.com/NC0DER/LMRank',
    keywords = 'nlp keyphrase extraction keyword extraction sentence transformer embeddings',
    classifiers = [
        'License :: OSI Approved :: Apache Software License',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires = '>=3.7',
    install_requires = [
        'spacy<=3.6',
        'psutil<=5.9.5',
        'faiss-cpu-py36<=1.7.3',
        'sentence-transformers<=2.2.2'
    ]
)
