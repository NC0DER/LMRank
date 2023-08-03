import sys
import spacy

from typing import TypeVar, Dict, List, Tuple
from itertools import groupby
from operator import itemgetter
from spacy.matcher import Matcher

# Generic type class for spacy model objects.
Model = TypeVar('Model')
SpacyDoc = TypeVar('SpacyDoc')

language_nlp_model_associations = {
    'en': 'en_core_web_sm',
    'el': 'el_core_news_sm',
    'da': 'da_core_news_sm',
    'ca': 'ca_core_news_sm',
    'nl': 'nl_core_news_sm',
    'fi': 'fi_core_news_sm',
    'fr': 'fr_core_news_sm',
    'de': 'de_core_news_sm',
    'it': 'it_core_news_sm',
    'ja': 'ja_core_news_sm',
    'nb': 'nb_core_news_sm',
    'pt': 'pt_core_news_sm',
    'es': 'es_core_news_sm',
    'sv': 'sv_core_news_sm'
}


def retrieve_spacy_model(
        language_code: str, supported_languages: Dict[str, str]
    ) -> Model:
    """
    Function which loads or downloads, 
    the required nlp model given a correct language code.
    It also disables a list of unnecessary components.
    
    Input:
        model_name: (str)
            path to spaCy model.

        supported_languages: (Dict[str, str])
            the list of dictionaries of supported languages and their codes.
    
    Output:
        nlp: (Model)
            the spacy nlp object.
    """
    # Find the correct nlp model name based on the provided language code.
    model_name = language_nlp_model_associations.get(language_code)

    # Initialize a list of unnecessary components to be disabled.
    disable_list = [
        'ner', 'lemmatizer', 'textcat', 'entity-ruler', 
        'entity-linker', 'transformer', 'custom'
    ]

    if model_name is None:
        raise ValueError(f'{language_code} is not supported. The supported language codes are: {supported_languages}')

    try:
        nlp = spacy.load(model_name, disable = disable_list)
    except OSError:
        message = f'First time setup: Downloading the {language_code}:({supported_languages[language_code]}) NLP model....'
        print(message, file = sys.stderr)
        spacy.cli.download(model_name)
        nlp = spacy.load(model_name)
    return nlp


def form_candidate_keyphrases_as_noun_phrases(
        nlp: Model, doc: SpacyDoc, sentence_seps: str = '!?.'
    ) -> List[Tuple[str, int]]:
    """
    Function which extracts noun phrases,
    consisting of adjectives, nouns and proper nouns.
    This is used in case the noun_chunks component 
    of spaCy does not properly work for a specific language.

    Input:
        nlp: (Model)
            the nlp spaCy model.

        doc: (SpacyDoc)
            the spaCy Doc Object.

        sentence_seps: (str)
            String containing delimiters that separate sentences.
            These are removed from the end of each candidate keyphrase.

    Output:
        noun phrases: (List[Tuple[str, int]])
        a list of noun phrases and their first positional occurences.
    """

    # Define the patterns for extraction.
    adj_noun_phrases_patterns = [
        [{'POS': 'NOUN', 'OP': '+'}],
        [{'POS': 'ADJ', 'OP': '+'}, {'POS': 'NOUN', 'OP': '+'}],
        [{'POS': 'PROPN', 'OP': '+'}]
    ]

    # Initialize the token matcher with the nlp object vocabulary and add the patterns.
    matcher = Matcher(nlp.vocab)
    matcher.add('AdjNounPhrases', adj_noun_phrases_patterns)

    # Keep all noun phrases that have more than two characters.
    noun_phrases_occurences = [
        (remove_last_seps(doc[start:end].text, sentence_seps).lower(), start)
        for _, start, end in matcher(doc)
        if len(doc[start:end].text) > 2
    ]

    # Construct the set of seen keywords 
    # that are already part of keyphrases.
    seen = {
        word 
        for noun_phrase, _ in noun_phrases_occurences 
        if noun_phrase.count(' ')
        for word in noun_phrase.split()
    }
    
    # Keep only the unseen keywords.
    refined_noun_phrases_occurences = [
        (noun_phrase, occurence) 
        for noun_phrase, occurence in noun_phrases_occurences
        if noun_phrase.count(' ') or not noun_phrase in seen
    ]

    # Sort noun phrases by keyphrase text and groupby duplicate entries.
    # Only the first occurence of each keyphrase is preserved.
    candidate_keyphrases = {
            key: next(group)[1]
            for key, group in groupby(
                sorted(refined_noun_phrases_occurences, key = itemgetter(0)), 
                itemgetter(0))
    }

    # Some keyphrases may have a keyword that has an erroneous POS tag.
    # These leads to keyphrases ending where another one is starting.
    # These partial keyphrases are not kept.
    candidate_keyphrases_copy = list(candidate_keyphrases)
    for noun_phrase in candidate_keyphrases_copy:
        for other_phrase in candidate_keyphrases_copy:
            if (noun_phrase != other_phrase
                 and noun_phrase.rsplit(maxsplit = 1)[-1]
                     == other_phrase.split(maxsplit = 1)[0]):
                candidate_keyphrases.pop(noun_phrase, None)
                break

    return candidate_keyphrases


def remove_last_seps(string: str, seps: str = '!?.') -> str:
    """
    Function which removes the last occurence
    of sentence separators from a string.

    Input
    ------
    string: (str)
        the string which the characters are removed from.
    
    seps: (str)
        the sentence separators which are removed from the string.

    Output
    -------
    <object>: (str)
        a string with no sentence separators.

    """
    sep_set = set(seps)
    for i in range(len(string) - 1, -1, -1):
        if string[i - 1] in sep_set:
            return string[:i - 1]
    return string


def find_nth_occurence(string: str, substring: str, start: int, end: int, n: int) -> int:
    """
    Function which finds the nth occurence of a 
    substring given a range of the original string.
    
    Input:
        string: (str)
            the string which the characters are removed from.

        substring: (str)
            the substring to be found in the string.

        start: (int)
            the positional occurence to start the search from.

        end: (int)
            the positional occurence to end the search at.

        n: (int)
            the number of occurence for the substring.

    Output:
        <object>: (int)
            the n-th occurence of the substring.
    """

    i = string.find(substring, start, end)
    while i >= 0 and n > 1:
        i = string.find(substring, i + len(substring))
        n -= 1
    return i


def create_chunks(string: str, max_token_length: int, token_sep: str = ' ') -> List[str]:
    """
    Function which creates chunks of max_token_length from a string,
    by using a token separator.
    
    Input:
        string: (str)
            the string which the characters are removed from.
        
        max_token_length: (int)
            the maximum number of tokens of the model.

    Output:
        <object>: (List[str])
            A list of chunks of at most max_token_length.
    """
    
    # Initialize the chunk range values.
    chunk_ranges = []
    chunk_start = 0
    chunk_end = 0

    # Find chunk ranges.
    while chunk_end < len(string):

        # Shift the chunk window to the next chunk.
        chunk_start = chunk_end

        # Find the next chunking position.
        next_sep_pos = find_nth_occurence(
            string, token_sep, chunk_start, len(string), 
            max_token_length
        )

        # If it is not found, the last chunk is smaller than the others.
        # Thus, we reached the end of the string.
        if next_sep_pos == -1:
            chunk_end = len(string)
        else:
            chunk_end = next_sep_pos

        chunk_ranges.append((chunk_start, chunk_end))
        
    # Construct the chunks of texts based on the previously calculated ranges.
    chunks = [string[i:j] for (i,j) in chunk_ranges]

    return chunks
