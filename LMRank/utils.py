import sys
import spacy

from typing import TypeVar, Dict, List

# Generic type class for spacy model objects.
Model = TypeVar('Model')

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
    
    Input
    ----------
    model_name: path to spaCy model (str).

    Output
    -------
    nlp: the spacy nlp object (Model).

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
    
    Input
    ------
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

    Output
    -------
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
    
    Input
    ------
    string: (str)
        the string which the characters are removed from.
    
    max_token_length: (int)
        the maximum number of tokens of the model.

    Output
    -------
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
