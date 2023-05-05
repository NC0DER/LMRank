import sys
import spacy


def retrieve_spacy_model():
    """
    Function which loads or downloads, 
    the required English nlp model.
    """
    try:
        nlp = spacy.load('en_core_web_sm')
    except OSError:
        message = 'First time setup: Downloading the English NLP model....'
        print(message, file = sys.stderr)
        spacy.cli.download('en_core_web_sm')
        nlp = spacy.load('en_core_web_sm')
    return nlp


def remove_last_seps(string, seps = '!?.'):
    """
    Function which removes the last occurence
    of sentence separators from a string.
    """
    sep_set = set(seps)
    for i in range(len(string) - 1, -1, -1):
        if string[i - 1] in sep_set:
            return string[:i - 1]
    return string


def find_nth_occurence(string, substring, start, end, n):
    """
    Function which finds the nth occurence of a 
    substring given a range of the original string.
    """

    i = string.find(substring, start, end)
    while i >= 0 and n > 1:
        i = string.find(substring, i + len(substring))
        n -= 1
    return i


def create_chunks(string, max_token_length, token_sep = ' '):
    """
    Function which creates chunks of max_token_length from a string,
    by using a token separator.
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
