from __future__ import annotations
import os
import torch
import LMRank.config as cfg

# Set all the environment variables to the specified number of threads.
# This is done because the torch.set_num_threads call in some versions 
# of pytorch breaks and ignores this number.
os.environ['MKL_NUM_THREADS'] = cfg.num_threads
os.environ['NUMEXPR_NUM_THREADS'] = cfg.num_threads
os.environ['OMP_NUM_THREADS'] = cfg.num_threads
os.environ['OMP_NUM_THREADS'] = cfg.num_threads
os.environ['OPENBLAS_NUM_THREADS'] = cfg.num_threads
os.environ['MKL_NUM_THREADS'] = cfg.num_threads
os.environ['VECLIB_MAXIMUM_THREADS'] = cfg.num_threads
os.environ['NUMEXPR_NUM_THREADS'] = cfg.num_threads
torch.set_num_threads(int(cfg.num_threads))

import numpy
import numpy.typing
import faiss

from itertools import groupby
from operator import itemgetter
from difflib import get_close_matches
from sentence_transformers import SentenceTransformer
from LMRank.utils import *
from typing import TypeVar, List, Tuple, Any 

# Generic type class for sentence transformer model objects.
Model = TypeVar('Model')

class LMRank:
    """
    A keyphrase extraction approach, which utilizes
    pretrained Language Models (LM), e.g. the ones
    found in the sentence transformers library, 
    to extract the top_n most important keyphrases 
    of the text, by calculating the semantic similarity 
    between the embeddings of the candidate keyphrases  
    and the text itself.
    """

    def __init__(
            self: LMRank, model: Model = None, 
            language_setting: str = 'english') -> None:
        """
        Initialization method.

        Arguments: 
            model: (A sentence transformers model).
                (default is None, since the best model, at the time of writing,
                will be loaded automatically given the language code)
                These models can be found in the following link:
                https://www.sbert.net/docs/pretrained_models.html,

            language_setting: (str)
                This can be either set to 'english' or 'multilingual',
                to enable the loading of the appropriate sentence
                transformers model, before extract_keyphrases() is called.
                
        """
        self.model = None
        self.multilingual_model = None


        # This follows spacy language codes.
        # Spacy nlp models are used by this approach 
        # for POS tagging, dependency parsing 
        # and sentence splitting.
        
        self.supported_languages = {
            'en': 'English', 
            'el': 'Greek',
            'da': 'Danish',
            'ca': 'Catalan',
            'nl': 'Dutch', 
            'fi': 'Finnish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'ja': 'Japanese',
            'nb': 'Norwegian Bokmal',
            'pt': 'Portuguese',
            'es': 'Spanish',
            'sv': 'Swedish'
        }

        # Dictionary which holds each nlp model of each requested language.
        self.nlp_models = {
            language_code: None 
            for language_code in self.supported_languages.keys()
        }

        # If no model is provided, assume one of the best default ones, 
        # at the time of writing.
        if model is None:
            self.hardcoded_model = False

            if language_setting == 'english':    
                self.model = SentenceTransformer('all-mpnet-base-v2')
                self.nlp_models['en'] = retrieve_spacy_model(
                    'en', self.supported_languages
                )
            else:
                self.multilingual_model = SentenceTransformer(
                    'paraphrase-multilingual-mpnet-base-v2'
                )
                
        else:
            self.hardcoded_model = True
            self.model = SentenceTransformer(model)

        self.text = None
        self.doc = None


    def extract_candidate_keyphrases(
            self: LMRank, text: str, language_code: str, top_n: int, 
            sentence_seps: str = '!?.', deduplicate: bool = True, 
            keep_nouns_adjs: bool = True,
        ) -> List[Tuple[str, int]]:

        """
        This method creates the document object from text, and stores 
        it in the class object. Then it extracts candidate keyphrases 
        using spacy's noun chunks, while storing their relative position 
        from the beginning of the document. This also removes sentence 
        separators that spacy may erroneously keep from the end of the
        string. Candidate keyphrase that start with pronouns or a particle
        (e.g. 'not') or have a length less than two characters or start 
        with a number are not kept.

        Input:
            text: (str)
                The original text.

            language_code: (str)
                The language code for the language the text is written in.

            top_n: (int)
                The top_n keyphrases to be searched in the list of close matches.

            sentence_seps: (str)
                String containing delimiters that separate sentences.
                These are removed from the end of each candidate keyphrase.

            deduplicate: (bool)
                A boolean flag that controls if the candidate keyphrases
                are deduplicated or not. (Default = False)

            keep_nouns_adjs: (bool)
                A boolean flag that controls if the candidate keyphrases
                are composed only from nouns, proper nouns and adjectives.

        Output: 
            <object>: (List[Tuple[str, int]])
                A list of tuples containing the candidate keyphrases and
                their first positional occurence.

        State:
            Modifies the text and doc members.
        """

        # Select the requested nlp model given the language code.
        nlp = self.nlp_models[language_code]
        
        # Clear the text from unnecessary whitespace.
        self.text = ' '.join(text.split())

        # Pass the text into the NLP pipeline 
        # The document object is stored in the class state.
        self.doc = nlp(self.text)

        # Form the candidate keyphrases either with noun phrases or dependency parsing.
        if language_code == 'el':
            candidate_keyphrases = form_candidate_keyphrases_as_noun_phrases(nlp, self.doc, sentence_seps)
        else:
            candidate_keyphrases = [
                (remove_last_seps(chunk.text.lower(), sentence_seps), chunk.start)
                for chunk in self.doc.noun_chunks
                if chunk.text.lower() not in nlp.Defaults.stop_words
                and chunk[0].pos_ not in {'PRON', 'PART'}
                and all(
                    term.pos_ in {'PROPN','NOUN', 'ADJ'} 
                    if keep_nouns_adjs else True for term in chunk
                )
                and len(chunk.text) > 2
                and not chunk.text[:1].isdigit()
            ]

            # Sort noun chunks by keyphrase text and groupby duplicate entries.
            # Only the first occurence of each keyphrase is preserved.
            candidate_keyphrases = {
                key: next(group)[1]
                for key, group in groupby(
                    sorted(candidate_keyphrases, key = itemgetter(0)), 
                    itemgetter(0))
            }

        # Removing near string duplicates from the candidate_keyphrases.
        # Get close matches is utilized, which finds the top-n most 
        # similar close matches. The first result is discarded, 
        # as it is the keyphrase we are using in the search.
        # Since the candidate keyphrases are already sorted, 
        # the shortest phrase is kept.
        if deduplicate:
            # Use a higher string similarity cutoff for non-english languages.
            string_similarity = 0.65 if language_code == 'en' else 0.75

            for item in list(candidate_keyphrases):
                close_matches = get_close_matches(item, candidate_keyphrases.keys(), 
                                                  cutoff = 0.65, n = top_n)[1:]
                for close_match in close_matches:
                    # This check removes overlapping keywords and longer overlapping keyphrases.
                    if not item.count(' '):
                        candidate_keyphrases.pop(item, None)
                        break
                    elif (len(close_match) > len(item)
                           and len(get_close_matches(item, [close_match], n = 1, cutoff = string_similarity))):
                        candidate_keyphrases.pop(close_match, None)      

        return list(candidate_keyphrases.items())

    
    def encode(
            self: LMRank, string_list: List[str], language_code: str, 
            multi_processing: bool = False, device: str = 'cpu'
        ) -> numpy.typing.NDArray[Any]:
        """
        Wrapper method, which toggles multiprocessing and 
        selects a computing device, then encodes the sentence 
        list using sentence transformers model.

        Input:
            string_list: (List[str])
                A list containing textual strings to be encoded.

            language_code: (str)
                The language code for the language the strings are written in.

            multi_processing: (bool)
                Boolean flag. If true, multi-processing is used.
                If false, multi-threading is used.

            device: (str)
                Device type to parallelize the embedding calculations.
                Default: cpu (multi-threading).
                Can be set to e.g. 'cuda:0' (1st GPU) or 'cuda' (for multiple GPUs)
        
        Output: 
            <object>: (numpy.array[numpy.array[numpy.float32]])
                A numpy array which stores each embedding as a numpy array.
        """

        # Select the correct transformers model given the requested language code.
        if language_code == 'en':
            model = self.model
        else:
            if self.hardcoded_model:
                model = self.model
            else:
                model = self.multilingual_model

        if multi_processing:
            # Start a multi process pool and select the computing device.
            pool = model.start_multi_process_pool(target_devices = [device])
            embeddings = model.encode_multi_process(string_list, pool)
            # Stop the multi process pool.
            model.stop_multi_process_pool(pool)
        else: # This uses multi_threading and is faster on shorter texts.
            embeddings = model.encode(string_list, device = device)
        return embeddings


    def model_token_length(self: LMRank, language_code: str) -> int:
        """
        Wrapper method that returns the maximum input length of the model,
        depending on the provided language code.

        Input: 
            language code: (str)
                The provided language code.

        Output: 
            <object>: (int)
                The maximum input length of the model.
        """
        # Select the correct transformers model given the provided language code.
        if language_code == 'en':
            model = self.model
        else:
            if self.hardcoded_model:
                model = self.model
            else:
                model = self.multilingual_model

        return model.max_seq_length


    def get_keyphrases_embeddings(
            self: LMRank, candidate_keyphrases: List[Tuple[str, List[int]]],
            language_code: str
        ) -> numpy.typing.NDArray[Any]:

        """
        Method which encodes each candidate keyphrase to an embedding.
        
        Input:
            candidate_keyphrases: (List[Tuple[str, List[int]]])
                A list of tuples containing the candidate keyphrases and
                a list containing their positional appearances.

            language_code: (str)
                The language code for the language the text is written in.
        
        Output: 
            <object>: (numpy.array[numpy.array[numpy.float32]])
                A numpy array which stores each embedding as a numpy array.
        """
   
        embeddings = self.encode(
            [keyphrase for keyphrase, _ in candidate_keyphrases],
            language_code
        )
        return embeddings


    def get_document_embedding(
            self: LMRank, language_code: str, split_on_chunks: bool = True
        ) -> numpy.typing.NDArray[numpy.float32]:
        """
        Method which calculates the document embedding.
        In the first case, it creates an embedding for the entire document,
        if the document length is not larger than the max input 
        length of the model. In the second case, it chunks the text base 
        on the max input token length of the model and calculates the document 
        embedding as the mean of all encoded chunk embeddings. In the third case,
        it splits the text into sentences, and then calculates the document
        embedding as the mean of all encoded sentence embeddings.

        Input:
            language_code: (str)
                The language code for the language the text is written in.
            
            split_on_chunks: (bool)
                Boolean flag. If set to true, the document embedding is calculated
                in chunks instead of sentences, this operation is faster 
                due to less encode() calls.

        Output:
            <object>: (numpy.array[numpy.float32])
            A numpy array which represents the document embedding.
        """

        # As written in the publication of the LMRank approach,
        # Some languages (e.g., Chinese, Japanese, Korean etc.)
        # do not use whitespace for word separation,
        # thus the text chunking technique cannot work.
        if language_code in {'zh', 'ja', 'ko'}:
            split_on_chunks = False
    
        # If the text is smaller than the max input token length
        # of the model, we directly calculate its embedding.
        if len(self.doc) <= self.model_token_length(language_code):
            document_embedding = self.encode(self.text, language_code)
        elif split_on_chunks:
            # Calculate the document embedding as 
            # the mean embedding of the chunk embeddings (faster).
            chunks = create_chunks(self.text, self.model_token_length(language_code))
            document_embedding = numpy.mean(
                self.encode(chunks, language_code), axis = 0
            )
        else:
            # Calculate the document embedding as 
            # the mean embedding of the sentence embeddings (slower).
            sentences = [sentence.text for sentence in self.doc.sents]
            document_embedding = numpy.mean(
                self.encode(sentences, language_code), axis = 0
            )

        return document_embedding


    def calculate_positional_scores(
            self: LMRank, candidate_keyphrases: List[Tuple[str, int]]
        ) -> numpy.typing.NDArray[numpy.float32]:
        """
        This function is used to calculate the positional score 
        for each candidate keyphrase.

        Input: 
            candidate_keyphrases: (List[Tuple[str, int]])
                A list of tuples containing the candidate keyphrases and
                an integer containing their first positional occurences.
        
        Output:
            <object>: (numpy.array(float32))
                A numpy array which holds a positional score for each candidate keyphrase.
        """
        scores = numpy.array([
            1 / (position + 1)
            for _, position in candidate_keyphrases
        ])

        # Normalize the scores with the softmax function.
        e_scores = numpy.exp(scores - numpy.max(scores))
        scores = e_scores / e_scores.sum(axis = 0)

        return scores


    def extract_keyphrases(
            self: LMRank, text: str, language_code: str, top_n: int = 10, 
            sentence_seps: str = '.?!', deduplicate: bool = True, 
            split_on_chunks: bool = True, keep_nouns_adjs: bool = True,
            positional_feature: bool = True,
        ) -> List[Tuple[str, float]]:

        """
        Method which extracts the candidate keyphrases,
        calculates their embeddings and the document embedding,
        ranks these candidate keyphrases according to their 
        similarity descending (optionally multiplied by the positional score).
        and then returns the top_n alongside their computed scores.

        Input: 
            text: (str)
                The original text.

            language_code: (str)
                The language code for the language the text is written in.

            top_n: (int)
                The number of top_n extracted keyphrases (default = 10)

            sentence_seps: (str)
                String containing delimiters that separate sentences.
                These are removed from the end of each candidate keyphrase.

            deduplicate: (bool)
                Boolean flag, that removes close string matches during
                the candidate keyphrase extraction stage.

            split_on_chunks: (bool)
                Boolean flag. If set to true, the document embedding is calculated
                in chunks instead of sentences, this operation is faster 
                due to less encode() calls.

            positional_feature: (bool)
                Boolean flag that enables the positional feature.
                When set to True the similarity score is multiplied by the
                positional score. The main concept behind this feature is to rank
                keyphrases that appear closer to the document beginning higher.
                This technique leads to performance improvements as mentioned
                in the literature.

            keep_nouns_adjs: (bool)
                A boolean flag that controls if the candidate keyphrases
                are composed only from nouns, proper nouns and adjectives.

        Output: 
            <object>: (List[Tuple[str, float]])
                The final list of candidate keyphrases and their respective scores.
        """
        
        # Check if the language code is supported.
        if language_code not in self.supported_languages:
            raise ValueError(f'{language_code} is not supported. The supported language codes are: {self.supported_languages}')

        # Check if any transformer models require loading.
        # This loads the most accurate semantic search models at the time of writing.
        if not self.hardcoded_model:
            if language_code == 'en' and self.model is None: 
                self.model = SentenceTransformer('all-mpnet-base-v2')
            if language_code != 'en' and self.multilingual_model is None:
                self.multilingual_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

        # Check if the selected nlp model requires loading.
        if self.nlp_models[language_code] is None:
            self.nlp_models[language_code] = retrieve_spacy_model(language_code, self.supported_languages)

        # Extract the candidate keyphrases.
        candidate_keyphrases = self.extract_candidate_keyphrases(
            text, language_code, top_n, sentence_seps, deduplicate, keep_nouns_adjs
        )

        # If there are no candidate keyphrases in the case of empty text,
        # Then early return an empty list.
        if not candidate_keyphrases:
            return []

        # Retrieve the keyphrases and the document embeddings.
        embeddings = self.get_keyphrases_embeddings(candidate_keyphrases, language_code)
        document_embedding = numpy.atleast_2d(
            self.get_document_embedding(language_code, split_on_chunks)
        )

        # Create the ids and embeddings numpy arrays.
        # The ids refer to the index of the candidate keyphrase in the unranked list.
        unranked_ids = numpy.array(range(len(embeddings))).astype(numpy.int64)

        # Find the embedding dimension.
        embedding_dim = len(embeddings[0])

        # Create the inner product index 
        # This is equivalent to a cosine similarity index given normalized embeddings.
        index = faiss.index_factory(embedding_dim, 'IDMap,Flat', faiss.METRIC_INNER_PRODUCT)

        # Normalize the embeddings array.
        faiss.normalize_L2(embeddings)

        # Add the embedding storage to the index.
        index.add_with_ids(embeddings, unranked_ids)

        # Normalize the document embedding 
        # and search for similar candidate keyphrases.
        faiss.normalize_L2(document_embedding)

        # Instead of limiting the search to top-n most similar, we need the entire list
        # as we need to multiply each entry with its positional score.
        similarities, ranked_ids = index.search(document_embedding, len(candidate_keyphrases))

        if positional_feature:
            scores = self.calculate_positional_scores(candidate_keyphrases)

            # We select the top_n keyphrases based on similarity and position.
            # In both cases the ids and similarities are 2d arrays because of a faiss requirement.
            # However their dimensions are 1 x #candidate_keyphrases.
            ranked_list = sorted([
                (candidate_keyphrases[key_id][0], sim * score)
                for key_id, sim, score in zip(ranked_ids[0], similarities[0], scores)
            ], key = itemgetter(1), reverse = True)
        else:
            # We select the top_n keyphrases based only on similarity.
            ranked_list = [
                (candidate_keyphrases[key_id][0], sim)
                for key_id, sim in zip(ranked_ids[0], similarities[0])
            ]

        return ranked_list[:top_n]
