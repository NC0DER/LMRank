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

import spacy
import numpy
import faiss

from itertools import groupby
from operator import itemgetter
from difflib import get_close_matches
from sentence_transformers import SentenceTransformer
from LMRank.utils import *


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

    def __init__(self, model = 'all-mpnet-base-v2', language_code = 'en'):
        """
        Initialization method.

        Arguments: 
            model: A sentence transformers model.
                   (default is the highest scoring english model)
                   other models can be found in the following link:
                   https://www.sbert.net/docs/pretrained_models.html,
            language_code: (str)
                This follows spacy language codes.
                Spacy is used by this approach 
                for POS tagging, noun chunking 
                and sentence splitting.
        """
        self.model = SentenceTransformer(model)
        self.nlp = retrieve_spacy_model()
        self.text = None
        self.doc = None
        

    def extract_candidate_keyphrases(self, text, deduplicate = False, 
                                     keep_nouns_adjs = True, sentence_seps = '.?!'):

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
            text: str
                The original text 

            deduplicate: bool
                A boolean flag that controls if the candidate keyphrases
                are deduplicated or not. (Default = False)

            keep_nouns_adjs: bool
                A boolean flag that controls if the candidate keyphrases
                are composed only from nouns, proper nouns and adjectives.

            sentence_seps: str
                String containing delimiters that separate sentences.
                These are removed from the end of each candidate keyphrase.
        
        Output: List(Tuple(str, int))
            A list of tuples containing the candidate keyphrases and
            an integer containing their first positional occurence.

        State:
            Modifies the text and doc members.
        """
        
        # Clear the text from unnecessary whitespace.
        self.text = ' '.join(text.split())

        # Pass the text into the NLP pipeline 
        # and disable some of its unnecessary components.
        # The document object is stored in the class object.
        self.doc = self.nlp(
            self.text, 
            disable = ['ner', 'lemmatizer', 'textcat', 'custom']
        )

        candidate_keyphrases = [
            (remove_last_seps(chunk.text.lower()), chunk.start)
            for chunk in self.doc.noun_chunks
            if chunk.text.lower() not in self.nlp.Defaults.stop_words
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
        # Get close matches is utilized, which finds at most top-10 most 
        # similar close matches. The first result is discarded, 
        # since it is the keyphrase we are using in the search.
        # Since the candidate keyphrases are already sorted, 
        # the shortest phrase is kept.
        if deduplicate:
            for item in list(candidate_keyphrases):
                close_matches = get_close_matches(item, candidate_keyphrases.keys(), 
                                                  cutoff = 0.65, n = 10)[1:]
                for close_match in close_matches:
                    del candidate_keyphrases[close_match]

        return list(candidate_keyphrases.items())

    
    def encode(self, string_list, multi_processing = False, device = 'cpu'):
        """
        Wrapper method, which toggles multiprocessing and 
        selects a computing device, then encodes the sentence 
        list using sentence transformers model.

        Input:
            string_list: List((str))
                A list containing textual strings to be encoded.

            multi_processing: bool
                Boolean flag. If true, multi-processing is used.
                If false, multi-threading is used.

            device: str
                Device type to parallelize the embedding calculations.
                Default: cpu (multi-threading).
                Can be set to e.g. 'cuda:0' (1st GPU) or 'cuda' (for multiple GPUs)
        
        Output: np.array(np.array(float32))
            A numpy array which stores each embedding as a numpy array
        """
        if multi_processing:
            # Start a multi process pool and select the computing device.
            pool = self.model.start_multi_process_pool(target_devices = [device])
            embeddings = self.model.encode_multi_process(string_list, pool)
            # Stop the multi process pool.
            self.model.stop_multi_process_pool(pool)
        else: # This uses multi_threading and is faster on shorter texts.
            embeddings = self.model.encode(string_list, device = device)
        return embeddings


    def model_token_length(self):
        """
        Wrapper method that returns the maximum input length of the model.

        Input: None
        Output: int
        """
        return self.model.max_seq_length


    def get_keyphrases_embeddings(self, candidate_keyphrases):

        """
        Method which encodes each candidate keyphrase to an embedding.
        
        Input:
            candidate_keyphrases: List(Tuple(str, List(int)))
            A list of tuples containing the candidate keyphrases and
            a list containing their positional appearances.
        
        Output: np.array(np.array(float32))
            A numpy array which stores each embedding as a numpy array.
        """
   
        embeddings = self.encode(
            [keyphrase for keyphrase, _ in candidate_keyphrases],
            device = 'cpu'
        )
        return embeddings


    def get_document_embedding(self, split_on_chunks = True):
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
            split_on_chunks: bool
                Boolean flag. If set to true, the document embedding is calculated
                in chunks instead of sentences, this operation is faster 
                due to less encode() calls.
        
        Output: np.array(float32)
            A numpy array which represents the document embedding.
        """
    
        # If the text is smaller than the max input token length
        # of the model, we directly calculate its embedding.
        if len(self.doc) <= self.model_token_length():
            document_embedding = self.encode(self.text)
        elif split_on_chunks:
            # Calculate the document embedding as 
            # the mean embedding of the chunk embeddings (faster).
            chunks = create_chunks(self.text, self.model_token_length())
            document_embedding = numpy.mean(
                self.encode(chunks, device = 'cpu'), axis = 0
            )
        else:
            # Calculate the document embedding as 
            # the mean embedding of the sentence embeddings (slower).
            sentences = [sentence.text for sentence in self.doc.sents]
            document_embedding = numpy.mean(
                self.encode(sentences, device = 'cpu'), axis = 0
            )

        return document_embedding


    def calculate_positional_scores(self, candidate_keyphrases):
        """
        This function is used to calculate the positional score 
        for each candidate keyphrase.

        Input: 
            candidate_keyphrases: List(Tuple(str, int))
            A list of tuples containing the candidate keyphrases and
            an integer containing their first positional occurences.
        
        Output: numpy.array(float32)
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


    def extract_keyphrases(self, text, 
                           sentence_seps = '.?!',
                           deduplicate = False, 
                           split_on_chunks = True,
                           keep_nouns_adjs = True,
                           positional_feature = True,
                           top_n = 10):

        """
        Method which extracts the candidate keyphrases,
        calculates their embeddings and the document embedding,
        ranks these candidate keyphrases according to their 
        similarity descending (optionally multiplied by the positional score).
        and then returns the top_n alongside their computed scores.

        Input: 
            text: str
                The original text.

            deduplicate: bool
                Boolean flag, that removes close string matches during
                the candidate keyphrase extraction stage.

            sentence_seps: str
                String containing delimiters that separate sentences.
                These are removed from the end of each candidate keyphrase.

            split_on_chunks: bool
                Boolean flag. If set to true, the document embedding is calculated
                in chunks instead of sentences, this operation is faster 
                due to less encode() calls.

            positional_feature: bool
                Boolean flag that enables the positional feature.
                When set to True the similarity score is multiplied by the
                positional score. The main concept behind this feature is to rank
                keyphrases that appear closer to the document beginning higher.
                This technique leads to performance improvements as mentioned
                in the literature.

            keep_nouns_adjs: bool
                A boolean flag that controls if the candidate keyphrases
                are composed only from nouns, proper nouns and adjectives.

            top_n: int
                The number of top_n keyphrases to return (default = 10)
        
        Output: List(Tuple(str, float)
            The final list of candidate keyphrases and their respective scores.
        """

        # Extract the candidate keyphrases.
        candidate_keyphrases = self.extract_candidate_keyphrases(
            text, deduplicate, keep_nouns_adjs, sentence_seps
        )

        # If there are no candidate keyphrases in the case of empty text,
        # Then early return an empty list.
        if not candidate_keyphrases:
            return []

        # Retrieve the keyphrases and the document embeddings.
        embeddings = self.get_keyphrases_embeddings(candidate_keyphrases)
        document_embedding = numpy.atleast_2d(
            self.get_document_embedding(split_on_chunks)
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
