import pickle
from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple

import gensim
import pandas as pd
from summa import summarizer


class ModelData(NamedTuple):
    """
    A class for saving a trained Doc2Vec model and the associated data.
    """

    model: gensim.models.doc2vec.Doc2Vec
    titles: List[str]
    authors: List[str]
    genres: List[str]
    summaries: List[str]
    train_corpus: List[gensim.models.doc2vec.TaggedDocument]
    test_corpus: List[List[str]]


class Book(NamedTuple):
    """
    A class to save book metadata.
    """

    title: str
    author: str
    genres: str
    summary: str


def read_corpus(books: List[Dict], tokens_only: bool = False):
    """
    Reads in the corpus and associates a tag to each document in the training
    corpus.

    Args:
        books (List[Dict]): List of book metadata. The metadata is stored as a
        dictionary, and must contain a key called `Summary`. The `Summary` will
        be used to train the Doc2Vec model.

        tokens_only (bool, optional): If False, returns an iterator over Doc2Vec
        `TaggedDocument`s, else an iterator over tokens. Default is False.

    Returns:
        An iterator over Doc2Vec `TaggedDocument`s or over tokens.
    """
    ctr = -1
    for book in books:
        text = book["Summary"].strip()
        text = text.replace("\n", " ")
        tokens = gensim.utils.simple_preprocess(text)
        if tokens_only:
            yield tokens
        else:
            ctr += 1
            yield gensim.models.doc2vec.TaggedDocument(tokens, [ctr])


def save_trained_model(
    model_data: ModelData,
    folder: str,
    fname: str,
    protocol: int = 4,
):
    """
    Saves the trained model, and associated book metadata to file.

    Args:
        model_data (ModelData): Trained Doc2Vec model and associated data.

        folder (str): Directory in which model_data is to be saved.

        fname (str): File to which model_data is to be saved.

        protocol (int, optional): Indicates which protocol should be used by the
        pickler, default is 4. Check
        https://docs.python.org/3/library/pickle.html for details.

    """
    p = Path(folder)
    p.mkdir(parents=True, exist_ok=True)

    fp = p / fname
    with fp.open("wb") as f:
        pickle.dump(model_data, f, protocol=protocol)

    return


def load_trained_model(fname: str):
    """
    Loads a saved ModelData object from file.

    Args:
        fname (str): Complete path to the file.

    Returns: A ModelData object containing the model, list of titles, list of
        authors, list of genres, list of summaries, training corpus, and test
        corpus.

    """

    fp = Path(fname)

    with fp.open("rb") as f:
        model_data = pickle.load(f)
    return model_data


def train_model(model: gensim.models.doc2vec.Doc2Vec, data: pd.DataFrame):
    """
    Trains a Doc2Vec model, on the given data.

    Args:
        model (gensim.models.doc2vec.Doc2Vec): A Doc2Vec model which is to be
        trained.

        data (pd.DataFrame): Data on which the model is trained. Must have the
        columns `Title`, `Author`, `Genres`, `Summary`.

    Returns: A ModelData object containing the trained model, list of titles,
        list of authors, list of genres, list of summaries, training corpus, and
        test corpus.

    """
    books = data.to_dict("records")
    train_corpus = list(read_corpus(books))
    test_corpus = list(read_corpus(books, tokens_only=True))

    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

    titles = list(data["Title"].values)
    authors = list(data["Author"].values)
    genres = list(data["Genres"].values)
    summaries = list(data["Summary"].values)

    return ModelData(
        model, titles, authors, genres, summaries, train_corpus, test_corpus
    )


def get_similar_books(
    model_data: ModelData, title: str = None, summary: str = None, nsim: int = 10
):
    """
    Get books similar to the given book. At least either `title` or `summary`
    must be provided.

    Args:
        model_data (ModelData): Trained Doc2Vec model and associated data.

        title (str, optional): Title of the book to be compared to.

        summary (str, optional): Summary of the book to be compared to.

        nsim (int, optional): Number of similar books. Default is 10.

    Returns:
        A list of similar books.

    """

    titles = model_data.titles
    summaries = model_data.summaries
    summary_ = None

    if title and summary:
        if title in titles:
            summary_ = summaries[titles.index(title)]
        else:
            summary_ = summary
    elif title:
        if title in titles:
            summary_ = summaries[titles.index(title)]
        else:
            raise ValueError(
                "Title not in database. Please provide a summary of the book."
            )
    elif summary:
        summary_ = summary
    else:
        raise ValueError("Either title or summary of the book must be provided.")

    return _get_similar_books_helper(model_data, summary_, nsim)


def _get_similar_books_helper(model_data: ModelData, summary: str, nsim: int = 10):
    """
    Get books similar to the given summary of a book.

    Args:
        model_data (ModelData): Trained Doc2Vec model and associated data.

        summary (str): Summary of the book to be compared to.

        nsim (int, optional): Number of similar books. Default is 10.

    Returns:
        A list of similar books.
    """

    model, titles, authors, genres, summaries, _, _ = model_data

    query_tokens = gensim.utils.simple_preprocess(summary)
    inferred_vector = model.infer_vector(query_tokens)
    sims = model.docvecs.most_similar([inferred_vector], topn=nsim + 1)

    books = []
    for s in sims:
        book = (Book(titles[s[0]], authors[s[0]], genres[s[0]], summaries[s[0]]), s[1])
        books.append(book)
    return books[1:]  # The most similar book is itself.


def pprint_books(books: List[Tuple], show_summary: bool = False):
    """
    Pretty prints books.

    Args: books (List[Tuple]): List of books. The metadata for each book is
        stored as a tuple containing a Book object and a correlation score,
        which shows how similar is the book to the input.

        show_summary (bool, optional): If True, shows the summaries of the
        similar books. Default is False.

    """

    for book in books:
        print(book[0].title + " by " + book[0].author)
        print("Genres: " + book[0].genres)
        print(f"Correlation: {book[1]:.2f}")
        if show_summary:
            print("Short Summary:")
            print(summarizer.summarize(book[0].summary))
        print("\n")
