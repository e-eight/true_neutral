import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gensim
import pandas as pd
from summa import summarizer


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
    model: gensim.models.doc2vec.Doc2Vec,
    titles: List[str],
    authors: List[str],
    genres: List[str],
    summaries: List[str],
    train_corpus: List[Any],
    test_corpus: List[List[str]],
    fname_tag: str,
    folder: str = "data",
):
    """
    Saves the trained model, and associated book metadata to file.

    Args:
        model (gensim.models.doc2vec.Doc2Vec): Trained Doc2Vec model.

        titles (List[str]): List of all the book titles in the data.

        authors (List[str]): List of authors corresponding to each title.

        genres (List[str]): List of genres corresponding to each title.

        summaries (List[str]): List of summaries corresponding to each title.

        train_corpus (List[Any]): The training corpus for the Doc2Vec model.

        test_corpus (List[List[str]]): The test corpus for the Doc2Vec model.

        fname_tag (str): A tag to identify the data.

        folder (str, optional): Folder where the data will be stored. If none
        is provided then it will be stored in a folder named `data` in the
        present location.
    """
    p = Path(folder)
    if not p.exists():
        p.mkdir(parents=True)

    model.save(folder + fname_tag + ".trainedmodel")
    with open(folder + fname_tag + "_titles.pkl", "wb") as f:
        pickle.dump(titles, f)
    with open(folder + fname_tag + "_authors.pkl", "wb") as f:
        pickle.dump(authors, f)
    with open(folder + fname_tag + "_genres.pkl", "wb") as f:
        pickle.dump(genres, f)
    with open(folder + fname_tag + "_summaries.pkl", "wb") as f:
        pickle.dump(summaries, f)
    with open(folder + fname_tag + "_train_corpus.pkl", "wb") as f:
        pickle.dump(train_corpus, f)
    with open(folder + fname_tag + "_test_corpus.pkl", "wb") as f:
        pickle.dump(test_corpus, f)
    return


def load_trained_model(fname_tag: str):
    """
    Loads the trained model and associated book metadata from file.

    Args:
        fname_tag (str): A tag to identify the model and the data files. Can
        include the folder path.

    Returns:
        A tuple containing the model, list of titles, list of authors, list of
        genres, list of summaries, training corpus, and test corpus.
    """

    model = gensim.models.Word2Vec.load(fname_tag + ".trainedmodel")
    with open(fname + "_titles.pkl", "rb") as f:
        titles = pickle.load(f)
    with open(fname + "_authors.pkl", "rb") as f:
        authors = pickle.load(f)
    with open(fname + "_genres.pkl", "rb") as f:
        genres = pickle.load(f)
    with open(fname + "_summaries.pkl", "rb") as f:
        summaries = pickle.load(f)
    with open(fname + "_train_corpus.pkl", "rb") as f:
        train_corpus = pickle.load(f)
    with open(fname + "_test_corpus.pkl", "rb") as f:
        test_corpus = pickle.load(f)
    return model, titles, authors, genres, summaries, train_corpus, test_corpus


def train_model(model: gensim.models.doc2vec.Doc2Vec, data: pd.DataFrame):
    """
    Trains a Doc2Vec model, on the given data.

    Args:
        model (gensim.models.doc2vec.Doc2Vec): A Doc2Vec model which is to be
        trained.

        data (pd.DataFrame): Data on which the model is trained. Must have the
        columns `Title`, `Author`, `Genres`, `Summary`.

    Returns:
        A tuple containing the trained model, list of titles, list of authors,
        list of genres, list of summaries, training corpus, and test corpus.
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

    return model, titles, authors, genres, summaries, train_corpus, test_corpus


def get_similar_books(
    model_data: Tuple[Any], book: str, nsim: int = 10, show_summary: bool = False
):
    """
    Get books similar to the given book.

    Args:
        model_data (Tuple[Any]): A tuple containing the trained model, and
        associated metadata, including list of titles, list of authors, list of
        genres, list of summaries, training corpus, and test corpus.

        book (str): Title of the book to be compared to.

        nsim (int, optional): Number of similar books. Default is 10.

        show_summary (bool, optional): If true, shows the summaries of the
        similar books. Default is False.

    Returns:
        A list of similar books.
    """

    model, titles, authors, genres, summaries, train_corpus, test_corpus = model_data

    if book not in titles:
        raise ValueError("Book not found.")

    idx = titles.index(book)
    query_tokens = gensim.utils.simple_preprocess(summaries[idx])
    inferred_vector = model.infer_vector(query_tokens)
    sims = model.docvecs.most_similar([inferred_vector], topn=nsim + 1)

    books = []
    for s in sims:
        new_book = {
            "Title": titles[s[0]],
            "Author": authors[s[0]],
            "Genres": genres[s[0]],
            "Correlation": s[1],
        }
        if show_summary:
            new_book["Summary"] = summaries[s[0]]
        books.append(new_book)
    return books[1:]  # The most similar book is itself.


def pprint_books(books: List[Dict]):
    """
    Pretty prints books.

    Args: books (List[Dict]): List of books. The metadata for each book is
        stored as a dictionary with the keys `Title`, `Author`, `Genres`,
        `Correlation`, and `Summary` (optional), where `Correlation` denotes the
        similarity between these books, and the book that they have been
        compared to.

    """
    # Get if `Summary` is present or not.
    nkeys = len(books[0].keys())

    for book in books:
        print(book["Title"] + " by " + book["Author"])
        print("Genres: " + book["Genres"])
        print(f'Correlation: {book["Correlation"]:.2f}')
        if nkeys == 5:
            print("Summary:")
            print(summarizer.summarize(book["Summary"]))
        print("\n")
