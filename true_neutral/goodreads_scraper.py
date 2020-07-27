from typing import List, DefaultDict, Tuple
from pathlib import Path
from collections import defaultdict
import time
import re
import requests
from bs4 import BeautifulSoup
from multiprocessing import Pool, cpu_count
from itertools import repeat
import jsonlines


def get_book_links(
    list_name: str, page_range: Tuple[int, int], user_agent: str
) -> List[str]:
    """
    Function to get links to books from a GoodReads list,
    https://www.goodreads.com/list/show/list_name?page=page_no.

    Parameters:
    ===========
    list_name: str
        Name of the Rising Shadow list.

    page_range: Tuple[int, int]
        Starting and ending page numbers. The numbers should be ≥ 1.

    user_agent: str
        User agent to send to the site.

    Returns:
    ========
    book_links: List[str]
        List containing urls to all the books on the Rising Shadow list.
    """

    book_links = []
    headers = {"User-Agent": user_agent}

    goodreads_url_prefix = "https://www.goodreads.com"
    list_url_prefix = goodreads_url_prefix + f"/list/show/{list_name}"

    start_page_no, end_page_no = page_range

    try:
        for page in range(start_page_no, end_page_no + 1):
            page_url = list_url_prefix + f"?page={page}"
            page = requests.get(page_url, headers=headers)

            if page.status_code == 200:
                soup = BeautifulSoup(page.text, "html.parser")

                for link in soup.select(".bookTitle"):
                    book_links.append(goodreads_url_prefix + link["href"])
    except Exception as ex:
        print(str(ex))
    finally:
        return book_links


def write_books(book_links: List[str], user_agent: str, nprocs: int, out: str):
    """
    Function to save the details (title, author(s), genre(s), description) of
    books on a GoodReads list to file.

    Parameters:
    ===========
    book_links: List[str]
        List containing urls to all the books on the GoodReads list.

    user_agent: str
        User agent to send to the site.

    nprocs: int
        Number of processes for multiprocessing.

    out: str
        Name of file to write the book details to. This will be in the JSON
        Lines text format.
    """

    with Pool(nprocs) as p:
        books = p.starmap(get_single_book, zip(book_links, repeat(user_agent)))

    with jsonlines.open(out, mode="w") as writer:
        for book in books:
            writer.write(book)
    return


def get_single_book(url: str, user_agent: str) -> DefaultDict[str, str]:
    """
    Function to get the details (title, author(s), genre(s), description) of a
    book from a GoodReads book page,
    https://www.goodreads.com/book/show/book-name.

    Parameters:
    ===========
    url: str
        Link to the Rising Shadow book page.

    user_agent: str
        User agent to send to the site.

    Returns:
    ========
    book: DefaultDict[str, str]
        Dictionary object with book details.
    """

    book = defaultdict(str)
    headers = {"User-Agent": user_agent}

    try:
        page = requests.get(url, headers=headers)
        time.sleep(20)

        if page.status_code == 200:
            soup = BeautifulSoup(page.text, "html.parser")
            book["id"] = get_id(url)
            book["title"] = get_title(soup)
            book["authors"] = get_authors(soup)
            book["genres"] = get_genres(soup)
            book["description"] = get_description(soup)
    except Exception as ex:
        print(str(ex))
    finally:
        return book


def get_id(url: str) -> int:
    """
    Function to get the id of a book from a GoodReads book page,
    https://www.goodreads.com/book/show/book-name. The id is part of the
    book-name: /book/show/{id}-{rest of book-name} or /book/show/{id}.{rest of book-name}.

    Parameters:
    ===========
    url: str
        Link to the GoodReads book page.

    Returns:
    ========
    book_id: int
        Id of the book
    """

    book_id = re.sub(".*/book/show/\\s*|-.*", "", url)
    book_id = re.sub("\\..*", "", book_id)
    return int(book_id)


def get_title(book: BeautifulSoup) -> str:
    """
    Function to get the title of the book.

    Parameters:
    ===========
    book: BeautifulSoup
        BeautifulSoup object representing the book, created from
        https://www.goodreads.com/book/show/book-name.

    Returns:
    ========
    title: str
        Title of the book.
    """

    title = book.find("h1", id="bookTitle").text.strip()
    return title


def get_authors(book: BeautifulSoup) -> List[str]:
    """
    Function to get the author(s) of the book.

    Parameters:
    ===========
    book: BeautifulSoup
        BeautifulSoup object representing the book, created from
        https://www.goodreads.com/book/show/book-name.

    Returns:
    ========
    authors: List[str]
        Author(s) of the book
    """

    authors = [a.text.strip() for a in book.find_all("a", class_="authorName")]
    return authors


def get_genres(book: BeautifulSoup) -> List[str]:
    """
    Function to get the genre(s) of the book.

    Parameters:
    ===========
    book: BeautifulSoup
        BeautifulSoup object representing the book, created from
        https://www.goodreads.com/book/show/book-name.

    Returns:
    ========
    genres: List[str]
        Genre(s) of the book
    """

    genres = [
        g.text.strip()
        for g in book.find_all(
            "a", class_="bookPageGenreLink", href=re.compile("/genres/*")
        )
    ]
    return genres


def get_description(book: BeautifulSoup) -> str:
    """
    Function to get book description from GoodReads.

    Parameters:
    ===========
    book: BeautifulSoup
        BeautifulSoup object representing the book, created from
        https://www.goodreads.com/book/show/book-name.

    Returns:
    ========
    desc: str
        Description of the book given on GoodReads.
    """

    desc = ""
    try:
        desc_list = book.find("div", id="descriptionContainer").find_all("span")
        if desc_list:
            desc = desc_list[-1].text.strip()
    except Exception as ex:
        print(str(ex))
    finally:
        return desc


def main(list_name: str, page_range: Tuple[int, int], user_agent: str, out: str):
    book_links = get_book_links(list_name, page_range, user_agent)

    nprocs = max(cpu_count() - 1, 1)
    write_books(book_links, user_agent, nprocs, out)


if __name__ == "__main__":
    list_name = "3.Best_Science_Fiction_Fantasy_Books"
    page_range = (1, 5)

    user_agent = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.87 Safari/537.36"

    Path("data").mkdir(parents=True, exist_ok=True)
    out = "data/goodreads_best_science_fiction_fantasy_01_05.jsonl"

    main(list_name, page_range, user_agent, out)
