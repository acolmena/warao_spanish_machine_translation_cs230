import json
from bs4 import BeautifulSoup
import requests
import time
import re


# --- Function to fetch content from a URL ---
def fetch_html_from_url(url):
    """
    Fetches the HTML content from the given URL.
    Includes a polite delay.
    """
    print(f"Fetching content from: {url}")
    time.sleep(1)  # Polite delay

    try:
        # Use a standard User-Agent header to mimic a browser, which can sometimes prevent issues
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        # Raise an exception for bad status codes (4xx or 5xx)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}")
        return None


def extract_bible_chapter(html_content):
    """
    Extracts structured Bible chapter text from the __NEXT_DATA__ JSON
    embedded within the provided HTML content.
    """
    if not html_content:
        return None

    soup = BeautifulSoup(html_content, 'html.parser')

    # 1. Find the script tag containing the JSON data
    next_data_script = soup.find('script', {'id': '__NEXT_DATA__'})

    if not next_data_script:
        print("Error: Could not find the '__NEXT_DATA__' script tag. The page structure may have changed.")
        return None

    try:
        # 2. Extract the JSON and parse it
        json_data = json.loads(next_data_script.string)

        # 3. Navigate the nested structure to find the verses
        page_props = json_data.get('props', {}).get('pageProps', {})
        chapter_text = page_props.get('chapterText', [])

        if not chapter_text:
            print("Error: 'chapterText' array is empty or not found in JSON data. Check JSON path.")
            return None

        book_name = page_props.get('activeBookName', 'Unknown Book')
        chapter_number = page_props.get('activeChapter', 'Unknown Chapter')

        formatted_content = []
        for verse in chapter_text:
            # We assume 'verse_start' is the verse number and 'verse_text' is the content
            verse_num = verse.get('verse_start')
            text = verse.get('verse_text', '').strip()

            # Format the output as: [V#] Text
            formatted_content.append(f"[{verse_num}] {text}")

        full_content = "\n".join(formatted_content)

        return {
            "title": f"{book_name} {chapter_number}",
            "content": full_content,
            "language_code": page_props.get('activeIsoCode', 'wba'),
            "source_type": "Embedded JSON"
        }

    except json.JSONDecodeError:
        print("Error: Failed to decode the JSON content from the script tag.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during JSON processing: {e}")
        return None


# --- Main Execution ---



print("Starting extraction...")

# NOTE: Replace this placeholder URL with the actual URL you want to scrape.
warao = {
    "MAT": list(range(1, 29)),
    "MRK": list(range(1, 17)),
    "LUK": list(range(1, 25)),
    "JHN": list(range(1, 22)),
    "ACT": list(range(1, 29)),
    "ROM": list(range(1, 17)),
    "1CO": list(range(1, 17)),
    "2CO": list(range(1, 14)),
    "GAL": list(range(1, 7)),
    "EPH": list(range(1, 7)),
    "PHP": list(range(1, 5)),
    "COL": list(range(1, 5)),
    "1TH": list(range(1, 6)),
    "2TH": list(range(1, 4)),
    "1TI": list(range(1, 7)),
    "2TI": list(range(1, 5)),
    "TIT": list(range(1, 4)),
    "PHM": list(range(1, 2)),
    "HEB": list(range(1, 14)),
    "JAS": list(range(1, 6)),
    "1PE": list(range(1, 6)),
    "2PE": list(range(1, 4)),
    "1JN": list(range(1, 6)),
    "2JN": list(range(1, 2)),
    "3JN": list(range(1, 2)),
    "JUD": list(range(1, 2)),
    "REV": list(range(1, 23)),

}
WARAO_URL = "https://live.bible.is/bible/WBABIV/{book}/{chapter}"

BASE_SAVE = "bible_chapter_output{book}_{chapter}.txt"

TARGET_URL = "https://live.bible.is/bible/WBABIV/MAT/1"

for book, chapters in warao.items():
    for chapter in chapters:
        url = WARAO_URL.format(book=book, chapter=chapter)
        # print(url)
        # Call the fetching function to get live HTML data
        html_data = fetch_html_from_url(url)


        article = extract_bible_chapter(html_data)

        # Save results
        if article:
            filename = f"warao_{book}_{chapter}.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(article['content'])

            # print(f"\nSuccessfully extracted and saved content to {filename}")
        else:
            print(
                "\nExtraction failed. Please check the 'TARGET_URL' and ensure the website's JSON structure remains the same.")


spanish = {
    "MAT": list(range(1, 29)),
    "MRK": list(range(1, 17)),
    "LUK": list(range(1, 25)),
    "JHN": list(range(1, 22)),
    "ACT": list(range(1, 29)),
    "ROM": list(range(1, 17)),
    "1CO": list(range(1, 17)),
    "2CO": list(range(1, 14)),
    "GAL": list(range(1, 7)),
    "EPH": list(range(1, 7)),
    "PHP": list(range(1, 5)),
    "COL": list(range(1, 5)),
    "1TH": list(range(1, 6)),
    "2TH": list(range(1, 4)),
    "1TI": list(range(1, 7)),
    "2TI": list(range(1, 5)),
    "TIT": list(range(1, 4)),
    "PHM": list(range(1, 2)),
    "HEB": list(range(1, 14)),
    "JAS": list(range(1, 6)),
    "1PE": list(range(1, 6)),
    "2PE": list(range(1, 4)),
    "1JN": list(range(1, 6)),
    "2JN": list(range(1, 2)),
    "3JN": list(range(1, 2)),
    "JUD": list(range(1, 2)),
    "REV": list(range(1, 23)),
}

SPANISH_URL = "https://live.bible.is/bible/SPNDLH/{book}/{chapter}"

for book, chapters in spanish.items():
    for chapter in chapters:
        url = SPANISH_URL.format(book=book, chapter=chapter)
        # print(url)
        html_data = fetch_html_from_url(url)


        article = extract_bible_chapter(html_data)

        # Save results
        if article:
            filename = f"spanish_{book}_{chapter}.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(article['content'])

            # print(f"\nSuccessfully extracted and saved content to {filename}")
        else:
            print(
                "\nExtraction failed. Please check the 'TARGET_URL' and ensure the website's JSON structure remains the same.")
