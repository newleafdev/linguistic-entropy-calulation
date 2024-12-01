import requests
from bs4 import BeautifulSoup
import os
import csv
import re

# Base URL for constructing full links
BASE_URL = "https://hakuteikoubou.skr.jp/storage/script_FFT/"

# Output directory
OUTPUT_DIR = "scraped_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def fetch_menu_links(menu_url):
    """
    Fetch all menu links from the given URL.
    """
    response = requests.get(menu_url)
    response.encoding = 'utf-8'  # Adjust encoding if needed
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all menu links
    menu_links = []
    for link in soup.find_all('a', href=True):
        # Construct full URLs for the menu links
        full_url = BASE_URL + link['href']
        menu_links.append(full_url)

    return menu_links

def scrape_page_sentences(page_url):
    """
    Scrape and extract sentences from a single page.
    """
    response = requests.get(page_url)
    response.encoding = 'utf-8'
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract all visible text (exclude script, style, and irrelevant tags)
    for irrelevant in soup(["script", "style", "img", "input"]):
        irrelevant.decompose()
    raw_text = soup.body.get_text(separator="\n", strip=True)

    # Split text into sentences using regex
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|！|。)\s', raw_text)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]  # Clean up whitespace
    return sentences

def save_to_csv(data, filename):
    """
    Save scraped sentences to a CSV file.
    """
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, mode='w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Page URL', 'Sentence'])  # Add headers
        writer.writerows(data)

def main():
    # Step 1: Define the menu URL
    menu_url = BASE_URL + "FFT_scriptmenu.html"

    # Step 2: Fetch menu links
    print(f"Fetching menu links from {menu_url}...")
    menu_links = fetch_menu_links(menu_url)
    print(f"Found {len(menu_links)} pages to scrape.")

    # Step 3: Scrape sentences from each page
    all_data = []
    for idx, page_url in enumerate(menu_links):
        print(f"Scraping page {idx + 1}/{len(menu_links)}: {page_url}...")
        sentences = scrape_page_sentences(page_url)
        for sentence in sentences:
            all_data.append([page_url, sentence])  # Collect URL and sentences

    # Step 4: Save the data to a CSV file
    save_to_csv(all_data, "scraped_sentences.csv")
    print(f"Scraping completed. Sentences saved to '{OUTPUT_DIR}/scraped_sentences.csv'.")

if __name__ == "__main__":
    main()
