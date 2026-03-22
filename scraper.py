import requests
from bs4 import BeautifulSoup


def scrape_page(url, label):
    print(f"Scraping: {url}")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    response = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(response.text, "html.parser")

    # Remove junk (menus, footers, scripts)
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    # Get the clean text
    text = soup.get_text(separator="\n", strip=True)

    # Save to a file
    filename = f"{label}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Saved to {filename}")
    return text

scrape_page("https://www.ameriabank.am/en/individuals/deposits", "ameriabank_deposits")