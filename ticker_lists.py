import pandas as pd
import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent

ua = UserAgent()

def fetch_tickers_from_table(url: str) -> list[str]:
    """Fetches a list of ticker symbols from a StockAnalysis.com market cap list."""
    headers = {'User-Agent': ua.random}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")

    table = soup.find("table")
    if not table:
        return []

    headers = [th.text.strip() for th in table.find_all("th")]
    df = pd.DataFrame(columns=headers)

    for row in table.find_all("tr"):
        cells = row.find_all("td")
        row_data = [cell.text.strip() for cell in cells]
        if len(row_data) == len(df.columns):
            df.loc[len(df)] = row_data

    return df["Symbol"].tolist() if "Symbol" in df.columns else []

def fetch_micro_cap_tickers(url: str) -> list[str]:
    """Fetches ticker symbols from micro-cap table (different HTML structure)."""
    headers = {'User-Agent': ua.random}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")

    table = soup.find("table")
    if not table or not table.find("tbody"):
        return []

    cells = table.find("tbody").find_all("td")
    return [cells[i].text.strip() for i in range(1, len(cells), 7)]

# URLs for different market caps
urls = {
    "mega": "https://stockanalysis.com/list/mega-cap-stocks/",
    "large": "https://stockanalysis.com/list/large-cap-stocks/",
    "mid": "https://stockanalysis.com/list/mid-cap-stocks/",
    "small": "https://stockanalysis.com/list/small-cap-stocks/",
    "micro": "https://stockanalysis.com/list/micro-cap-stocks/"
}

# Fetch tickers
mega_caps = fetch_tickers_from_table(urls["mega"])
large_caps = fetch_tickers_from_table(urls["large"])
mid_caps = fetch_tickers_from_table(urls["mid"])
small_caps = fetch_tickers_from_table(urls["small"])
micro_caps = fetch_micro_cap_tickers(urls["micro"])

# Combine all tickers into a single list
all_tickers = mega_caps + large_caps + mid_caps + small_caps + micro_caps

