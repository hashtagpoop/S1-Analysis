import sec_parser as sp
from sec import download_filing
from latest_ipos import get_latest_ipos

MAX_FILINGS_TO_PARSE = 10

# Utility function to make the example code a bit more compact
def print_first_n_lines(text: str, *, n: int):
    print("\n".join(text.split("\n")[:n]), "...", sep="\n")

could_not_find_filing = []
latest_ipos = get_latest_ipos()

for i, row in latest_ipos.iterrows():
    try:
        s1_content = download_filing(ticker=row["ticker"], form="S-1")
        elements: list = sp.Edgar10QParser().parse(s1_content)
        demo_output: str = sp.render(elements)
        print_first_n_lines(demo_output, n=20)
    except Exception as e:
        could_not_find_filing.append(row["ticker"])
        print(f"Unable to find S1 filing for {row['ticker']}: {e}")
        
    if i > MAX_FILINGS_TO_PARSE: break
    
print(f"Could not find filing for the following tickers: {", ".join(could_not_find_filing)}")