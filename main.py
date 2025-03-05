import sec_parser as sp
from sec import get_s1_filings, download_filing, create_edgar_url

MAX_FILINGS_TO_PARSE = 10

# Utility function to make the example code a bit more compact
def print_first_n_lines(text: str, *, n: int):
    print("\n".join(text.split("\n")[:n]), "...", sep="\n")

s1_filings = get_s1_filings(start_date="2024-01-02", end_date="2024-12-31")

for i, filings in enumerate(s1_filings):
    if i > MAX_FILINGS_TO_PARSE: break

    try:
        s1_content = download_filing(input_file=create_edgar_url(filings["cik"], filings["filing_id"]))

        print(f"Parsing S1 filing for {filings['display_names'][0]}")
        elements: list = sp.Edgar10QParser().parse(s1_content)
        demo_output: str = sp.render(elements)
        print_first_n_lines(demo_output, n=20)

    except Exception as e:
        print(f"Error parsing S1 filing for {filings['display_names'][0]}: {e}")
        continue