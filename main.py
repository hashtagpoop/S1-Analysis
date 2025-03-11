import sec_parser as sp
from sec import download_filing
from latest_ipos import get_latest_ipos
from typing import cast
import os
from llm import LLM
from dotenv import load_dotenv
import argparse

# Load environment variables
load_dotenv()

MAX_FILINGS_TO_PARSE = 10

# Add argument parser
parser = argparse.ArgumentParser(description='Parse S-1 filings and analyze lockup periods')
parser.add_argument('--analyze', action='store_true', help='Use GPT-4 to analyze lockup texts')
args = parser.parse_args()

# Initialize LLM if analyze flag is set
llm = None
if args.analyze:
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    llm = LLM.create(
        provider_type="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o"
    )

# TODO: need to see if we are missing any lockups or common share offered.
# TODO: need to tune this to make sure we capture all edge cases.
# Method that gets the lockup text along with all the other
# text within the given section so it pulls all the context of lockups.
def extract_lockup_text(
    tree: sp.SemanticTree,
    keywords=("lock-up", "lockup", "lock up")
) -> list[str]:
    # Given that the sec_parser library already gets us the tree structure
    # we can just traverse the tree and collect the text of the nodes that
    # mention the lockup keywords.
    def dfs(node: sp.TreeNode, level=0, collecting=False):
        # Check if the current node's text contains any of the keywords
        node_text = node.semantic_element.text.lower()
        if any(keyword in node_text for keyword in keywords):
            collecting = True

        # If collecting is True, add the node's text to the collected_text list
        if collecting:
            collected_text.append(node.semantic_element.text)

        # Recursively call dfs on each child node
        for child in node.children:
            dfs(child, level + 1, collecting)

    collected_text = []
    # Start the DFS from each top-level section
    top_level_sections = [item for part in tree for item in part.children]
    for section in top_level_sections:
        dfs(section)

    return collected_text

could_not_find_filing = []
latest_ipos = get_latest_ipos()

for i, row in latest_ipos.iterrows():
    try:
        s1_content = download_filing(ticker=row["ticker"], form="S-1")
        elements: list = sp.Edgar10QParser().parse(s1_content)
        tree: sp.SemanticTree = sp.TreeBuilder().build(elements)
        lockup_texts = extract_lockup_text(tree)
        
        for lockup_text in lockup_texts:
            if args.analyze and llm:
                print(f"\nAnalyzing lockup texts for {row['ticker']}:")
                # TODO: the prompt needs a lot more tuning.
                prompt = f"""
                Analyze the following lockup text from an S-1 filing and extract (I want this information for each type of share):
                
                For example:
                directors, 100000 shares, 1 year, common, quote here
                
                1. The number of shares affected
                2. The lockup period duration
                3. The type of shares (common stock, preferred, etc.)
                4. Who the lockup applies to (directors, employees, etc.)
                5. Quote where you got it from
                
                Format the response as a structured CSV.
                
                Text to analyze:
                {lockup_text}
                """
                analysis = llm.complete(prompt)
                print("\nLLM Analysis:")
                print(analysis)
                
            print("-" * 100)
        
    except Exception as e:
        could_not_find_filing.append(row["ticker"])
        print(f"Unable to find S1 filing for {row['ticker']}: {e}")
        
    if i > MAX_FILINGS_TO_PARSE: break
    
print(f"Could not find filing for the following tickers: {', '.join(could_not_find_filing)}")