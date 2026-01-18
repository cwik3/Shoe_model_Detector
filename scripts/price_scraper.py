import re
from duckduckgo_search import DDGS


def clean_shoe_name(raw_name):
    # Converts 'nike_dunk_low_purple_court' -> 'Nike Dunk Low Purple Court'
    return raw_name.replace("_", " ").title()


def get_market_price(shoe_name):
    clean_name = clean_shoe_name(shoe_name)
    print(f"\n[Scraper]  Looking for: {clean_name}...")

    # We try two different queries to increase success chances
    queries = [
        f"{clean_name} price ebay",
        f"{clean_name} price goat sneaker"
    ]

    for q in queries:
        try:
            # Run the search
            results = DDGS().text(q, max_results=2)

            if not results:
                print(f"[Scraper] ❌ No results for '{q}'")
                continue

            # Combine all text found to search through it
            combined_text = " ".join([r.get('body', '') for r in results])

            # --- DEBUGGING: Uncomment this line to see what the internet sends back ---
            # print(f"[Debug Raw Text] {combined_text[:200]}...")

            # REGEX EXPLANATION:
            # \$?       -> Optional dollar sign
            # \s?       -> Optional space
            # \d{2,4}   -> 2 to 4 digits (e.g., 80, 100, 1200)
            # \s?       -> Optional space
            # (?:USD)?  -> Optional "USD" text
            # We look for simple patterns like "$120", "120 USD", "$ 140"
            price_pattern = r"(\$\s?\d{2,4})"

            matches = re.findall(price_pattern, combined_text)

            if matches:
                found_price = matches[0].replace(" ", "")  # Clean up spaces
                print(f"[Scraper] ✅ Found Price: {found_price}")
                return f"{found_price} (Est.)"

        except Exception as e:
            print(f"[Scraper Error] {e}")

    print("[Scraper]  Price not found in search snippets.")
    return "N/A"


# Test block
if __name__ == "__main__":
    # Test with a shoe from your list
    print(f"Final Result: {get_market_price('nike_dunk_low_purple_court')}")