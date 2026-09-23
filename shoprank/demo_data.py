"""A deterministic mini catalog so the whole project works before ESCI is downloaded."""

PRODUCTS = [
    {"product_id": "P01", "title": "Apple 20W USB-C Power Adapter", "description": "Fast charger for iPhone 15 and iPad", "brand": "Apple", "color": "white"},
    {"product_id": "P02", "title": "Samsung 25W USB-C Charger", "description": "Super fast charging adapter for Galaxy phones", "brand": "Samsung", "color": "black"},
    {"product_id": "P03", "title": "Anker USB-C Cable 2m", "description": "Braided Type C charging cable, cable only", "brand": "Anker", "color": "black"},
    {"product_id": "P04", "title": "Sony WH-1000XM5 Headphones", "description": "Wireless noise cancelling over-ear headphones", "brand": "Sony", "color": "black"},
    {"product_id": "P05", "title": "JBL Tune 510BT Headphones", "description": "Affordable wireless on-ear headphones", "brand": "JBL", "color": "blue"},
    {"product_id": "P06", "title": "Apple AirPods Pro 2", "description": "Wireless earbuds with active noise cancellation", "brand": "Apple", "color": "white"},
    {"product_id": "P07", "title": "Nike Pegasus 40 Running Shoes", "description": "Men's road running shoes with responsive cushioning", "brand": "Nike", "color": "black"},
    {"product_id": "P08", "title": "Adidas Ultraboost Light", "description": "Comfortable lightweight running shoes for men", "brand": "Adidas", "color": "white"},
    {"product_id": "P09", "title": "Nike Revolution 7 Shoes", "description": "Everyday jogging and gym shoes", "brand": "Nike", "color": "blue"},
    {"product_id": "P10", "title": "Logitech G304 Gaming Mouse", "description": "Wireless lightweight gaming mouse with HERO sensor", "brand": "Logitech", "color": "black"},
    {"product_id": "P11", "title": "Dell MS116 Wired Mouse", "description": "Simple optical USB office mouse", "brand": "Dell", "color": "black"},
    {"product_id": "P12", "title": "Razer DeathAdder V3 Mouse", "description": "Ergonomic wired esports gaming mouse", "brand": "Razer", "color": "black"},
    {"product_id": "P13", "title": "Wildcraft 50L Hiking Backpack", "description": "Water resistant trekking rucksack with rain cover", "brand": "Wildcraft", "color": "green"},
    {"product_id": "P14", "title": "American Tourister 32L Backpack", "description": "Laptop and travel day pack", "brand": "American Tourister", "color": "black"},
    {"product_id": "P15", "title": "Samsung Galaxy Buds FE", "description": "Noise cancelling true wireless earbuds", "brand": "Samsung", "color": "black"},
    {"product_id": "P16", "title": "Philips Large Button Feature Phone", "description": "Easy phone for seniors with loud speaker", "brand": "Philips", "color": "black"},
    {"product_id": "P17", "title": "Apple iPhone 15 Clear Case", "description": "Protective phone cover compatible with iPhone 15", "brand": "Apple", "color": "clear"},
    {"product_id": "P18", "title": "Puma Black Formal Shoes", "description": "Men's synthetic leather office shoes size 9", "brand": "Puma", "color": "black"},
]

# Each tuple is (query, exact product, substitutes, complements). Everything else is irrelevant.
QUERY_TEMPLATES = [
    ("iphone 15 type c charger", "P01", ["P02"], ["P03", "P17"]),
    ("apple fast charger", "P01", ["P02"], ["P03"]),
    ("samsung galaxy charger", "P02", ["P01"], ["P03"]),
    ("usb c charging cable", "P03", [], ["P01", "P02"]),
    ("WH-1000XM5", "P04", ["P05"], []),
    ("sony noise cancelling headphones", "P04", ["P05"], ["P06"]),
    ("cheap wireless headphones", "P05", ["P04"], ["P06"]),
    ("apple noise cancelling earbuds", "P06", ["P15"], ["P04"]),
    ("nike running shoes", "P07", ["P09", "P08"], []),
    ("comfortable shoes for jogging", "P08", ["P07", "P09"], []),
    ("nike gym shoes blue", "P09", ["P07"], []),
    ("wireless gaming mouse", "P10", ["P12"], ["P11"]),
    ("G304", "P10", ["P12"], []),
    ("office usb mouse", "P11", ["P10"], ["P12"]),
    ("razer esports mouse", "P12", ["P10"], []),
    ("waterproof hiking backpack 50l", "P13", ["P14"], []),
    ("black laptop backpack", "P14", ["P13"], []),
    ("samsng earbud", "P15", ["P06"], ["P02"]),
    ("phone for elderly people", "P16", [], []),
    ("iphone 15 cover", "P17", [], ["P01"]),
    ("black formal shoes size 9", "P18", [], []),
    ("adidas running shoes", "P08", ["P07", "P09"], []),
    ("headphones for travel", "P04", ["P05"], ["P06"]),
    ("mouse for laptop", "P11", ["P10", "P12"], []),
    ("trekking rucksack rain cover", "P13", ["P14"], []),
    ("galaxy wireless buds", "P15", ["P06"], ["P02"]),
    ("airpods", "P06", ["P15"], []),
    ("lightweight running shoe", "P08", ["P07", "P09"], []),
    ("iphone adapter", "P01", ["P02"], ["P03", "P17"]),
    ("senior citizen mobile loud speaker", "P16", [], []),
]


def judgments() -> list[dict]:
    """Return 30 queries x 18 candidates with graded labels."""
    rows = []
    for number, (query, exact, substitutes, complements) in enumerate(QUERY_TEMPLATES, start=1):
        for product in PRODUCTS:
            product_id = product["product_id"]
            label = "E" if product_id == exact else "S" if product_id in substitutes else "C" if product_id in complements else "I"
            rows.append({"query_id": f"Q{number:03d}", "query": query, "product_id": product_id, "label": label})
    return rows

