
import csv
import random

random.seed(42)

products = ["phone", "laptop", "blender", "headphones", "shoes", "backpack",
            "watch", "camera", "coffee maker", "keyboard", "monitor", "vacuum cleaner"]

genuine_templates = [
    "I bought this {p} three weeks ago and the battery life is noticeably better than my old one, though the setup instructions were a bit confusing at first.",
    "The {p} works well overall. Build quality feels solid, but the packaging arrived slightly dented so I'd suggest better shipping protection.",
    "Decent {p} for the price. It does what it says, delivery took 4 days, and customer support was helpful when I had a question about the warranty.",
    "Mixed feelings about this {p}. The design is nice and lightweight, but I noticed the button feels a little loose after two weeks of daily use.",
    "This {p} replaced one that broke last year. So far it's quieter and the instructions included a helpful troubleshooting section.",
    "Good value {p}. Not perfect -- the color was slightly different from the photos -- but functionally it's been reliable for my daily commute.",
    "After using this {p} for a month, I can say the performance is consistent, though I wish the manual explained the settings better.",
    "The {p} arrived on time and matched the listing description. One of the screws was slightly stripped but it still assembled fine.",
    "I compared this {p} to two competitor brands before buying. It's a bit heavier than expected but the price justified the tradeoff.",
    "Solid {p} for everyday use. Customer service responded within a day when I asked about compatibility with my setup.",
]

fake_templates = [
    "BEST {P} EVER!!! Changed my life completely!!! Buy now you will NOT regret it!!! 5 stars!!!",
    "Amazing amazing amazing product!!! This {p} is perfect in every single way, no flaws at all, 100% recommend to everyone!!!",
    "WOW just WOW! This {p} is a miracle product, best purchase of my entire life, everyone needs to buy this right now!!!",
    "Incredible incredible!! Five stars five stars!! This {p} solved all my problems instantly, absolutely perfect, buy immediately!!!",
    "This is the greatest {p} in the whole world, nothing compares, perfect quality, perfect price, everyone must buy today!!!",
    "OMG best {p} ever made!!! I love love love it so much, zero problems,100% perfect, highly highly recommend!!!",
    "Perfect perfect perfect! This {p} exceeded all expectations in every possible way, no downsides whatsoever, must buy now!!!",
    "Unbelievable quality!!! This {p} is flawless, best on the market, everyone is buying it, don't miss out, buy today!!!",
    "Life changing product!!! This {p} is simply the best, no complaints at all, guaranteed satisfaction, buy right now!!!",
    "Five stars isn't enough!!! This {p} is the greatest invention ever, absolutely perfect, everyone should own one!!!",
]


def build_rows(n_per_template=12):
    rows = []
    reviewer_id = 1
    for t in genuine_templates:
        for p in products[:n_per_template]:
            text = t.format(p=p)
            rows.append((f"user{reviewer_id}", text, round(random.uniform(3.0, 5.0), 1), "Genuine"))
            reviewer_id += 1
    for t in fake_templates:
        for p in products[:n_per_template]:
            text = t.format(p=p, P=p.upper())
            rows.append((f"fakeuser{reviewer_id}", text, 5.0, "Fake"))
            reviewer_id += 1
    random.shuffle(rows)
    return rows


if __name__ == "__main__":
    rows = build_rows()
    with open("sample_reviews.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["reviewer_name", "review_text", "rating", "label"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to sample_reviews.csv")
