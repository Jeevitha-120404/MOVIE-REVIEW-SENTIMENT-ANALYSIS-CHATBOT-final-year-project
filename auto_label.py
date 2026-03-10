import pandas as pd

# Load your original dataset
df = pd.read_csv("movie_reviews.csv")  # <-- your current file

positive_words = [
    "amazing","best","excellent","masterpiece","brilliant","beautiful",
    "perfect","outstanding","powerful","unforgettable","thrilling",
    "iconic","stunning","innovative","satisfying","epic"
]

negative_words = [
    "boring","slow","confusing","overhyped","dragged","depressing",
    "disturbing","tiring","predictable","too long","uncomfortable",
    "forced","unclear"
]

def auto_label(review):
    r = review.lower()
    if any(w in r for w in positive_words):
        return "Positive"
    elif any(w in r for w in negative_words):
        return "Negative"
    else:
        return "Neutral"

df["Sentiment"] = df["Review"].apply(auto_label)

# Save final labeled dataset
df.to_csv("movie_reviews_labeled.csv", index=False)

print("✅ movie_reviews_labeled.csv created successfully")
import pandas as pd

# Load your original dataset
df = pd.read_csv("movie_reviews.csv")  # <-- your current file

positive_words = [
    "amazing","best","excellent","masterpiece","brilliant","beautiful",
    "perfect","outstanding","powerful","unforgettable","thrilling",
    "iconic","stunning","innovative","satisfying","epic"
]

negative_words = [
    "boring","slow","confusing","overhyped","dragged","depressing",
    "disturbing","tiring","predictable","too long","uncomfortable",
    "forced","unclear"
]

def auto_label(review):
    r = review.lower()
    if any(w in r for w in positive_words):
        return "Positive"
    elif any(w in r for w in negative_words):
        return "Negative"
    else:
        return "Neutral"

df["Sentiment"] = df["Review"].apply(auto_label)

# Save final labeled dataset
df.to_csv("movie_reviews_labeled.csv", index=False)

print("✅ movie_reviews_labeled.csv created successfully")
import pandas as pd

# Load your original dataset
df = pd.read_csv("movie_reviews.csv")  # <-- your current file

positive_words = [
    "amazing","best","excellent","masterpiece","brilliant","beautiful",
    "perfect","outstanding","powerful","unforgettable","thrilling",
    "iconic","stunning","innovative","satisfying","epic"
]

negative_words = [
    "boring","slow","confusing","overhyped","dragged","depressing",
    "disturbing","tiring","predictable","too long","uncomfortable",
    "forced","unclear"
]

def auto_label(review):
    r = review.lower()
    if any(w in r for w in positive_words):
        return "Positive"
    elif any(w in r for w in negative_words):
        return "Negative"
    else:
        return "Neutral"

df["Sentiment"] = df["Review"].apply(auto_label)

# Save final labeled dataset
df.to_csv("movie_reviews_labeled.csv", index=False)

print("✅ movie_reviews_labeled.csv created successfully")

