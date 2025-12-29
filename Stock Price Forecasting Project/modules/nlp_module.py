import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------- Ensure NLTK data ----------
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    return " ".join(tokens)

def generate_news_text(row):
    if row["Close"] > row["Open"]:
        return "Stock rises due to positive market sentiment"
    elif row["Close"] < row["Open"]:
        return "Stock falls amid investor concerns"
    else:
        return "Stock remains stable with no major movements"

def run_nlp(input_data, max_features=10):
    """
    Streamlit-ready NLP preprocessing.

    Returns:
        dict: {
            'df_final': DataFrame with TF-IDF columns,
            'tfidf_columns': list of feature names
        }
    """
    if hasattr(input_data, "read"):
        df = pd.read_csv(input_data)
    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        df = pd.read_csv(input_data)

    df["News_Text"] = df.apply(generate_news_text, axis=1)
    df["Cleaned_Text"] = df["News_Text"].apply(clean_text)

    vectorizer = TfidfVectorizer(max_features=max_features)
    X_tfidf = vectorizer.fit_transform(df["Cleaned_Text"])
    tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=vectorizer.get_feature_names_out())

    df_final = pd.concat([df, tfidf_df], axis=1)

    return {
        'df_final': df_final,
        'tfidf_columns': vectorizer.get_feature_names_out().tolist()
    }



