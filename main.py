import sys

import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import wikipedia
import tkinter as tk
from tkinter import simpledialog, scrolledtext


nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('omw-1.4') 


def lemma_me(sent: str) -> list:
    """Tokenizes and lemmatizes the input sentence based on part-of-speech tags.
    
    Args
    -----
    * sent (str): Input sentence to be processed.
    
    Returns
    -----
    * list: A list of lemmatized tokens.

    """
    lemmatizer = WordNetLemmatizer()
    sentence_tokens = nltk.word_tokenize(sent.lower())
    pos_tags = nltk.pos_tag(sentence_tokens)

    sentence_lemmas = []
    for token, pos_tag in zip(sentence_tokens, pos_tags):
        if pos_tag[1][0].lower() in ["n", "v", "a", "r"]:
            lemma = lemmatizer.lemmatize(token, pos_tag[1][0].lower())
            sentence_lemmas.append(lemma)
    return sentence_lemmas


def process(text: str, question: str) -> str | None:
    """
    Finds the sentence in the given text most similar to the question using TF-IDF and cosine similarity.

    Args
    ----
    * text (str): The text to search within.
    * question (str): The question to compare against sentences in the text.

    Returns:
    * str or None: The sentence most similar to the question if similarity > 0.3, otherwise None.
    """
    sentence_tokens = nltk.sent_tokenize(text)
    sentence_tokens.append(question)

    # Instantiate a tokrnizer
    tv = TfidfVectorizer(tokenizer=lemma_me)
    # Vectorize the sentence_tokens (computes importance of each word)
    tf = tv.fit_transform(sentence_tokens)
    # Calculates similarities between question (the last) and each sentence.
    values = cosine_similarity(tf[-1], tf)
    # Sort by similarities in ascending order
    # Get the similarities list [0] and the second largest one (avoiding the question itself.)
    index = values.argsort()[0][-2]
    values_flat = values.flatten()
    values_flat.sort()
    coeff = values_flat[-2]
    if coeff > 0.3:
        return sentence_tokens[index]


def start_chat():
    """
    Launches a simple GUI chat application that lets the user input a topic,
    fetches related content from Wikipedia, and allows asking questions
    about the topic. Answers are retrieved by finding the most relevant sentence
    in the Wikipedia content.

    The GUI displays the conversation history and provides an input field
    for submitting questions.
    """
    root = tk.Tk()
    root.withdraw()
    topic = simpledialog.askstring("Enter topic", "Please tell me about the topic of your interest.")
    if not topic:
        return
    
    text = wikipedia.page(topic).content 
    
    chat_root = tk.Tk()
    chat_root.title(f"Q&A Chat - Topic: {topic}")
    chat_history = scrolledtext.ScrolledText(chat_root, wrap=tk.WORD, width=60, height=20, state=tk.DISABLED)
    chat_history.pack(padx=10, pady=10)

    entry_frame = tk.Frame(chat_root)
    entry_frame.pack(padx=10, pady=5, fill="x")
    entry = tk.Entry(entry_frame, width=50)
    entry.pack(side=tk.LEFT, fill="x", expand=True)

    def on_submit():
        question = entry.get().strip()
        entry.delete(0, tk.END) 
        if question is None:
            root.destroy()
            return
        answer = process(text, question) or "I have no knowledge."
        chat_history.config(state=tk.NORMAL)
        chat_history.insert(tk.END, f"Q: {question}\n", "question")
        chat_history.insert(tk.END, f"A: {answer}\n\n", "answer")
        chat_history.config(state=tk.DISABLED)
        chat_history.see(tk.END) 

    entry.bind("<Return>", lambda e: on_submit())
    send_btn = tk.Button(entry_frame, text="Submit question", command=on_submit)
    send_btn.pack(side=tk.LEFT, padx=5)

    chat_history.tag_config("question", foreground="blue")
    chat_history.tag_config("answer", foreground="green")

    chat_root.protocol("WM_DELETE_WINDOW", sys.exit)
    chat_root.mainloop()

if __name__ == "__main__":
    start_chat()
