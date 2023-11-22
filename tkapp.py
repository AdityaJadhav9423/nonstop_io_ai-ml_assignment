import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter.scrolledtext import ScrolledText
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

class NewsClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title('News Headline Classification')

        # Load the scraped data
        self.data_df = pd.read_excel("dataset_news.xlsx")

        # Vectorize the text data
        self.vectorizer = CountVectorizer()
        self.X_vec = self.vectorizer.fit_transform(self.data_df['headline_name'])

        # Train the model
        self.model = MultinomialNB()
        self.model.fit(self.X_vec, self.data_df['headline_category'])

        # GUI components
        self.user_input_label = ttk.Label(root, text='Enter a headline for classification:')
        self.user_input_entry = ttk.Entry(root, width=50)
        self.classify_button = ttk.Button(root, text='Classify', command=self.classify_headline)
        self.result_label = ttk.Label(root, text='Headline Category:')

        # Text widget to display the 'headline_name' column
        self.data_text = ScrolledText(root, width=70, height=10, wrap=tk.WORD)
        self.data_text.insert(tk.END, "\n".join(self.data_df['headline_name']))

        # Grid layout
        self.user_input_label.grid(row=0, column=0, padx=10, pady=10, sticky='w')
        self.user_input_entry.grid(row=0, column=1, padx=10, pady=10)
        self.classify_button.grid(row=1, column=0, columnspan=2, pady=10)
        self.result_label.grid(row=2, column=0, columnspan=2, pady=10)
        self.data_text.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

    def classify_headline(self):
        user_input = self.user_input_entry.get()

        if not user_input:
            messagebox.showwarning('Warning', 'Please enter a headline for classification.')
            return

        # Vectorize the user input
        user_input_vec = self.vectorizer.transform([user_input])

        # Make prediction
        prediction = self.model.predict(user_input_vec)

        # Display the result
        self.result_label.config(text=f'Headlines: {prediction[0]}')


if __name__ == "__main__":
    root = tk.Tk()
    app = NewsClassifierApp(root)
    root.mainloop()
