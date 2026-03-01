import pandas as pd

class booksData:
    def __init__(self, file):
        self.data = pd.read_csv(file)
        """Separate from data set"""
        self.X = None
        self.Y = None
        self.features = None


def main():
    trending_books = booksData('kindle_data-v2.csv') 

    print("---Original Data (No processing)---")
    print(trending_books.data.head())

if __name__ == "__main__":
    main()


