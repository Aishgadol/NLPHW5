def count_words(text):
    """
    Counts the number of words in the given text.
    :param text: The input text (string).
    :return: The word count (integer).
    """
    # Split the text into words based on whitespace
    words = text.split()
    # Return the length of the resulting list of words
    return len(words)

# Example usage
if __name__ == "__main__":
    input_text = input("Enter your text: ")
    word_count = count_words(input_text)
    print(f"The number of words in the text is: {word_count}")
