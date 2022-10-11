from typing import List


class CountVectorizer:
    """Converts a collection of text documents to a matrix of word counts"""
    def __init__(self):
        self.text_dict = {}

    def fit_transform(self, text: List[str]) -> List[int]:
        """ transforms given text to word count matrix"""
        transformed_all = []
        index = 0
        punctuation = '!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        for sentence in text:
            transformed_sentence = [0]*len(self.text_dict.keys())
            split_text = sentence.split()
            for word in split_text:
                word = word.lower()
                word = ''.join(ch for ch in word if ch not in punctuation)
                if word not in self.text_dict:
                    self.text_dict[word] = index
                    transformed_sentence.append(1)
                    index += 1
                else:
                    transformed_sentence[self.text_dict[word]] += 1
            transformed_all.append(transformed_sentence)
        for i in range(len(transformed_all)):
            length = len(transformed_all[i])
            if length < index - 1:
                transformed_all[i] = transformed_all[i] + [0]*(index - length)
        return transformed_all

    def get_feature_names(self) -> List[List[int]]:
        """Returns feature names in word matrix"""
        if len(self.text_dict.keys()) == 0:
            print("No feature names, please, first transform your text!")
            return None
        else:
            return list(self.text_dict.keys())


if __name__ == '__main__':
    corpus = ['Crock Pot Pasta Never boil pasta again',
              'Pasta Pomodoro Fresh ingredients Parmesan to taste']
    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform(corpus)
    print(vectorizer.get_feature_names())
    print(count_matrix)
