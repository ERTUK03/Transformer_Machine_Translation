class DataPreProcessor():
    def __init__(self, num_words, start_token, stop_token):
        self.num_words = num_words
        self.start_token = start_token
        self.stop_token = stop_token

    def preprocess(self, pair):
        result = (self.crop_to_words(pair[0]), self.crop_to_words(pair[1]))
        result = (pair[0].strip(), pair[1].strip())
        result = (result[0].replace("'", ''), result[1].replace("'", ''))
        result = (result[0].replace('"', ''), result[1].replace('"', ''))
        result = (f"{self.start_token} {result[0]} {self.stop_token}", result[1])
        return result

    def crop_to_words(self, text):
        words = text.split()
        cropped_words = words[:self.num_words]
        cropped_text = ' '.join(cropped_words)
        return cropped_text

def preprocess_data(data, num_words, start_token, stop_token):
    data_pre_processor = DataPreProcessor(num_words, start_token, stop_token)
    sentence_pairs = list(map(data_pre_processor.preprocess, data))
    return sentence_pairs
