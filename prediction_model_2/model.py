import os
from typing import Optional, List, Dict
from TexSoup import TexSoup
import numpy as np
import pandas as pd
import pickle
import re
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Embedding,
    Conv1D,
    GlobalMaxPooling1D,
    Dense,
    GlobalAveragePooling1D,
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class Model:
    """ """

    def __init__(self):
        """ """
        self._test_results: pd.DataFrame = pd.DataFrame()
        self._model = None
        self._max_len = None
        self._tokenizer = Tokenizer()

    def train(
        self,
        output_dim: Optional[int] = 256,
        filters: Optional[int] = 512,
        global_max_pooling: Optional[bool] = True,
        quiet: Optional[bool] = False,
    ) -> None:
        """

        :param output_dim:
        :param filters:
        :param quiet:
        :return:
        """

        self._get_all_original_texts()
        self._get_random_texts()
        texts = self._all_original_texts  + self._all_random_texts
        labels = [1 for _ in range(len(self._all_original_texts))] + [0 for _ in range(len(self._all_random_texts))]
        self._tokenizer.fit_on_texts(texts)
        sequences = self._tokenizer.texts_to_sequences(texts)
        vocab_size = len(self._tokenizer.word_index) + 1
        self._max_len = max(len(seq) for seq in sequences)
        X = pad_sequences(sequences, maxlen=self._max_len)
        y = np.array(labels)
        if global_max_pooling:
            e = GlobalMaxPooling1D()
        else:
            e = GlobalAveragePooling1D()
        self._model = Sequential(
            [
                Embedding(input_dim=vocab_size, output_dim=output_dim),
                Conv1D(filters=filters, kernel_size=2, activation="relu"),
                e,
                Dense(64, activation="relu"),
                Dense(1, activation="sigmoid"),
            ]
        )
        self._model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )
        self._model.fit(X, y, epochs=10, batch_size=2, verbose=not quiet)

    def predict(self, text: str) -> float:
        """
        :param text:
        :return: float value between 0 and 1 where 0 means non-plagiat and 1 plagiat.
        """
        new_sequences = self._tokenizer.texts_to_sequences([text])
        new_X = pad_sequences(new_sequences, maxlen=self._max_len)
        predictions = self._model.predict(new_X)
        return predictions[0][0]

    def find_best(
        self,
        test_dict: Dict[str, bool],
        output_dim_list: Optional[List[int]] = [256],
        filters_list: Optional[List[int]] = [512],
        pooling_avg: Optional[bool] = False
    ) -> pd.DataFrame:
        """

        :param test_dict:
        :param output_dim_list:
        :param filters_list:
        :return:
        """
        self._test_results = pd.DataFrame(
            columns=[
                "output_dim",
                "filters"
                "pooling_avg",
                "expected_value",
                "real_value",
                "difference",
            ]
        )
        for output_dim in output_dim_list:
            for filters in filters_list:
                self.train(output_dim, filters,pooling_avg, quiet=True)
                for text, value in test_dict.items():
                    answer = self.predict(text)
                    new_row = {
                        "output_dim": output_dim,
                        "filters": filters,
                        "expected_value": value,
                        "pooling_avg" :pooling_avg,
                        "real_value": answer,
                        "difference": abs(value - answer),
                    }
                    self._test_results = pd.concat(
                        [self._test_results, pd.DataFrame([new_row])],
                        ignore_index=True
                    )
        return self._test_results

    def save_predictions(self,file_name:str):
        self._test_results.to_csv(file_name)

    def save_model(self):
        self._model.save('aa.keras')
        with open("tokenizer.pkl", "wb") as f:
            pickle.dump(self._tokenizer, f)

    def load_model(self):
        self._model = load_model('aa.keras')
        with open("tokenizer.pkl", "rb") as f:
            self._tokenizer = pickle.load(f)
        self._model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    def _get_all_original_texts(self):


        self._all_original_texts = []
        for file_name in os.listdir("train_original"):
            with open("train_original/"+file_name, 'r', encoding='utf-8') as f:
                raw = f.read()
            print(raw)

            smieci1 = re.findall(r'\$\$\\begin{array}.*?\\end{array}\$\$', raw, re.DOTALL)
            smieci2 = re.findall(r'\\begin{tabular}.*?\\end{tabular}', raw, re.DOTALL)
            smieci3 = re.findall(r'\$\$\s*\\begin{array}.*?\\end{array}\s*\$\$', raw, re.DOTALL)
            my1_math = re.findall(r'\\begin{align\*}.*?\\end{align\*}', raw, re.DOTALL)
            my2_math = re.findall(r'\$\$.*?\$\$', raw)

            # Łączenie wyników
            smieci = smieci1 + smieci2 + smieci3 + my1_math + my2_math

            pattern = '|'.join([re.escape(sentence) for sentence in smieci])

            # Usuwamy te zdania z tekstu
            text_cleaned = re.sub(pattern, '', raw)

            soup = TexSoup(text_cleaned)
            all_in_one = []
            for text in soup.text:
                all_in_one.append(text)
            all_in_one = ' '.join(all_in_one)
            all_in_one = re.sub('\t', '', all_in_one)
            all_in_one = all_in_one.split('\n')
            for i in all_in_one:
                i = i.strip()
                if i != '':
                    self._all_original_texts.append(i)

    def _get_random_texts(self):
        self._all_random_texts = []
        for file_name in os.listdir("train_random"):
            with open("train_random/"+file_name, 'r', encoding='utf-8') as f:
                raw = f.read()






