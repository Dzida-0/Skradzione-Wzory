from typing import Optional, List, Dict
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
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
        quiet: Optional[bool] = False,
    ) -> None:
        """

        :param output_dim:
        :param filters:
        :param quiet:
        :return:
        """
        with open("learning_text", "r", encoding="iso-8859-2") as f:
            l = f.read()
        t = l.split(".")
        with open("los_text", "r", encoding="iso-8859-2") as a:
            l = a.read()
        z = l.split(".")
        texts = t + z
        labels = [1 for _ in range(len(t))] + [0 for _ in range(len(z))]
        self._tokenizer.fit_on_texts(texts)
        sequences = self._tokenizer.texts_to_sequences(texts)
        vocab_size = len(self._tokenizer.word_index) + 1
        self._max_len = max(len(seq) for seq in sequences)
        X = pad_sequences(sequences, maxlen=self._max_len)
        y = np.array(labels)
        self._model = Sequential(
            [
                Embedding(input_dim=vocab_size, output_dim=output_dim),
                Conv1D(filters=filters, kernel_size=2, activation="relu"),
                GlobalMaxPooling1D(),
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
                "filters",
                "expected_value",
                "real_value",
                "difference",
            ]
        )
        for output_dim in output_dim_list:
            for filters in filters_list:
                self.train(output_dim, filters, quiet=True)
                for text, value in test_dict.items():
                    answer = self.predict(text)
                    new_row = {
                        "output_dim": output_dim,
                        "filters": filters,
                        "expected_value": value,
                        "real_value": answer,
                        "difference": abs(value - answer),
                    }
                    self._test_results = pd.concat(
                        [self._test_results, pd.DataFrame([new_row])],
                        ignore_index=True
                    )
        return self._test_results
                # tab = []
                # print(f"output_dim: {output_dim}, filters: {filters}")
                # for text, value in test_dict.items():
                #     answer = self.predict([text])
                #     print(value, answer, abs(value - answer))
                #     tab.append(abs(value - answer))
                # print(np.mean(tab))
                # print("-----------------------------------")
