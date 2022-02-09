from sklearn.pipeline import Pipeline

from classification_model.config.core import config
from classification_model.preprocessing import preprocessing as pp

text_process_pipe = Pipeline(
    [
        ("text_cleanup", pp.textcleanup()),
        ("text_tokenenize", pp.texttokenize()),
        ("text_stopwordremove", pp.textstopwordremove()),
        ("text_lemmatize", pp.textlemmatize()),
        ("text_stemmer", pp.textstemmer()),
        ("text_token", pp.texttokenize2(config.model_config.MAX_LENGHT)),
    ]
)
