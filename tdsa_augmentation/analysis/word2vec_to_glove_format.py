import argparse
from pathlib import Path

from gensim.models import Word2Vec

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

if __name__=='__main__':
    model_save_fp_help = "File Path to save the model. NOTE if N>1 then the "\
                         "first model will be saved. This needs to be a directory"\
                         " that does not exist currently."
    parser = argparse.ArgumentParser()
    parser.add_argument("word2vec_embedding_fp", type=parse_path, 
                        help='File Path to the word2vec embedding')
    parser.add_argument("save_fp", type=parse_path, 
                        help='File Path to the embedding in the Glove format')
    args = parser.parse_args()

    model = Word2Vec.load(str(args.word2vec_embedding_fp))
    with args.save_fp.open('w+') as save_file:
        for index, word in enumerate(model.wv.vocab):
            vector = model.wv[word].tolist()
            word_vector = [word] + vector
            str_word_vector = [str(value) for value in word_vector]
            str_word_vector = ' '.join(str_word_vector).strip()
            if index == 0:
                save_file.write(f'{str_word_vector}')
            else:
                save_file.write(f'\n{str_word_vector}')