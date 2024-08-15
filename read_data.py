def read_data(source, target):
    with open(target, 'r', encoding='utf-8') as tgt_file, \
         open(source, 'r', encoding='utf-8') as src_file:
        target_sentences = tgt_file.readlines()
        source_sentences = src_file.readlines()
        sentence_pairs = list(zip(target_sentences, source_sentences))
        return sentence_pairs
