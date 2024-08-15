from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

def yield_tokens(data_iter):
    for (de, en) in data_iter:
        yield de
        yield en

def encode_batch(batch, tokenizer):
    return [tokenizer.encode(sentence, add_special_tokens=True).ids for sentence in batch]

def tokenize_data(data, special_tokens, vocab_size, t_name):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(special_tokens=special_tokens, vocab_size=vocab_size)
    tokenizer.train_from_iterator(yield_tokens(data), trainer=trainer)
    tokenizer.save(f'{t_name}.json')
    tgt_encoded = encode_batch([tgt for tgt, _ in data], tokenizer)
    src_encoded = encode_batch([src for _, src in data], tokenizer)
    return src_encoded, tgt_encoded
