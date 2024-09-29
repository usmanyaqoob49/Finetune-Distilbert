from transformers import AutoTokenizer

model= 'Distilbert-base-uncased'
distilbert_tokenier= AutoTokenizer.from_pretrained(model)
print('Vocabluary size of this tokenizer: ', distilbert_tokenier.vocab_size)

test= 'Hello I am Usman.'
test_token= distilbert_tokenier(test)
print('Tokenized: ',test_token)

#converting input_ids back to tokens(words)
token_words= distilbert_tokenier.convert_ids_to_tokens(test_token['input_ids'])
print('Token words: ',token_words)

#now converting back to string
test_sentence= distilbert_tokenier.convert_tokens_to_string(token_words)
print('Sentence: ',test_sentence)