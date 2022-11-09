from transformers import BertTokenizer, AutoTokenizer
import torch
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.classes.preprocessor import TextPreProcessor
from model import SpanEmo
import numpy as np

def twitter_preprocessor():
    preprocessor = TextPreProcessor(
        normalize=['url', 'email', 'phone', 'user'],
        annotate={"hashtag", "elongated", "allcaps", "repeated", 'emphasis', 'censored'},
        all_caps_tag="wrap",
        fix_text=False,
        segmenter="twitter_2018",
        corrector="twitter_2018",
        unpack_hashtags=True,
        unpack_contractions=True,
        spell_correct_elong=False,
        tokenizer=SocialTokenizer(lowercase=True).tokenize).pre_process_doc
    return preprocessor

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
segment_a = "anger anticipation disgust fear joy love optimism hopeless sadness surprise or trust?"
label_names = ["anger", "anticipation", "disgust", "fear", "joy",
                           "love", "optimism", "hopeless", "sadness", "surprise", "trust"]
preprocessor = twitter_preprocessor()

def process_text(text, lang='English', max_length=512):
    global bert_tokenizer, segment_a, preprocessor

    inputs, lengths, label_indices = [], [], []
    x = ' '.join(preprocessor(text))
    x = bert_tokenizer.encode_plus(segment_a,
                                        x,
                                        add_special_tokens=True,
                                        max_length=max_length,
                                        truncation=True)
    input_id = x['input_ids']
    input_length = len([i for i in x['attention_mask'] if i == 1])
    inputs.append(input_id)
    lengths.append(input_length)

    #label indices
    label_idxs = [bert_tokenizer.convert_ids_to_tokens(input_id).index(label_names[idx])
                        for idx, _ in enumerate(label_names)]
    label_indices.append(label_idxs)

    inputs = torch.tensor(inputs, dtype=torch.long)
    data_length = torch.tensor(lengths, dtype=torch.long)
    label_indices = torch.tensor(label_indices, dtype=torch.long)
    return inputs, torch.LongTensor([0] * 11).unsqueeze(0), data_length, label_indices

def process(model, text, return_binary=False):
    tokenized = process_text(text)
    out = model(tokenized, 'cuda:0')
    labelmap = list(out[2][0].astype(bool))
    if return_binary:
        return labelmap
    out_labels = np.array(label_names)[labelmap]
    return out_labels

def load_model(checkpoint, lang='English', device="cuda:0"):
    model = SpanEmo(lang=lang)
    model.to(device).load_state_dict(torch.load(checkpoint), strict=False)
    return model