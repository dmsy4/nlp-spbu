import torch
from transformers import AutoModel, AutoTokenizer

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def mean_pooling(model_output, attention_mask):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]  
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def init_model():
    tokenizer = AutoTokenizer.from_pretrained("ai-forever/sbert_large_nlu_ru")
    model = AutoModel.from_pretrained("ai-forever/sbert_large_nlu_ru").to(DEVICE)

    return model, tokenizer


def encode_sentence(sentence, model, tokenizer):
    encoded_input = tokenizer(
        sentence, padding=True, truncation=True, max_length=128, return_tensors="pt"
    ).to(DEVICE)
    with torch.no_grad():
        model_output = model(**encoded_input)

    sentence_embedding = (
        mean_pooling(model_output, encoded_input["attention_mask"])[0].cpu().numpy()
    )

    return sentence_embedding
