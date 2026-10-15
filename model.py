import torch, os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv() # load the .env file
device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"[+] Using device: {device}")

def load_model(): # download or load the model checkpints
    model_card = os.getenv("MODEL_CARD")
    model_path = os.getenv("MODEL_PATH")
    print(f"[+] model card: {model_card}")
    print(f"[+] loading model checkpints...", end="")
    model = None
    if os.path.isfile(os.path.join(model_path, "modules.json")):
        print(f"[-] Model already exists: {model_path}. Reading checkpoints...")
        model = SentenceTransformer(str(model_path), device=device, local_files_only=True)
    else:
        os.makedirs(model_path, exist_ok=True)
        model = SentenceTransformer(model_card, device=device)

    # GPT-2 has no padding token; reuse EOS without adding vocabulary entries.
    if model.tokenizer.pad_token_id is None:
        if model.tokenizer.eos_token_id is None:
            raise ValueError("The model tokenizer has neither a padding token nor an EOS token.")
        model.tokenizer.pad_token = model.tokenizer.eos_token
    model[0].auto_model.config.pad_token_id = model.tokenizer.pad_token_id

    if not os.path.isfile(os.path.join(model_path, "modules.json")):
        model.save_pretrained(str(model_path))
        print(f"[+] Model saved to: {model_path}")
    return model

def get_embeddings(model, word): # return the embeddings and wordpieces
    _tokens = model.tokenizer.tokenize(word)
    token_ids = model.tokenizer.convert_tokens_to_ids(_tokens)
    tokens = model.tokenizer.convert_ids_to_tokens(token_ids)
    embeddings = model.encode(word)
    return {
        "word" : word,
        "word_piece" : tokens,
        "embedding" : embeddings
    }

if __name__ == "__main__": # driving code
    model = load_model()
    sentence = "I love programming in Python."
    # sentence = "ধারা ১ সমস্ত মানুষ স্বাধীনভাবে সমান মর্যাদা এবং অধিকার নিয়ে জন্মগ্রহণ করে" # bengali
    emb = get_embeddings(model, sentence)
    print(emb["word"])
    print(len(emb["word_piece"]))
    print(len(emb["embedding"]))
    print(emb["word_piece"])
