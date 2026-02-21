# ============================================================
# 1. Imports
# ============================================================
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import pandas as pd
from tqdm import tqdm

# ============================================================
# 2. Device
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================================================
# 3. Load Dataset
# ============================================================
df = pd.read_excel("Nyaya_dataset.xlsx")  # change filename

df.columns = df.columns.str.strip().str.lower()

required_cols = ["question", "ground truth answer"]
for col in required_cols:
    assert col in df.columns, f"Column '{col}' not found!"

questions = df["question"].tolist()
references = df["ground truth answer"].tolist()

print("Total samples:", len(questions))

# ============================================================
# 4. Load InLegalLLaMA (Answer Generation)
# ============================================================
gen_model_name = "sudipto-ducs/InLegalLLaMA-Instruct"

gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)

gen_model = AutoModelForCausalLM.from_pretrained(
    gen_model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

gen_model.eval()

def generate_answer(question):
    prompt = f"Answer the following legal question:\n\n{question}\n\nAnswer:"

    inputs = gen_tokenizer(prompt, return_tensors="pt").to(gen_model.device)

    with torch.no_grad():
        outputs = gen_model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )

    decoded = gen_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove prompt from output (optional cleanup)
    answer = decoded.split("Answer:")[-1].strip()

    return answer

# ============================================================
# 5. Load InLegalBERT (Embeddings)
# ============================================================
bert_model_name = "law-ai/InLegalBERT"

bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
bert_model = AutoModel.from_pretrained(bert_model_name).to(device)

bert_model.eval()

def get_token_embeddings(sentence):
    encoded = bert_tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    input_ids = encoded["input_ids"].to(device)
    mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        outputs = bert_model(input_ids, attention_mask=mask)

    return outputs.last_hidden_state[0]  # keep all tokens


def cosine_similarity_matrix(cand_emb, ref_emb):
    cand_norm = F.normalize(cand_emb, p=2, dim=1)
    ref_norm = F.normalize(ref_emb, p=2, dim=1)
    return torch.mm(cand_norm, ref_norm.transpose(0, 1))


def legal_bert_score(candidate, reference):
    cand_emb = get_token_embeddings(candidate)
    ref_emb = get_token_embeddings(reference)

    sim_matrix = cosine_similarity_matrix(cand_emb, ref_emb)

    precision = sim_matrix.max(dim=1)[0].mean().item()
    recall = sim_matrix.max(dim=0)[0].mean().item()

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


# ============================================================
# 6. Generate Answers + Compute BERTScore
# ============================================================
generated_answers = []
bert_precisions = []
bert_recalls = []
bert_f1s = []

for question, reference in tqdm(zip(questions, references), total=len(questions)):

    # Generate
    gen_answer = generate_answer(question)
    generated_answers.append(gen_answer)

    # Evaluate
    p, r, f1 = legal_bert_score(gen_answer, reference)

    bert_precisions.append(p)
    bert_recalls.append(r)
    bert_f1s.append(f1)

# ============================================================
# 7. Save Results
# ============================================================
df["generated answer"] = generated_answers
df["bert_precision"] = bert_precisions
df["bert_recall"] = bert_recalls
df["bert_f1"] = bert_f1s

output_file = "InLegalLLaMA_BERTScore_Results.xlsx"
df.to_excel(output_file, index=False)

print("Saved results to:", output_file)
