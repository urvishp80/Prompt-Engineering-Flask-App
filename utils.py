import numpy as np
import openai
import pandas as pd
import json
import os
from gpt_index import SimpleDirectoryReader, GPTListIndex, readers, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.environ.get("OPENAI_API_KEY")

# GPT-3 Default Model
COMPLETIONS_MODEL = "ada:ft-hyly-ai:greystar-2023-02-14-16-49-00"

# OpenAI Embedding API Model
openai_embedding_model = "text-embedding-ada-002"
df_content = pd.read_csv("data/final_with_content_finetune_data.csv")
openai_embedding_greystar = json.load(open("data/doc_embeddings_greystar.json"))


def get_openai_embedding(text: str, model=openai_embedding_model):
    result = openai.Embedding.create(
        model=model,
        input=text
    )
    return result["data"][0]["embedding"]


def vector_similarity(x, y):
    return np.dot(np.array(x), np.array(y))


def order_document_sections_by_query_similarity(query, contexts):
    query_embedding = get_openai_embedding(query)
    # vectory similarity between user input and embedding_dict
    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)
    return document_similarities


def get_answer_from_openai_embedding_dict(user_input, context_embedding, dataframe):
    try:
        most_relevant_document_sections = order_document_sections_by_query_similarity(user_input, context_embedding)[:2]
        for i in most_relevant_document_sections:
            document_section = dataframe.iloc[int(i[1])]
            if isinstance(document_section, pd.DataFrame):
                document_section = document_section.iloc[0]
            return str(document_section['completion'])
    except:
        return str("Please call us for more information.")


# GPT-Index Model
gpt_index_json = "data/greystar_gpt_index.json"
gpt_index_vector_index = GPTSimpleVectorIndex.load_from_disk(gpt_index_json)


# FAISS Model
model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]


def get_faiss_embeddings(text_list):
    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="pt"
    )
    encoded_input = {k: v for k, v in encoded_input.items()} # v.to(device)
    model_output = model(**encoded_input)
    return cls_pooling(model_output)


# load vector file
faiss_vector_path = "data/greystar_faiss_vectors.npy"
faiss_vectors = np.load(faiss_vector_path)  # load
faiss_index_flat = faiss.IndexFlatL2(faiss_vectors.shape[1])  # build a flat (CPU) index


def get_answer_from_faiss(user_input, dataframe):
    user_embedding = get_faiss_embeddings([user_input]).cpu().detach().numpy()
    k = 2  # we want to see 2 nearest neighbors
    scores, index_num = faiss_index_flat.search(user_embedding, k)
    samples_df = pd.DataFrame({"index_num": index_num.flatten(), "scores": scores.flatten()})
    samples_df.sort_values("scores", ascending=False, inplace=True)
    idx = samples_df.iloc[0].index_num
    result = dataframe.iloc[int(idx)].completion
    # for _, r in samples_df.iterrows():
    #     idx = int(r['samples'])
    #     row = df.iloc[idx]
    #     response = row.completion
    #     if idx == 1:  # we will show only first row only
    #         break
    return str(result).strip()


def get_answer_with_combined_approach(user_input):
    chosen_sections = []

    # embedding api
    embedding_api_response = get_answer_from_openai_embedding_dict(
        user_input=user_input,
        context_embedding=openai_embedding_greystar,
        dataframe=df_content
    )
    embedding_api_response = str(embedding_api_response).strip()
    chosen_sections.append(embedding_api_response)

    # gpt-index
    gpt_index_response = gpt_index_vector_index.query(
        user_input,
        response_mode="compact"
    )
    gpt_index_response = str(gpt_index_response.response).strip()
    chosen_sections.append(gpt_index_response)

    # faiss
    faiss_response = get_answer_from_faiss(
        user_input=user_input,
        dataframe=df_content
    )
    faiss_response = str(faiss_response).strip()
    chosen_sections.append(faiss_response)

    # openai completion
    header = """Answer the question using the provided context and be as truthful as possible, and if you are unsure, say "Please rephrase the question."\n\nContext:\n"""
    final_prompt = header + "".join(list(set(chosen_sections))) + "\n\n Q: " + user_input + "?" + "\n A:"

    final_response = openai.Completion.create(
        model=COMPLETIONS_MODEL,
        prompt=final_prompt,
        max_tokens=1000,
        temperature=0,
        stop=[" END"],
    )

    final_response = final_response["choices"][0]["text"].replace("\n", "").strip()
    return final_response






