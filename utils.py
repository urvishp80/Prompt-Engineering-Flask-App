import shutil
import re
import numpy as np
import openai
import pandas as pd
import json
import os
import time
from gpt_index import SimpleDirectoryReader, GPTListIndex, readers, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel
import torch
import traceback
import faiss
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.environ.get("OPENAI_API_KEY")

# GPT-3 Default Model
COMPLETIONS_MODEL = "ada:ft-hyly-ai:greystar-2023-02-14-16-49-00"

# OpenAI Embedding API Model
openai_embedding_model = "text-embedding-ada-002"
df_content = pd.read_csv("data/final_with_content_finetune_data.csv")
openai_embedding_greystar = json.load(open("data/doc_embeddings_greystar.json"))


def get_openai_embedding(text: str, model_name=openai_embedding_model):
    result = openai.Embedding.create(
        model=model_name,
        input=text
    )
    return result["data"][0]["embedding"]


def vector_similarity(x, y):
    return np.dot(np.array(x), np.array(y))


def normalize_text(s, sep_token=" \n "):
    s = re.sub(r'\s+', ' ', s).strip()
    s = re.sub(r". ,", "", s)
    # remove all instances of multiple spaces
    s = s.replace("..", ".")
    s = s.replace(". .", ".")
    s = s.replace("\n", "")
    s = s.replace("#", "")
    s = s.strip()
    return s


def order_document_sections_by_query_similarity(query, contexts):
    query_embedding = get_openai_embedding(query)

    # vectory similarity between user input and embedding_dict
    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)
    return document_similarities


def compute_document_embedding(csv_file, output_dir):
    dataframe = pd.read_csv(csv_file)
    dataframe['prompt'] = dataframe["prompt"].apply(lambda x: normalize_text(x))
    dataframe['completion'] = dataframe["completion"].apply(lambda x: normalize_text(x))
    dataframe['content'] = "Question: " + dataframe['prompt'] + " " + "Answer: " + dataframe['completion']

    doc_emb = {}

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    embedding_file_path = f"{output_dir}/doc_embeddings_greystar.json"

    for idx, r in tqdm(dataframe.iterrows(), total=dataframe.shape[0]):
        try:
            doc_emb[idx] = get_openai_embedding(str(r.content))
        except:
            json.dump(doc_emb, open(embedding_file_path, 'w'))
            print(f"\nERROR REPORTED AT IDX: {idx}\n{traceback.format_exc()}\n")
            print("waiting for 120 s...")
            time.sleep(120)
            print(f"\nRESUMING FROM IDX: {idx}")
            doc_emb[idx] = get_openai_embedding(str(r.content))

    json.dump(doc_emb, open(embedding_file_path, 'w'))
    print(f"Document Embeddings file saved!\nSave path: {embedding_file_path}")
    return doc_emb


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
def construct_gpt_index(csv_file, output_dir):
    input_dir = 'gpt_index_data'
    try:
        shutil.rmtree(input_dir)
    except:
        pass

    if not os.path.exists(input_dir):
        os.makedirs(input_dir, exist_ok=True)

    dataframe = pd.read_csv(csv_file)
    dataframe['prompt'] = dataframe["prompt"].apply(lambda x: normalize_text(x))
    dataframe['completion'] = dataframe["completion"].apply(lambda x: normalize_text(x))
    dataframe['content'] = "Question: " + dataframe['prompt'] + " " + "Answer: " + dataframe['completion']

    df_gpt_index = dataframe['content']
    df_gpt_index.to_csv(f'{input_dir}/final_content.txt', header=None, index=None,
                        sep=' ', mode='a')

    max_input_size = 4096  # set maximum input size
    num_outputs = 256  # set number of output tokens
    max_chunk_overlap = 20  # set maximum chunk overlap
    chunk_size_limit = 600  # set chunk size limit

    # define LLM
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-ada-001", max_tokens=num_outputs))
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    documents = SimpleDirectoryReader(input_dir).load_data()
    index = GPTSimpleVectorIndex(
        documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper,
    )  # verbose=True

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    gpt_index_file_path = f'{output_dir}/greystar_gpt_index.json'
    index.save_to_disk(gpt_index_file_path)
    print(f"GPT Index Embeddings file saved!\nSave path: {gpt_index_file_path}")
    return index, gpt_index_file_path


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
    encoded_input = {k: v for k, v in encoded_input.items()}  # v.to(device)
    model_output = model(**encoded_input)
    return cls_pooling(model_output)


def construct_faiss_vector(csv_file, output_dir):
    dataframe = pd.read_csv(csv_file)
    dataset_ = Dataset.from_pandas(dataframe)
    faiss_dataset = dataset_.map(
        lambda x: {"embeddings": get_faiss_embeddings(x["content"]).detach().cpu().numpy()[0]}
    )
    faiss_embedding_vectors = np.array(faiss_dataset['embeddings']).astype('float32')
    faiss_vector_file_path = f"{output_dir}/greystar_faiss_vectors.npy"
    np.save(faiss_vector_file_path, faiss_embedding_vectors)
    print(f"FAISS Embeddings file saved!\nSave path: {faiss_vector_file_path}")
    return faiss_vector_file_path


# load vector file
faiss_vector_path = "data/greystar_faiss_vectors.npy"
faiss_vectors = np.load(faiss_vector_path)  # load
faiss_index_flat = faiss.IndexFlatL2(faiss_vectors.shape[1])  # build a flat (CPU) index
faiss_index_flat.add(faiss_vectors)  # add vectors to the index


def get_answer_from_faiss(user_input, dataframe):
    user_embedding = get_faiss_embeddings([user_input]).cpu().detach().numpy()
    k = 2  # we want to see 2 nearest neighbors
    scores, index_num = faiss_index_flat.search(user_embedding, k)
    samples_df = pd.DataFrame({"index_num": index_num.flatten(), "scores": scores.flatten()})
    samples_df.sort_values("scores", ascending=False, inplace=True)
    idx = samples_df.iloc[0].index_num
    result = dataframe.iloc[int(idx)].completion
    # for _, r in samples_df.iterrows():
    #     idx = int(r['index_num'])
    #     row = dataframe.iloc[idx]
    #     result = row.completion
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
