import pandas as pd

import utils as utils

import os

import openai
from dotenv import load_dotenv
from flask import Flask, request

load_dotenv()

openai.api_key = os.environ.get("OPENAI_API_KEY")

app = Flask(__name__)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/finetuned-gpt3-model", methods=["POST"])
def finetuned_gpt3_completion():
    data = request.json
    prompt = data['prompt'] + "?"
    response = openai.Completion.create(
        model=utils.COMPLETIONS_MODEL,
        prompt=prompt,
        max_tokens=1500,
        temperature=0,
        stop=[" END"],
    )
    response_str = response["choices"][0]["text"].replace("\n", "").strip()
    return response_str


@app.route("/create-embedding-file", methods=["POST"])
def generate_openai_embedding_file():
    data = request.json
    csv_path = data['csv_file']
    df = pd.read_csv(csv_path)

    embedding_file_path = utils.compute_document_embedding(
        csv_file=df,
        output_dir='data'
    )
    response_str = f"OpenAI embeddings file saved!\nSave path: {str(embedding_file_path)}"
    return response_str


@app.route("/openai-embedding-api", methods=["POST"])
def embedding_api_completion():
    data = request.json
    prompt = data['prompt'] + "?"

    response = utils.get_answer_from_openai_embedding_dict(
        user_input=prompt,
        context_embedding=utils.openai_embedding_greystar,
        dataframe=utils.df_content
    )
    return str(response).strip()


@app.route("/create-gpt-index-file", methods=["POST"])
def gpt_index_vector_file():
    data = request.json
    csv_path = data['csv_file']
    df = pd.read_csv(csv_path)

    _, gpt_index_file_path = utils.construct_gpt_index(
        csv_file=df,
        output_dir='data'
    )
    response_str = f"GPT Index embeddings file saved!\nSave path: {str(gpt_index_file_path)}"
    return response_str


@app.route("/gpt-index-model", methods=["POST"])
def gpt_index_completion():
    data = request.json
    prompt = data['prompt'] + "?"

    response = utils.gpt_index_vector_index.query(prompt, response_mode="compact")
    return str(response).strip()


@app.route("/create-faiss-file", methods=["POST"])
def faiss_vector_file():
    data = request.json
    csv_path = data['csv_file']
    df = pd.read_csv(csv_path)

    faiss_vector_file_path = utils.construct_faiss_vector(
        csv_file=df,
        output_dir='data'
    )
    response_str = f"FAISS embeddings file saved!\nSave path: {str(faiss_vector_file_path)}"
    return response_str


@app.route("/faiss-model", methods=["POST"])
def faiss_completion():
    data = request.json
    prompt = data['prompt'] + "?"

    response = utils.get_answer_from_faiss(
        user_input=prompt,
        dataframe=utils.df_content
    )
    return str(response)


@app.route("/combined-model", methods=["POST"])
def combined_completion():
    data = request.json
    prompt = data['prompt'] + "?"

    response = utils.get_answer_with_combined_approach(user_input=prompt)
    return str(response)


if __name__ == '__main__':
    app.run(debug=True)
