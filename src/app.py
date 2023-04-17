from flask import Flask, request, render_template
import pickle
import dotenv
import os
from src.utils.common import answer_query_with_context
from langchain.memory import ConversationBufferWindowMemory
import sys
from pathlib import Path, PurePath
sys.path.append(PurePath(Path(__file__).parents[1]).as_posix())
from utils.logging.custom_logging import logger


# loading the .env file
dotenv.load_dotenv()

# load prompt template

memory = ConversationBufferWindowMemory(k=2)
memory.ai_prefix = 'FINAL ANSWER'
memory.human_prefix = 'QUESTION'

logger.debug('Starting Flask APP')
app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/answer", methods=["POST"])
def api_answer():
    data = request.get_json()
    question = data["question"]
    logger.info(f'Question: {question}')
    # check if the vector store is set
    if data['active_docs'] !='Choose documentation':
        vector_store = "vectorstores/" + data["active_docs"]
    else:
        vector_store = ""
    logger.debug(f'vectorstore path: {vector_store}')
    # load pkl from vector store
    with open(os.path.join(vector_store,"df.pkl"), "rb") as f:
        df = pickle.load(f)
    with open(os.path.join(vector_store,"document_embeddings.pkl"), "rb") as f:
        document_embeddings = pickle.load(f)
    # loading prompt
    prompt_path = os.path.join("./prompts", data["active_docs"])
    logger.debug(f'prompt path: {prompt_path}')
    with open(os.path.join(prompt_path,"combine_prompt.txt"), "r") as f:
         template = f.read()

    # loading the index and the store and the prompt template

    answer = answer_query_with_context(question, df, document_embeddings, template, memory, show_prompt=True)
    result = {'answer': answer}
    #some formatting for the frontend
    temp = result['answer'].split('SOURCES:')
    result['answer'] = result['answer'].replace("\\n", "<br>")
    # SOURCE = ""
    # if temp[1] != ' None':
    #     result['answer'] = temp[0]
    #     SOURCE = temp[1]
    #     logger.debug(f'Sources/Page from which the answer is derived: {str(temp[1])}')
    # else:
    #     if temp[0] != "I don't know.\n":
    #         result['answer'] = temp[0] + "\n" + "Source: Answer is not from this document"
    #         logger.debug(f'Sources/Page from which the answer is derived: Answer is not from this document')
    #     else:
    #         result['answer'] = temp[0]

    #result['answer'] = result['answer'].replace("SOURCES:", "")
    logger.info(f'Answer: {result["answer"]}')
    memory.save_context({"input": question}, {"ouput": result["answer"]})
    return result


# handling CORS
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response


if __name__ == "__main__":
    app.run(debug=True)
