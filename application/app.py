from flask import Flask, request, render_template
import pickle
import dotenv
import os
from utils import answer_query_with_context

# loading the .env file
dotenv.load_dotenv()

# load prompt template
with open("combine_prompt.txt", "r") as f:
    template = f.read()


app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/answer", methods=["POST"])
def api_answer():
    data = request.get_json()
    question = data["question"]

    # check if the vector store is set
    if data['active_docs'] !='Choose documentation':
        vector_store = "vectorstores/" + data["active_docs"]
    else:
        vector_store = ""
    # load pkl from vector store
    with open(os.path.join(vector_store,"df.pkl"), "rb") as f:
        df = pickle.load(f)
    with open(os.path.join(vector_store,"document_embeddings.pkl"), "rb") as f:
        document_embeddings = pickle.load(f)

    # loading the index and the store and the prompt template
    answer = answer_query_with_context(question, df, document_embeddings, template, show_prompt=True)
    result = {'answer': answer}
    print(answer)
    # some formatting for the frontend
    temp = result['answer'].split('SOURCES:')
    result['answer'] = result['answer'].replace("\\n", "<br>")
    result['answer'] = temp[0]
    print('SOURCE: ', temp[1])
    #result['answer'] = result['answer'].replace("SOURCES:", "")
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
