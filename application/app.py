import pickle
import dotenv
from flask import Flask, request, render_template
import pickle
import dotenv
import os
from utils import answer_query_with_context

# Redirect PosixPath to WindowsPath on Windows
import platform
if platform.system() == "Windows":
    import pathlib
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

# loading the .env file
dotenv.load_dotenv()

with open("combine_prompt.txt", "r") as f:
    template = f.read()


app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/answer", methods=["POST"])
def api_answer():
    data = request.get_json()
    print(data)
    question = data["question"]
    api_key = data["api_key"]
    # check if the vectorstore is set\
    print(data)
    if data['active_docs'] !='Choose documentation':
        vectorstore = "vectorstores/" + data["active_docs"]

    else:
        vectorstore = ""
    #print(vectorstore)

    with open(os.path.join(vectorstore,"df.pkl"), "rb") as f:
        df = pickle.load(f)

    with open(os.path.join(vectorstore,"document_embeddings.pkl"), "rb") as f:
        document_embeddings = pickle.load(f)

    # loading the index and the store and the prompt template

    answer = answer_query_with_context(question, df, document_embeddings)
    result = {}
    result['answer'] = answer
    # some formatting for the frontend
    result['answer'] = result['answer'].replace("\\n", "<br>")
    #result['answer'] = result['answer'].replace("SOURCES:", "")
    # mock result
    # result = {
    #     "answer": "The answer is 42",
    #     "sources": ["https://en.wikipedia.org/wiki/42_(number)", "https://en.wikipedia.org/wiki/42_(number)"]
    # }
    # print('Answer: ', result['answer'])
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
