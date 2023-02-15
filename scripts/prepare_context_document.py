import PyPDF2
from langchain.text_splitter import CharacterTextSplitter
import pandas as pd
import openai
import dotenv,os
from transformers import GPT2TokenizerFast
import pickle
from pathlib import Path, PurePath


# set api key
env = dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# enter the document name for which vector to be created
document_name = str(input('Enter the PDF document name for which vector to be created(keep it short): '))

# pdf to text

pdfFileObj = open('PDP document for QA bot_v1.pdf', 'rb')
pdfReader = PyPDF2.PdfReader(pdfFileObj)
num_pages = len(pdfReader.pages)
print(pdfReader.pages)
data = []
for page in range(0, num_pages):
    print('print', page)
    pageObj = pdfReader.pages[page]
    page_text = pageObj.extract_text()
    data.append(page_text)
pdfFileObj.close()
data = data[0:15]  # Number of page for which vector is created

text_splitter = CharacterTextSplitter(chunk_size=1500, separator="\n")
docs = []
metadatas = []
for i, d in enumerate(data):
    splits = text_splitter.split_text(d)
    #print(i, len(splits))
    docs.extend(splits)
    #metadatas.extend([{"source": sources[i]}] * len(splits))

metadatas = [{"source":"PDP DOCUMENTATION INDEX"}, {"source":"SUPPORT"},{"source":"API INDEX BY TYPE"},
    {"source":"INTRO TO PDP"},{"source":"How PDP differs from After-market devices?"},
    {"source":"PDP’s APIs RePEAT"}, {"source":"Quick brief about GraphQL"},{"source":"GraphQL Methods"},
    {"source":"Modules"}]

df = pd.DataFrame(metadatas)
df.insert(1, 'content', docs)
df = df.set_index(["source"])
print(f"{len(df)} rows in the data.")
#df.sample(5)

# Tokenize

def count_tokens(text: str) -> int:
    """count the number of tokens in a string"""
    return len(tokenizer.encode(text))

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2") ##Todo: Use the logic provided by openai

content_token = [ count_tokens(text) for text in df.content.tolist()]
df.insert(1, 'tokens', content_token)


COMPLETIONS_MODEL = "text-davinci-003"   ##Todo: Babbage/Curie models may be more suitable. Need to test this
EMBEDDING_MODEL = "text-embedding-ada-002"

def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> list[float]:
    result = openai.Embedding.create(
        model=model,
        input=text
    )
    return result["data"][0]["embedding"]

def compute_doc_embeddings(df: pd.DataFrame) -> dict[tuple[str, str], list[float]]:
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.

    Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
    """
    return {
        idx: get_embedding(r.content) for idx, r in df.iterrows()
    }

# compute embedding for the document
document_embeddings = compute_doc_embeddings(df)

# Save as pkl file

root_path = PurePath(Path(__file__).parents[1]).as_posix()
vector_path = os.path.join(root_path, 'application', 'vectorstores', 'tvs', f'{document_name}')
os.makedirs(vector_path, exist_ok=True)
# write docs.index and pkl file

df.to_pickle(os.path.join(vector_path,'df.pkl'))

with open(os.path.join(vector_path,"document_embeddings.pkl"), "wb") as f:
     pickle.dump(document_embeddings, f)

# Todo: File doesn't terminate once the code run