import PyPDF2
from langchain.text_splitter import CharacterTextSplitter
import pandas as pd
import openai
import dotenv,os
from transformers import GPT2TokenizerFast
import pickle
from pathlib import Path, PurePath
import time

import sys
from pathlib import Path, PurePath
sys.path.append(PurePath(Path(__file__).parents[1]).as_posix())
from utils.logging.custom_logging import logger

# set api key
env = dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# enter the document name for which vector to be created
document_name = str(input('Enter the PDF document name for which vector to be created(keep it short): '))

# pdf to text

pdfFileObj = open('TVS Jupiter 125 - SMW.pdf', 'rb')
pdfReader = PyPDF2.PdfReader(pdfFileObj)
num_pages = len(pdfReader.pages)
data = []
for page in range(0, num_pages):
    pageObj = pdfReader.pages[page]
    page_text = pageObj.extract_text()
    data.append(page_text)
pdfFileObj.close()
logger.info(f'Number of pages in the document is: {len(data)}')

# Split small chucks to so that LLMs can perform well
text_splitter = CharacterTextSplitter(chunk_size=1500, separator="\n")
docs = []
metadatas = []
sources = None
for i, d in enumerate(data):
    splits = text_splitter.split_text(d)
    docs.extend(splits)
    metadatas.extend([{"source": i}] * len(splits))

df = pd.DataFrame(metadatas)
df.insert(1, 'content', docs)
df.insert(1,'raw_index', df.index)
df = df.set_index(['raw_index',"source"])
logger.info(f'Number of rows in the document after chunk splits: {str(len(df))}')


# Tokenize

def count_tokens(text: str) -> int:
    """count the number of tokens in a string"""
    return len(tokenizer.encode(text))

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2") ##Todo: Use the logic provided by openai

content_token = [ count_tokens(text) for text in df.content.tolist()]
logger.info(f'Total number of tokens in document: {(str(sum(content_token)))}')
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
    logger.info(f'Embedding process is started')
    counter = 0
    embed_dict = {}
    for idx, r in df.iterrows():
        embed_dict[idx] = get_embedding(r.content)
        counter = counter + 1
        if counter == 25:
            counter = 0
            logger.info('waiting for 60 seconds')
            time.sleep(61) # Workaround for rate limit for a min
    logger.info(f'Embedding process is completed')
    return embed_dict

# compute embedding for the document
document_embeddings = compute_doc_embeddings(df)

# Save as pkl file

root_path = PurePath(Path(__file__).parents[1]).as_posix()
vector_path = os.path.join(root_path, 'application', 'vectorstores', 'tvs', f'{document_name}')
os.makedirs(vector_path, exist_ok=True)
# write docs.index and pkl file

df.to_pickle(os.path.join(vector_path,'df.pkl'))
df.to_csv(os.path.join(vector_path,'df.csv'))


with open(os.path.join(vector_path,"document_embeddings.pkl"), "wb") as f:
     pickle.dump(document_embeddings, f)

logger.info('Vectorization is successful')