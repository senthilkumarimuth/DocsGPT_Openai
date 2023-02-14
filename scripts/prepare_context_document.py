import PyPDF2
from langchain.text_splitter import CharacterTextSplitter
import pandas as pd
import openai
import dotenv,os
import numpy as np
import tiktoken
from transformers import GPT2TokenizerFast


# set api key
env = dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# enter the document name for which vector to be created
#document_name = str(input('Enter the document name for which vector to be created(keep it short): '))

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

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

content_token = [ count_tokens(text) for text in df.content.tolist()]
df.insert(1,'tokens', content_token)

COMPLETIONS_MODEL = "text-davinci-003"
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
document_embeddings = compute_doc_embeddings(df[:10])
#print(document_embeddings)

example_entry = list(document_embeddings.items())[0]
print(f"{example_entry[0]} : {example_entry[1][:5]}... ({len(example_entry[1])} entries)")

# Order document chunks by query similarity

def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    Returns the similarity between two vectors.

    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))

def order_document_sections_by_query_similarity(query: str, contexts: dict[(str, str), np.array]) -> list[
    (float, (str, str))]:
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections.

    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_embedding(query)

    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)

    return document_similarities

sections = order_document_sections_by_query_similarity("what is pdp", document_embeddings)[:5]
print(sections)


# Question Answer with Openai

MAX_SECTION_LEN = 500
SEPARATOR = "\n* "
ENCODING = "gpt2"  # encoding for text-davinci-003

encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))

f"Context separator contains {separator_len} tokens"


def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame) -> str:
    """
    Fetch relevant
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)

    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []

    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.
        document_section = df.loc[section_index]

        chosen_sections_len += document_section.tokens + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break

        chosen_sections.append(SEPARATOR + document_section.content.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))

    # Useful diagnostic information
    print(f"Selected {len(chosen_sections)} document sections:")
    print("\n".join(chosen_sections_indexes))

    header = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""

    return header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"


prompt = construct_prompt(
    "Who won the 2020 Summer Olympics men's high jump?",
    document_embeddings,
    df
)

print("===\n", prompt)


COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0.0,
    "max_tokens": 300,
    "model": COMPLETIONS_MODEL,
}

def answer_query_with_context(
        query: str,
        df: pd.DataFrame,
        document_embeddings: dict[(str, str), np.array],
        show_prompt: bool = False) -> str:
    prompt = construct_prompt(
            query,
            document_embeddings,
            df
        )

    if show_prompt:
        print(prompt)

    response = openai.Completion.create(
        prompt=prompt,
        **COMPLETIONS_API_PARAMS
    )

    return response["choices"][0]["text"].strip(" \n")


result = answer_query_with_context("what is GraphQL", df, document_embeddings)
print(result)
