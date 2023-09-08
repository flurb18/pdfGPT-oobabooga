import gradio as gr
import fitz
import tensorflow_hub as hub
import numpy as np
import re
from sklearn.neighbors import NearestNeighbors

from modules.text_generation import generate_reply
from modules.ui import gather_interface_values, list_interface_input_elements
from modules.utils import gradio

recommender = None

params = {
    "display_name" : "pdfGPT_oobabooga",
    "is_tab" : True
}

def preprocess(text):
    text = text.replace('\n', ' ')
    text = re.sub('\s+', ' ', text)
    return text

def pdf_to_text(path, start_page=1, end_page=None):
    doc = fitz.open(path)
    total_pages = doc.page_count

    if end_page is None:
        end_page = total_pages

    text_list = []

    for i in range(start_page - 1, end_page):
        text = doc.load_page(i).get_text("text")
        text = preprocess(text)
        text_list.append(text)

    doc.close()
    return text_list


def text_to_chunks(texts, word_length=150, start_page=1):
    text_toks = [t.split(' ') for t in texts]
    chunks = []

    for idx, words in enumerate(text_toks):
        for i in range(0, len(words), word_length):
            chunk = words[i : i + word_length]
            if (
                (i + word_length) > len(words)
                and (len(chunk) < word_length)
                and (len(text_toks) != (idx + 1))
            ):
                text_toks[idx + 1] = chunk + text_toks[idx + 1]
                continue
            chunk = ' '.join(chunk).strip()
            chunk = f'[Page no. {idx+start_page}]' + ' ' + '"' + chunk + '"'
            chunks.append(chunk)
    return chunks


class SemanticSearch:
    def __init__(self):
        self.use = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
        self.fitted = False

    def fit(self, data, batch=1000, n_neighbors=5):
        self.data = data
        self.embeddings = self.get_text_embedding(data, batch=batch)
        n_neighbors = min(n_neighbors, len(self.embeddings))
        self.nn = NearestNeighbors(n_neighbors=n_neighbors)
        self.nn.fit(self.embeddings)
        self.fitted = True

    def __call__(self, text, return_data=True):
        inp_emb = self.use([text])
        neighbors = self.nn.kneighbors(inp_emb, return_distance=False)[0]

        if return_data:
            return [self.data[i] for i in neighbors]
        else:
            return neighbors

    def get_text_embedding(self, texts, batch=1000):
        embeddings = []
        for i in range(0, len(texts), batch):
            text_batch = texts[i : (i + batch)]
            emb_batch = self.use(text_batch)
            embeddings.append(emb_batch)
        embeddings = np.vstack(embeddings)
        return embeddings

def load_recommender(fileObject, start_page=1):
    global recommender
    if recommender is None:
        recommender = SemanticSearch()
    texts = pdf_to_text(fileObject.name, start_page=start_page)
    chunks = text_to_chunks(texts, start_page=start_page)
    recommender.fit(chunks)
    return 'Corpus Loaded.'

def generate_answer(question, state):
    if recommender is None:
        return "No file loaded"
    topn_chunks = recommender(question)
    prompt = 'Search results:\n\n'
    for c in topn_chunks:
        prompt += c + '\n\n'
    prompt += """
        Instructions: Compose a comprehensive reply to the query using the search results given.
        Cite each reference using [ Page Number] notation (every result has this number at the beginning).
        Citation should be done at the end of each sentence. If the search results mention multiple subjects with the same name, create separate answers for each.
        Only include information found in the results and don't add any additional information.
        Make sure the answer is correct and don't output false content.
        If the text does not relate to the query, simply state 'Text Not Found in PDF'.
        Ignore outlier search results which has nothing to do with the question. Only answer what is asked.
        The answer should be short and concise. Answer step-by-step.\n\n"""
    prompt += f"Query: {question}\nAnswer:"
    answer = generate_reply(prompt, state)
    return answer

def ui():
    state = gr.State({})
    with gr.Row():
        with gr.Group():
            f = gr.File(
                label='Upload your PDF/ Research Paper / Book here', file_types=['.pdf']
            )
            question = gr.Textbox(label='Enter your question here')
            btn = gr.Button(value='Submit')
            btn.style(full_width=True)

        with gr.Group():
            answer = gr.Textbox(label='The answer to your question is :')

        btn.click(
            gather_interface_values,
            inputs=gradio(list_interface_input_elements()),
            outputs=state
        ).then(
            generate_answer,
            inputs=[question, state],
            outputs=[answer]
        )
        question.submit(
            gather_interface_values,
            inputs=gradio(list_interface_input_elements()),
            outputs=state
        ).then(
            generate_answer,
            inputs=[question, state],
            outputs=[answer]
        )
        f.upload(
            load_recommender,
            inputs=[f],
            outputs=None
        )