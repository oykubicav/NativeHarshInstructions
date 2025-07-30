from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import pipeline

# Define LLM pipeline
pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",  # string must be in quotes
    max_new_tokens=256,
    early_stopping=True,
    do_sample=False,
    no_repeat_ngram_size=6,
    repetition_penalty=1.2  # you had a typo: "repetation_penalty"
)

# Wrap LLM pipeline with LangChain
llm = HuggingFacePipeline(pipeline=pipe)

# Define embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
