# **Travel Policy RAG Pipeline with Fine-Tuned LLaMa 2**

## **Project Overview**
This project implements a **Retrieval-Augmented Generation (RAG) pipeline** using **LangChain** and a **fine-tuned LLaMa 2 model** to answer user queries related to USA or Canadian travel policies and guidelines.  
The pipeline leverages **Pinecone** as a vector database and a classification LLM to identify the relevant context (USA or Canada).  

The LLaMa 2 model has been fine-tuned on a specific dataset hosted on HuggingFace to enhance its accuracy for travel-related questions.

---

## **File Structure**

1. **`Finetuning_Llama_2_using_QLORA.ipynb`**  
   - Contains the code for fine-tuning the LLaMa 2 model using QLoRA (Parameter Efficient Fine-Tuning) on **Google Colab Free Tier**.  
   - The model is trained on a HuggingFace dataset, optimized for travel policy question answering.  

2. **`ingest.py`**  
   - Handles document ingestion into the **Pinecone** vector database.  
   - Steps include:  
     - Extracting text from relevant PDF documents.  
     - Splitting text into smaller chunks for embedding.  
     - Generating embeddings using a HuggingFace embedding model.  
     - Uploading embeddings to the Pinecone database under appropriate namespaces (`usa_docs` and `canada_docs`).

3. **`RAG_on_fine_tuned_LLAMA_model.ipynb`**  
   - Implements the RAG pipeline to answer user queries.  
   - Steps include:  
     1. **Model Initialization**:  
        - Fine-tuned **LLaMa 2** for answer generation.  
        - **Mixtral** model (Groq) to classify queries as "USA" or "Canada".  
        - HuggingFace embedding model for embedding user queries.  
     2. **Classification**:  
        - The query is classified to determine whether it pertains to **USA** or **Canada** travel guidelines.  
     3. **Retrieval**:  
        - Based on the classification result, the pipeline connects to the relevant Pinecone **namespace** (`usa_docs` or `canada_docs`).  
        - Retrieves the most relevant document chunks using a vectorstore-backed retriever.  
     4. **Answer Generation**:  
        - A LangChain prompt template combines the retrieved context and the user query.  
        - The fine-tuned LLaMa 2 generates the final response.  

4. **`requirements.txt`**  
   - Contains all the dependencies required to run the project, including:  
     - LangChain  
     - Pinecone  
     - HuggingFace Transformers  
     - Groq (for Mixtral model)  
     - PyPDFLoader (for PDF processing)  

---

## Setup instructions


## **Key Features**
1. **Fine-Tuned LLaMa 2**  
   - LLaMa 2 is fine-tuned specifically for travel-related queries, ensuring high-quality answers.  
   - QLoRA fine-tuning method optimizes training for limited resources.

2. **Query Classification**  
   - Utilizes Mixtral model (Groq) to classify queries as "USA" or "Canada".  
   - Enables accurate retrieval of relevant context.  

3. **Efficient Retrieval**  
   - PDF documents are ingested into Pinecone with namespace mapping for **USA** and **Canada** guidelines.  
   - Embedding-based retrieval ensures the most relevant document chunks are fetched.  

4. **RAG Pipeline**  
   - Combines retrieved context and fine-tuned LLaMa 2 to generate precise, context-aware answers.  

5. **Scalability**  
   - Uses Pinecone for scalable vector storage and retrieval.  
   - Modular pipeline that can be extended for additional namespaces or models.  

---

## **How It Works**

### **1. Document Ingestion**  
- Run the `ingest.py` file to process relevant PDF documents and upload embeddings to Pinecone.  
- Document chunks are stored under separate namespaces:  
   - `usa_guides` for USA travel policies.  
   - `canada_guides` for Canadian travel policies.  

### **2. RAG Pipeline**  
- Run the `RAG_on_fine_tuned_LLAMA_model.ipynb` notebook.  
- User query is:  
   - **Classified**: Determines if it's related to USA or Canada.  
   - **Retrieved**: Connects to Pinecone to fetch context from the relevant namespace.  
   - **Answered**: Combines retrieved context with the query, and the fine-tuned LLaMa 2 model generates the answer.

---

## Dataset information
The dataset was taken from Huggingface having around 1000 samples. The link is https://huggingface.co/mlabonne/llama-2-7b-guanaco

## Acknowledgments
HuggingFace for pre-trained models and datasets.
Pinecone for scalable vector storage.
Groq for providing Mixtral LLM.