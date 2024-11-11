import os
from openai import OpenAI
import gradio as gr
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.llms.nvidia import NVIDIA
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings

# ------------------------------------------------------------------------------------------------Functions----------------------------------------------------------------------------------------
def get_ai_response_ot(model_use="meta/llama3-70b-instruct", message=None, max_tokens=1024, stream_var=False, temperature_m=0.5, top_p_value=1):
    client = OpenAI(
    base_url = "https://integrate.api.nvidia.com/v1",
    api_key = os.getenv('NVIDIA_API_KEY')
    )
    completion = client.chat.completions.create(
        model=model_use,
        messages=message,
        temperature=temperature_m,
        top_p=top_p_value,
        max_tokens=max_tokens,
        stream=stream_var
    )

    generated_text = str(completion.choices[0].message.content)
    clean_text = "\n".join(
        line for line in generated_text.splitlines() if "here" not in line.lower() and "latex" not in line.lower()
    )
    return clean_text

def calculate_tokens(messages):
    total_tokens = 0
    for message in messages:
        total_tokens += len(message['content'].split())
    return total_tokens

def compile_latex(path):
    import subprocess
    try:
        # Define the working directory
        working_dir = path

        # Compile the LaTeX file using pdflatex with non-interactive options
        result = subprocess.run(
            [
                "pdflatex",
                "-interaction=nonstopmode",   # Run without stopping for user input
                "generated_cv_latex.tex"
            ],
            cwd=working_dir,
            stdin=subprocess.DEVNULL,          # Prevent pdflatex from waiting for input
            capture_output=True,
            text=True,
            check=True
        )

        print("Compilation Output:")
        print(result.stdout)
        print("Compilation Errors (if any):")
        print(result.stderr)

    except subprocess.CalledProcessError as e:
        print("An error occurred during LaTeX compilation:")
        print(e.stdout)
        print(e.stderr)

    except Exception as e:
        print("An unexpected error occurred:")
        print(str(e))

def make_latex(my_info, job_profile, project_path):
    max_context = 8192
    # CV Sections
    cv_sections = []
    # List of sections to generate
    sections = ["Personal Information and Summary", "Education", "Work Experience", "Skills and Projects", "Hobbies"]

    for section in sections:
        # Prepare the messages for the API call
        messages_profile = [
            {
                "role": "system",
                "content": (
                    f"You are a resume generator. Generate the '{section}' section "
                    "of a CV in LaTeX format based on the provided details. Be concise and relevant."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Personal information:\n{my_info}\n\n"
                    f"Job vacancy details:\n{job_profile}"
                ),
            },
        ]
        tokens_in_messages = calculate_tokens(messages_profile)
        buffer_tokens = 100  # Buffer to prevent exceeding the limit
        available_tokens = max_context - tokens_in_messages - buffer_tokens
        max_tokens = min(512, available_tokens)

        # Ensure there are enough tokens for the response
        if max_tokens <= 0:
            raise ValueError(f"Not enough tokens available for the '{section}' section.")

        # Append the generated section to the cv_sections list
        generated_text = get_ai_response_ot(message=messages_profile)

        cv_sections.append(generated_text)

        print(f"Generated '{section}' section successfully.")

    # Combine all sections into the final CV
    final_cv = "\n\n".join(cv_sections)
    print("The CV has been generated and saved as 'generated_cv.tex'.")

    # Auditor message
    auditor_message = [
        {
            "role": "system",
            "content": (
                "You are a LaTeX auditor. Your task is to enhance the structure, indentation, and visual appeal of the provided LaTeX document. "
                "Ensure all sections, especially tables and lists, are aligned and indented consistently for a professional appearance. "
                "If using tables, adjust column widths and ensure content is properly wrapped, without uneven spacing or awkward line breaks. "
                "Use techniques such as setting appropriate column widths, adding line breaks where necessary, and aligning text to avoid cluttered or uneven formatting. "
                "If necessary, add packages like `geometry` (for margins), `hyperref` (for links), `enumitem` (for lists), and `array` or `longtable` (for text wrapping in tables). "
                "Ensure that only the modified LaTeX code is returned. If any additional text, explanations, or comments are generated inadvertently, they should be commented out."
                "If any additional text or comments are generated, they should be commented out."
            )
        },
        {
            "role": "user",
            "content": final_cv
        }
    ]
    auditor_generated_text = get_ai_response_ot(message=auditor_message, max_tokens=2000)

    with open('generated_cv_latex.tex', 'w', encoding='utf-8') as file:
        file.write(auditor_generated_text)
    #compile_latex(path=project_path)

# Function to get file names from file objects
def get_files_from_input(file_objs):
    if not file_objs:
        return []
    return [file_obj.name for file_obj in file_objs]

# Function to load documents and create the index
def load_documents(file_objs, progress=gr.Progress()):
    global index, query_engine
    try:
        if not file_objs:
            return "Error: No files selected."

        file_paths = get_files_from_input(file_objs)
        documents = []
        for file_path in file_paths:
            documents.extend(SimpleDirectoryReader(input_files=[file_path]).load_data())

        if not documents:
            return f"No documents found in the selected files."

        # Create a Milvus vector store and storage context
        vector_store = MilvusVectorStore(uri="./milvus_demo.db", dim=1024, overwrite=True)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Create the index from the documents
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

        # Create the query engine
        query_engine = index.as_query_engine(similarity_top_k=5)
        return f"Successfully loaded {len(documents)} documents from {len(file_paths)} files."
    except Exception as e:
        return f"Error loading documents: {str(e)}"

def get_job_data():
    global query_engine
    global job_description_info
    project_path = os.getcwd()
    if query_engine is None:
        return "Please load documents first."
    try:
        message = "Extract my personal information from the documents."
        response = query_engine.query(message)
        user_personal_info = response.response
        make_latex(my_info=user_personal_info, job_profile=job_description_info, project_path=project_path)
        return "CV Generated Successfully"
    except Exception as e:
        return f"Error processing query: {str(e)}"

def job_description(job_profile):
    global job_description_info
    job_profile_in = [
        {
            "role": "user",
            "content": f"Summarize the most relevant qualifications and responsibilities of the following job description, focusing only on key details needed for the role without introductory text or extra explanation:\n\n{job_profile}"
        }
    ]
    job_description_info = get_ai_response_ot(
        message=job_profile_in,
        max_tokens=300,
        stream_var=False,
        temperature_m=0.9,
        top_p_value=0.9
    )

    return f"Job Profile Analyzed: {job_description_info}"

# -------------------------------------------------------------------------------------------------Main--------------------------------------------------------------------------------------------
Settings.text_splitter = SentenceSplitter(chunk_size=500)
Settings.embed_model = NVIDIAEmbedding(model="NV-Embed-QA", truncate="END")
Settings.llm = NVIDIA(model="meta/llama3-70b-instruct")

# Initialize global variables for the index and query engine
index = None
query_engine = None
job_description_info = None

project_path = os.getcwd()

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# CV Generator using RAG Application")
    cv_profile = gr.Textbox(label="Enter the type of profile for the resume you want to be generated:")
    output = gr.Textbox(label="Output Box")
    analyze_btn = gr.Button("Analyze")
    analyze_btn.click(fn=job_description, inputs=cv_profile, outputs=output, api_name="job_description")
    with gr.Row():
        file_input = gr.File(label="Select files to load", file_count="multiple")
        load_btn = gr.Button("Load Documents")
    load_output = gr.Textbox(label="Load Status")
    generate_cv_btn = gr.Button("Generate CV")
    cv_output_file = gr.File(label="Generated CV")

    # Set up functions
    load_btn.click(load_documents, inputs=[file_input], outputs=[load_output], show_progress="hidden")
    generate_cv_btn.click(fn=get_job_data, inputs=[], outputs=output, api_name="greet")

demo.launch()
