# AI-Powered Resume Generator

This project is an AI-powered Resume Generator using retrieval-augmented generation (RAG) and LaTeX formatting. It leverages OpenAI's API and NVIDIA models to dynamically generate professional, customized CVs based on the input job description and personal information.

## Key Features

- **Automated Resume Generation**: Generate LaTeX-formatted resumes tailored to specific job profiles.
- **Advanced NLP Models**: Utilizes the "meta/llama3-70b-instruct" model for generating content, with NVIDIA embeddings for efficient document processing.
- **Document Parsing**: Extracts personal information and qualifications from uploaded documents for accurate CV customization.
- **Enhanced LaTeX Formatting**: Ensures professional, well-structured output with custom formatting and optional enhancements like margin adjustments and text wrapping in tables.
- **Gradio Interface**: Provides an easy-to-use UI for document uploads and CV generation.

## Requirements

- Python 3.7+
- Gradio
- OpenAI
- NVIDIA API Key
- Additional packages: `llama_index`, `MilvusVectorStore`, and `pdflatex` for LaTeX compilation

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/resume-generator.git
   cd resume-generator
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your API keys by exporting them as environment variables:
   ```bash
   export NVIDIA_API_KEY='your_nvidia_api_key'
   export OPENAI_API_KEY='your_openai_api_key'
   ```

## Usage

1. Launch the Gradio interface:
   ```bash
   python resume_generator.py
   ```

2. Input your job profile and select documents containing your personal information. Click **Analyze** to extract and summarize the necessary qualifications.

3. Once documents are loaded, click **Generate CV** to create a LaTeX-formatted resume tailored to the specified job profile.

## File Structure

- **resume_generator.py**: Core application code for generating CVs using Gradio, LaTeX formatting, and RAG techniques.
- **generated_cv_latex.tex**: The generated LaTeX file containing the customized resume content.
- **project_path**: The root path where output files are stored and where LaTeX compilation occurs.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any feature enhancements or bug fixes.
