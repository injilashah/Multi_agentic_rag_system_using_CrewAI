import streamlit as st
import os
import tempfile
import gc
import base64
import time
import os

from crewai import Agent, Crew, Process, Task, LLM
from dotenv import load_dotenv
from DOcumentsearchtool import DocumentSearchTool
from DocumentRerankerTool import DocumentRerankerTool

load_dotenv()

llm = LLM(
        model="gemini/gemini-1.5-flash",

        api_key = os.getenv("GEMINI_API_KEY"),
        temperature =0
    )
llm2 = LLM(
        model="gemini/gemini-2.0-flash-exp",

        api_key = os.getenv("GEMINI_API_KEY"),
        temperature =0
    )
  

llm3 = LLM(
        model="gemini/gemini-1.5-pro",

        api_key = os.getenv("GEMINI_API_KEY"),
        temperature =0
    )

#   Define Agents & Tasks

def create_agents_and_tasks(pdf_tool,reranker_tool):
   
    
   
    #agent 1
    retriever_agent = Agent(
    role="Document Retriever.",
    goal="Retrieve relevant documents based on the given user query: {query}. You should use PDF search tool.",
    backstory="You are an expert at finding and retrieving documents that match the user's query.",
        verbose=True,
        tools=[pdf_tool],
        
        memory =False,
        llm=llm
        
    )

    #agent 2
    reranker_agent = Agent(
    role="Document Re-ranker.",
    goal="Re-rank the retrieved documents in order of relevance to the given query: {query}.You should use Reranker Tool",
    backstory="You are an expert at determining the relevance of documents based on the given query.",
    tools=[reranker_tool], 
    
    llm =llm2,
    memory =False,
    verbose=True
    )
    
    #agent 3
    response_synthesizer_agent = Agent(
        
    role="Response Generator.",
    goal="Generate a final response based on the  information from the first document in  re-reanked documents.",
    backstory="You are an expert at summarizing and generating responses from documents.",

        verbose=True,
        memory =False,
        llm=llm3
    )

    retrieve_task = Task(
    description="Retrieve documents  relevant to the given query: {query}.",
    
    expected_output="A list of relevant documents.",

        tools=[pdf_tool],
        async_execution=False,
        agent=retriever_agent)
    
    rerank_task = Task(
    description="Re-rank the retrieved documents in order of relevance to the query: {query}.",
    
    expected_output="A re-ranked list of documents, with the most relevant document first.",
    agent=reranker_agent,
    async_execution=False,)


    response_task = Task(
    description="Generate a final response based on the re-ranked documents for the query: {query}.",
    
    expected_output="A clear and concise response to the user's query.",

        async_execution=False,
        agent=response_synthesizer_agent
    )

    crew = Crew(
        agents=[retriever_agent,reranker_agent, response_synthesizer_agent],
        tasks=[retrieve_task, rerank_task,response_task],
        process=Process.sequential,  
        verbose=True
    )
    return crew


#   Streamlit Setup

if "messages" not in st.session_state:
    st.session_state.messages = []  # Chat history
if "reranker_tool" not in st.session_state:
    st.session_state.reranker_tool = None
if "pdf_tool" not in st.session_state:
    st.session_state.pdf_tool = None  # Store the DocumentSearchTool
if "file_path" not in st.session_state:
    st.session_state.file_path = None
     # Store the Crew object

def reset_chat():
    st.session_state.messages = []
    gc.collect()

def display_pdf(file_bytes: bytes, file_name: str):
    """Displays the uploaded PDF in an iframe."""
    base64_pdf = base64.b64encode(file_bytes).decode("utf-8")
    pdf_display = f"""
    <iframe 
        src="data:application/pdf;base64,{base64_pdf}" 
        width="100%" 
        height="600px" 
        type="application/pdf"
    >
    </iframe>
    """
    st.markdown(f"### Preview of {file_name}")
    st.markdown(pdf_display, unsafe_allow_html=True)


#   Sidebar

with st.sidebar:
    st.header("Add Your PDF Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file is not None:
        # If there's a new file and we haven't set pdf_tool yet...
        if st.session_state.pdf_tool is None:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir = tempfile.mkdtemp()
            temp_file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            with st.spinner("Indexing PDF... Please wait..."):
                st.session_state.pdf_tool = DocumentSearchTool(file_path=temp_file_path)
                st.session_state.file_path = temp_file_path
                
            st.success("PDF indexed!")

        # Optionally display the PDF in the sidebar
        display_pdf(uploaded_file.getvalue(), uploaded_file.name)

    st.button("Clear Chat", on_click=reset_chat)


#   Main Chat Interface

st.markdown("""
    # Multi Agentic RAG""")

# Render existing conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
prompt = st.chat_input("Ask a question about your PDF...")

if prompt:
    # 1. Show user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    print(type(prompt))
    # 2. Initialize the reranker tool if not already set
    if st.session_state.reranker_tool is None:
        st.session_state.reranker_tool = DocumentRerankerTool(st.session_state.file_path)

    # 2. Build or reuse the Crew (only once after PDF is loaded)
    #if st.session_state.crew is None:
        #st.session_state.crew = create_agents_and_tasks(st.session_state.pdf_tool,st.session_state.reranker_tool)
    # Always create a new crew for each query
    crew = create_agents_and_tasks(st.session_state.pdf_tool, st.session_state.reranker_tool)

    # Get the response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Get the complete response first
        with st.spinner("Thinking..."):
            inputs = {"query": prompt}
            result = crew.kickoff(inputs=inputs).raw
        
        # Split by lines first to preserve code blocks and other markdown
        lines = result.split('\n')
        for i, line in enumerate(lines):
            full_response += line
            if i < len(lines) - 1:  # Don't add newline to the last line
                full_response += '\n'
            message_placeholder.markdown(full_response + "▌")
            time.sleep(0.15)  # Adjust the speed as needed
        
        # Show the final response without the cursor
        message_placeholder.markdown(full_response)

    #  Save assistant's message to session
    st.session_state.messages.append({"role": "assistant", "content": result})