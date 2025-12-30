import streamlit as st
import os

from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_classic.tools.retriever import create_retriever_tool
from langchain_classic.agents import create_openai_tools_agent, AgentExecutor
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_classic import hub

st.set_page_config(page_title="LangChain Search Agent", page_icon="🤖")

st.title("🔎 Search Engine with Tools & Agents")
st.markdown("This agent uses **Wikipedia**, **DuckDuckGo Search** and **Arxiv**. You can also provide a **Custom URL** in the sidebar to add your own data.")

st.sidebar.header("Configuration")
groq_api_key = st.sidebar.text_input("Groq API Key", type="password")
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
custom_url = st.sidebar.text_input("Custom URL (Optional)", placeholder="e.g., https://example.com/article")

if not groq_api_key or not openai_api_key:
    st.warning("Please enter both your Groq and OpenAI API Keys in the sidebar to proceed.")
    st.stop()

os.environ["GROQ_API_KEY"] = groq_api_key
os.environ["OPENAI_API_KEY"] = openai_api_key

@st.cache_resource(show_spinner=False)
def setup_tools(target_url):
    tools_list = []

    api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
    wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)
    tools_list.append(wiki)

    api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
    arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)
    tools_list.append(arxiv)

    search = DuckDuckGoSearchRun(name="internet_search")
    tools_list.append(search)

    if target_url:
        try:
            loader = WebBaseLoader(target_url)
            docs = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            documents = text_splitter.split_documents(docs)

            embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
            vectordb = FAISS.from_documents(documents, embeddings)
            retriever = vectordb.as_retriever()

            retriever_tool = create_retriever_tool(
                retriever=retriever,
                name="custom-web-search",
                description="Search for information from the specific custom URL provided by the user."
            )
            tools_list.append(retriever_tool)
        except Exception as e:
            st.error(f"Could not load the custom URL: {e}")

    return tools_list

with st.spinner("Initializing tools..."):
    tools = setup_tools(custom_url)

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="openai/gpt-oss-120b"
)

prompt = hub.pull("hwchase17/openai-functions-agent")

agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt_input := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = agent_executor.invoke({"input": prompt_input})
                output_text = response["output"]
                st.markdown(output_text)
                st.session_state.messages.append({"role": "assistant", "content": output_text})
            except Exception as e:
                st.error(f"An error occurred: {e}")
