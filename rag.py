from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

load_dotenv()

st.set_page_config(
    page_title="Youtube Summarizer",
    page_icon="🎥"
)

st.title("🎥 YouTube Summarizer")
st.write("Enter a YouTube video URL and ask questions about it.")


youtube_url = st.text_input(
    "Enter Youtube URL",
    placeholder="https://www.youtube.com/watch?v=..."
)

def get_video_id(url):
    if "v" in url:
        return url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be")[1].split("?")[0]

    return url

if youtube_url:
    video_id = get_video_id(youtube_url)
    if st.button("Process Video"):
        with st.spinner("Getting transcript..."):
            try:
                ytapi = YouTubeTranscriptApi()
                try:
                    transcript_file = ytapi.fetch(video_id, languages=["en"])
                except Exception:
                    transcript_list = ytapi.list(video_id)
                    transcript_file = transcript_list.find_transcript(['en']).fetch()

                transcript = " ".join(
                    chunk.text if hasattr(chunk, "text") else chunk["text"]
                    for chunk in transcript_file
                )
                st.success("Transcript loaded!")

            except TranscriptsDisabled:
                st.error("This video does not have captions.")
                st.stop()
            except Exception:
                st.error("No transcript found for this video. Try a different YouTube URL with captions enabled.")
                st.stop()

        with st.spinner("Creating document chunks..."):
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks =splitter.create_documents([transcript])

        with st.spinner("Creating embeddings..."):
            embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
            vector_store = FAISS.from_documents(chunks,embeddings)
            st.success("Video processed successfully! 🎉")
            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":4})
            llm= ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.3)
            prompt = PromptTemplate(
                template ="""
                You are a helpful assistant.
                Answer ONLY from the provided transcript context.
                If the context is insufficient, just say you don't know.

                {context}
                Question: {question}
                """,
                input_variables=['context','question']
            )

            def format_docs(retrieved_docs):
                context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
                return context_text

            parallel_chain = RunnableParallel({
                'context': retriever | RunnableLambda(format_docs),
                'question': RunnablePassthrough()
            })

            parser = StrOutputParser()

            main_chain = parallel_chain | prompt | llm |parser
            st.session_state["main_chain"] = main_chain

if "main_chain" in st.session_state:

    st.divider()

    st.subheader("💬 Ask a question")

    question = st.text_input(
        "Question",
        placeholder="Can you summarize the video?"
    )

    if st.button("Ask"):

        if question:

            with st.spinner("Thinking..."):

                answer = st.session_state["main_chain"].invoke(
                    question
                )

            st.write("### Answer")

            st.write(answer)

        else:

            st.warning("Please enter a question.")
