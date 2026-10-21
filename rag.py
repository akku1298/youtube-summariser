import re
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
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


def get_video_id(url):
    """Extract YouTube video ID from various URL formats."""
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/|youtube\.com\/v\/|youtube\.com\/shorts\/)([^&\n?#]+)',
        r'youtube\.com\/watch\?.*v=([^&\n?#]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return url


youtube_url = st.text_input(
    "Enter Youtube URL",
    placeholder="https://www.youtube.com/watch?v=..."
)

if youtube_url:
    video_id = get_video_id(youtube_url)
    if st.button("Process Video"):
        with st.spinner("Getting transcript..."):
            try:
                ytapi = YouTubeTranscriptApi()

                # Try English variants first, then any available transcript
                lang_codes = ["en", "en-US", "en-GB", "en-IN", "a.en"]

                transcript_file = None
                for lang in lang_codes:
                    try:
                        transcript_file = ytapi.fetch(video_id, languages=[lang])
                        break
                    except Exception:
                        continue

                if transcript_file is None:
                    # Last resort: list all available and try the first one
                    try:
                        transcript_list = ytapi.list(video_id)
                        available = [t.language_code for t in transcript_list]
                        st.info(f"Available transcript languages: {available}")
                        if available:
                            transcript_file = transcript_list.find_transcript(available).fetch()
                    except Exception as e:
                        st.error(f"Could not list transcripts: {str(e)}")
                        st.stop()

                if transcript_file is None:
                    st.error("No transcript found for this video.")
                    st.stop()

                transcript = " ".join(
                    chunk.text if hasattr(chunk, "text") else chunk["text"]
                    for chunk in transcript_file
                )
                st.success("Transcript loaded!")

            except TranscriptsDisabled:
                st.error("This video does not have captions.")
                st.stop()
            except NoTranscriptFound:
                st.error("No transcript found for this video. Try a different YouTube URL with captions enabled.")
                st.stop()
            except Exception as e:
                st.error(f"Error loading transcript: {str(e)}")
                st.stop()

        with st.spinner("Creating document chunks..."):
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.create_documents([transcript])

        with st.spinner("Creating embeddings..."):
            embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
            vector_store = FAISS.from_documents(chunks, embeddings)
            st.success("Video processed successfully! 🎉")
            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
            prompt = PromptTemplate(
                template="""
                You are a helpful assistant.
                Answer ONLY from the provided transcript context.
                If the context is insufficient, just say you don't know.

                {context}
                Question: {question}
                """,
                input_variables=['context', 'question']
            )

            def format_docs(retrieved_docs):
                context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
                return context_text

            parallel_chain = RunnableParallel({
                'context': retriever | RunnableLambda(format_docs),
                'question': RunnablePassthrough()
            })

            parser = StrOutputParser()

            main_chain = parallel_chain | prompt | llm | parser
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