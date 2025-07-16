import validators
import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain.chains.summarize import load_summarize_chain
#from langchain.callbacks import StreamlitCallbackHandler
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.retrieval_qa.base import RetrievalQA


st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')

groq_api_key=st.sidebar.text_input(label="GRoq API Key",type="password")

if not groq_api_key:
    st.info("Please add the groq api key")

if 'docs' not in st.session_state:
    st.session_state.docs = None
if 'summary' not in st.session_state:
    st.session_state.summary = None

llm=ChatGroq(groq_api_key=groq_api_key,model_name="Llama3-8b-8192",streaming=True)

generic_url=st.text_input('URL', label_visibility="collapsed")

prompt_template=''' 
provide summary of the following content in 300 words
content:{text}
'''
prompt=PromptTemplate(template=prompt_template, input_variables=['text'])

if st.button('summarize'):
    if not groq_api_key.strip() or not generic_url.strip():
        st.error('please provide all details')
    elif not validators.url(generic_url):
        st.error('please provide valid url')
    else:
        try:
            with st.spinner('waiting...'):
                if 'youtube.com' in generic_url:
                    loader=YoutubeLoader.from_youtube_url(generic_url,add_video_info=True)
                    
                    video_id = generic_url.split("v=")[-1].split("&")[0]
                    try:
                        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
                        transcript = " ".join([t["text"] for t in transcript_list])
                        docs = [Document(page_content=transcript)]
                    except Exception as e:
                        st.error(f"Transcript not available for this video. Error: {e}")
                        #st.stop()
                else:
                    loader=UnstructuredURLLoader(urls=[generic_url],ssl_verify=False,
                                                 headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"}
                                                )

                    docs=loader.load()                
                

                ## Chain For Summarization
                
                chain=load_summarize_chain(llm,chain_type="stuff",prompt=prompt)
                output_summary=chain.run(docs)

                st.session_state.docs = docs
                st.session_state.summary = output_summary
        except Exception as e:
            st.exception(f"Exception: {e}")

if st.session_state.summary:
    st.subheader("📄 Summary:")
    st.success(st.session_state.summary)

                # st.success(output_summary)
                # st.divider()

if st.session_state.docs:
                # Create embeddings
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

                # Create vectorstore from documents
    vectorstore = FAISS.from_documents(st.session_state.docs, embedding_model)

                # Create retriever
    retriever = vectorstore.as_retriever()

                # Build the retrieval-based QA chain
    qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                #return_source_documents=True
                )

    user_query = st.text_input("Ask a question about the content")

    if user_query:
        response = qa_chain.run(user_query)
        st.info(response)


# except Exception as e:
#             st.exception(f"Exception:{e}")


