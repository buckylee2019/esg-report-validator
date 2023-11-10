import streamlit as st
import numpy as np
from esg_chain_wx import generate_standard_chain,generate_esg_chain,framework,get_collection_list, vectorDB,translate_chain


st.title("ESG 報告建議")
chapter = st.selectbox("關於GRI, 你想查詢哪個章節的範本",set(framework().keys()))

collection = st.sidebar.selectbox('參考文件',set(get_collection_list()))


generate = st.button("生成建議")
st.write("Click to generate!")

if generate:
    with st.container():
        st.header("ESG GRI "+chapter)
        vector_gri = vectorDB("GRI")
        qa_chain = generate_standard_chain(vector_gri.vectorstore())

        vector_esg = vectorDB(collection)

    # You can call any Streamlit command, including custom components:
        st.markdown( translate_chain(generate_esg_chain(user_prompt=framework()[chapter],qa_chain=qa_chain,vector_instance=vector_esg.vectorstore())))

    # with st.container():
    #     st.header("ESG GRI 2-1-b 的指引")
    #     st.markdown(generate_template("所有權的性質與法律形式意指屬於公有或私有，屬於法人實體、合夥企業、獨資企業或其他類型的實體(例如:非營利組織、協會或慈善機構)。"))

   # You can call any Streamlit command, including custom components:
    # col1, col2, col3 = st.columns(3)

    # with col1:
    #     st.header("A cat")
    #     st.image("https://static.streamlit.io/examples/cat.jpg")

    # with col2:
    #     st.header("A dog")
    #     st.image("https://static.streamlit.io/examples/dog.jpg")

    # with col3:
    #     st.header("An owl")
    #     st.image("https://static.streamlit.io/examples/owl.jpg")
