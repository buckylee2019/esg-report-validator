import streamlit as st
import numpy as np
import json
from utils.esg_chain import GenerateEsgChain,framework,get_collection_list, vectorDB,TranslateChain,Generate
import re


st.set_page_config(page_title="ESG Report Checker", page_icon="💡")
st.title("ESG 報告檢核項目列表")
items = framework()

collection = st.sidebar.selectbox('ESG 報告',set(get_collection_list()))


generate = st.button("AI 自動生成")
st.write("Click to generate!")

if generate:

    for key in items:
        st.markdown(f"## {key}")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 項目")
            
        with col2:
            st.markdown("### 檢核結果")
                        
        with col3:
            st.markdown("### 說明")
        st.divider()
        for it in items[key]:
            vector_esg = vectorDB(collection)
            res = GenerateEsgChain(user_prompt=it,vector_instance=vector_esg.vectorstore())
            try:
                res_json = json.loads(res)
            except:
                res = res.replace('"\n','",\n').replace('",\n}','\n}')
                prompt = "[INST] <<SYS>>\n"\
                "You are a helpful, respectful and honest assistant.\n"\
                "Always answer as helpfully as possible, while being safe.\n"\
                "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.\n"\
                "Please ensure that your responses are socially unbiased and positive in nature.\n"\
                "\n"\
                "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct.\n"\
                "If you don't know the answer to a question, please don't share false information.\n"\
                "\n"\
                "Fix the invalid JSON format:\n"\
                "<</SYS>>\n"\
                f"Invalid JSON:{res}\n"\
                "This is the valid JSON look like:\n"\
                "{\n"\
                '"Question":"Placeholder for Question",'\
                '"Explanation":"Placeholder for Explanation",'\
                '"Answer":"Yes" or "No" or "Uncertain"'\
                "}\n"\
                'Please start the response with {"Question":"...\n'\
                "AVOID new lines in JSON format!!!\n"\
                "Valid JSON: [/INST]"
                res_fix = Generate(prompt,stop_sequences=["}"])

                explanation_pt = r'"Explanation":[ |](.*?)\n'
                answer_pt = r'"Answer":[ |](.*?)\n'
                # Use the findall method to extract the matched groups
                expl = re.search(explanation_pt, res_fix).group(1)
                answer = re.search(answer_pt, res_fix).group(1)
                
                try:
                    res_json = json.loads(res_fix)
                except:
                    res_json = {"Question":it, "Explanation":expl,"Answer":answer}
            with st.container():
                    
                col1, col2, col3 = st.columns(3)
     
                with col1:
                    
                    st.markdown(it)
                with col2:
                    
                    if "yes" in str(res_json['Answer']).lower() or "是" in str(res_json['Answer']).lower() or res_json['Answer']==True:
                        multi = ("- [X] 是  \n- [ ] 否  \n- [ ] 待確認  ")
                    elif "no" in str(res_json['Answer']).lower() or "否" in str(res_json['Answer']).lower() or res_json['Answer']==False:
                        multi = ("- [ ] 是  \n- [X] 否  \n- [ ] 待確認  ")
                    else:
                        multi = ("- [ ] 是  \n- [ ] 否  \n- [X] 待確認  ")
                    st.markdown(multi)
                                
                with col3:
                    
                    st.markdown(TranslateChain(res_json["Explanation"]))
                with st.expander("查看參考來源"):
                    docs = vector_esg.vectorstore().similarity_search(it,k=3)
                    source_document = "\n\n".join([f"Document {idx+1}. \n{r.page_content}" for idx, r in enumerate(docs)])
                    st.markdown(f"""
                        {source_document}
                        """)


