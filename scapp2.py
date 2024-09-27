import streamlit as st
import xml.etree.ElementTree as ET
import requests
from langchain.schema import Document  # Importing Document class
from langchain_community.vectorstores import FAISS
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableMap

# Your API key and URL (updated to request XML)
key = '6448425a457377653630797a536a4c'
url = 'http://openapi.seoul.go.kr:8088/6448425a457377653630797a536a4c/xml/parkingKickboard/1/1000/'

# Fetch data from the API
response = requests.get(url)
content = response.content.decode('utf-8')

# Parse the XML data
tree = ET.ElementTree(ET.fromstring(content))
root = tree.getroot()

# Function to filter and get scooter parking data
def scoot_parklot_data(root, region):
    items = root.findall(".//row")
    scooter_info = []
    for item in items:
        sgg_nm = item.find("SGG_NM").text if item.find("SGG_NM") is not None else ''
        # Filter based on the user's input (region)
        if region in sgg_nm:
            info = {
                '순번': item.find("SN").text if item.find("SN") is not None else '',
                '시군구명': sgg_nm,
                '주소': item.find("PSTN").text if item.find("PSTN") is not None else '',
                '상세위치': item.find("DTL_PSTN").text if item.find("DTL_PSTN") is not None else '',
                '거치대 유무': item.find("STAND_YN").text if item.find("STAND_YN") is not None else '',
                '거치대 크기': item.find("STAND_SIZE").text if item.find("STAND_SIZE") is not None else '',
            }
            scooter_info.append(info)
    return scooter_info




def main():
    #st.title("서울시 구별 전동킥보드 주차구역")
    st.markdown(
    """
    <h1 style='text-align: center;'>서울시 구별 전동킥보드 주차구역</h1>
    """,
    unsafe_allow_html=True
)
    # User input for region selection
    #quest = st.text_input("희망하는 주차구역 '구'를 입력해주세요")
    st.markdown("<h4 style='text-align: center;'>🛴 희망하는 주차구역 '구'를 입력해주세요 🛴</h4>", unsafe_allow_html=True)
    quest = st.text_input("", "")
    
    # If button is clicked, process the data
    if st.button("검색"):
        # Get the filtered scooter parking data based on user input
        scooter_info = scoot_parklot_data(root, quest)
        
        if scooter_info:
            # Convert the filtered data into documents for vector search
            documents = [
                Document(page_content=", ".join([f"{key}: {str(info[key])}" for key in ['순번', '시군구명', '주소', '상세위치', '거치대 유무', '거치대 크기']]))

                for info in scooter_info
            ]
            
            # Embed the documents
            embedding_function = SentenceTransformerEmbeddings(model_name="jhgan/ko-sroberta-multitask")
            db = FAISS.from_documents(documents, embedding_function)
            retriever = db.as_retriever(search_type="similarity", search_kwargs={'k': 1000, 'fetch_k': 1000})

            # Set up the prompt template for the chatbot
            template = """
            너는 서울시 전동킥보드 주차구역 안내로봇이야.
            입력받은 위치에 모든 킥보드 주차구역의
            상세위치를 알려주면서 가장 가까운 지하철역까지 안내해줘야해.
            사용자에게 가능한 많은 정보를 친절하게 제공해줘야해. 반드시 한국말로 답해줘.
            
            Answer the question as based only on the following context:
            {context}

            Question: {question}
            """
            llm = ChatOllama(model="gemma2:9b", temperature=0, base_url="http://127.0.0.1:11434/") # Modify the LLM base URL

            # Prepare the chain for question and answer
            prompt = ChatPromptTemplate.from_template(template)
            chain = RunnableMap({
                "context": lambda x: retriever.get_relevant_documents(x['question']),
                "question": lambda x: x['question']
            }) | prompt | llm

            # Get the chatbot's response
            content = chain.invoke({'question': quest}).content
            
            # Display the chatbot's response in Streamlit
            #st.write(content)
            st.markdown(f'<div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px;">{content}</div>', unsafe_allow_html=True)
        else:
            #st.write("해당 구에 대한 주차 정보가 없습니다.")
            st.markdown('<div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px;">해당 구에 대한 주차 정보가 없습니다.</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
