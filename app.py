import streamlit as st
import pandas as pd
import os
import plotly
from dotenv import load_dotenv
import google.generativeai as genai 
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
# from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import time 
import plotly.io as pio
from PIL import Image
import io

model = genai.GenerativeModel('gemini-pro')
vision_model = genai.GenerativeModel('gemini-pro-vision')

def insights_generator(img):
    # Convert Plotly figure object to PNG image data
    # print("insights fn called")
    image_bytes = pio.to_image(img, format="png")

    # Convert image bytes to PIL Image object
    image = Image.open(io.BytesIO(image_bytes))
    
    response = vision_model.generate_content(["Generate Key insights from this Plotly plot without any follow back questions",image])
    # print(response.text)
    return response.text

# i want to have a list which holds the number of tokens after each conversation




load_dotenv()
genai.configure(api_key=os.getenv('PALM_API_KEY'))



def init_prompt(df,adv=False):
    

    columns = df.columns if not adv else required_columns
    # st.write(required_columns)
    # st.write(columns)
    prompt = "There is already a dataframe df stored in df variable. \n The df has columns "+ (", ".join(columns)) + ".\n"
    for i in columns:
        if len(df[i].drop_duplicates()) < 20 and df.dtypes[i]=="O":
            prompt = prompt + "\nThe column '" + i + "' has categorical values '" + \
            "','".join(str(x) for x in df[i].drop_duplicates()) + "'. \n"
        elif df.dtypes[i]=="int64" or df.dtypes[i]=="float64":
            prompt = prompt + "\nThe column '" + i + "' is type " + str(df.dtypes[i])+". \n"
    prompt = prompt + "\n\nGive python plotly code without any replies. Add title in the plot as well for following query : \n"
    init_prompt_text = prompt
    return init_prompt_text


def init_conversation_chain():
    # Define your conversational AI model
    llm = ChatGoogleGenerativeAI(google_api_key=GOOGLE_API_KEY,model="gemini-pro", temperature=0.1, convert_system_message_to_human=True)
    
    # Define the prompt template
    PROMPT_TEMPLATE = """The following is a conversation between AI and human. If the AI does not know the answer to a question, it truthfully says it does not know.

                        Current conversation:
                        {history}
                        Human: {input}
                        AI Assistant:
                        """
    PROMPT = PromptTemplate(input_variables=["history", "input"], template=PROMPT_TEMPLATE)

    # Initialize the conversation chain with memory
    conversation = ConversationChain(
        llm=llm,
        verbose=True,
        prompt=PROMPT,
        memory=ConversationBufferWindowMemory(k=3)
    )
    
    return conversation

def clean_data(df):
    df=df.drop_duplicates()
    # clean dataset with imputation
    for i in df.columns:
        if df.dtypes[i]=="int64" or df.dtypes[i]=="float64":
            df[i]=df[i].fillna(df[i].mean())
        elif df.dtypes[i]=="O":
            df[i]=df[i].fillna(df[i].mode()[0])
    return df



def get_session_state_key():
    return f"messages_{st.session_state.current_csv}"

def get_session_history():
    return "conversation_chain"




st.set_page_config(page_title="Visistant", page_icon=":bar_chart:", layout="wide")
st.header("Visistant")
with st.sidebar:
    GOOGLE_API_KEY= st.text_input("Enter Google API key", type="password")
    if GOOGLE_API_KEY:
        if get_session_history() not in st.session_state:
            st.session_state[get_session_history()] = init_conversation_chain()
        conversation = st.session_state[get_session_history()]
    st.title("CSV File Uploader")
    uploaded_files = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)
    uploaded_files=uploaded_files[::-1]
    input_csv = st.selectbox("Select a file", options=uploaded_files, format_func=lambda x: x.name if x else "None")


    need_insights=st.toggle('Need Insights?')
        
    modes=['Default','Advanced']
    mode=st.radio("Mode",modes)

    
    
    
    if input_csv is not None:
        df = pd.read_csv(input_csv)
        df=clean_data(df)    
        if mode=='Advanced':

            required_columns = st.multiselect(
            'Select columns',
            df.columns) 
        st.dataframe(df)
        
    
if input_csv is not None and GOOGLE_API_KEY is not None:
    csv_hash = input_csv.file_id
    if "current_csv" not in st.session_state:
        st.session_state.current_csv = csv_hash
        # init_prompt_text = init_prompt(df) if mode=='Default' else init_prompt(df,True)
    st.session_state.current_csv = csv_hash 
    
    if not hasattr(st.session_state, get_session_state_key()):
        setattr(st.session_state, get_session_state_key(), {"messages": [], "plots": [],"insights":[]})
    init_prompt_text = init_prompt(df) if mode=='Default' else init_prompt(df,True)
    # st.write(init_prompt_text)



    # Display chat messages from history on app rerun
    messages = getattr(st.session_state, get_session_state_key())["messages"]
    plots = getattr(st.session_state, get_session_state_key())["plots"]
    insights = getattr(st.session_state, get_session_state_key())["insights"]

    #rerun
    for message, plot,insights in zip(messages, plots,insights):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
        with st.chat_message("assistant"):
            st.plotly_chart(plot)
            if len(insights)>0:
                st.markdown(insights)

def llm_pass(prompt):
    start_time = time.time()
    mem=conversation.memory.buffer
    if init_prompt_text not in mem:
        prompt =init_prompt_text + prompt
    else:
        prompt = prompt
    history = getattr(st.session_state, get_session_history(), [])
    # print("\n Combined prompt: ",prompt)
    res=conversation.predict(history=history,input=prompt)
    
    # print("RESPONSE\n",res)
    start_index = res.find("```python") + len("```python")
    end_index = res.rfind("```")
    python_code = res[start_index:end_index].strip()
    python_code = python_code.replace("fig.show()", "")
    # print("\npython_code: ",python_code)
    exec(python_code)
    end_time = time.time()  # Record the end time after receiving the response
    response_time = end_time - start_time
    
    print("\nTokens used \n",model.count_tokens(conversation.memory.buffer))
    print("\nResponse Time: ",response_time)

    # Display the generated plot
    plotly_fig = locals()['fig']
    
    # Storing the User Message and the generated plot
    getattr(st.session_state, get_session_state_key())["messages"].append(
        {
            "role": "user",
            "content": query
        }
    )

    getattr(st.session_state, get_session_state_key())["plots"].append(plotly_fig)
    if need_insights:
        insights_msg=insights_generator(plotly_fig)
        # print("insights_msg: ",insights_msg)
        getattr(st.session_state, get_session_state_key())["insights"].append(insights_msg)
    else:
        insights_msg="" 
        getattr(st.session_state, get_session_state_key())["insights"].append("")

    # Displaying the Assistant Message with the generated plot
    with st.chat_message("assistant"):
        st.plotly_chart(plotly_fig)
        # create a button to reveal code
        if len(insights_msg)>0:
            st.markdown(insights_msg)
        st.code(python_code, language='python')
 
query = st.chat_input("Ask your question?")

# Calling the Function when Input is Provided
if query:
    # Displaying the User Message
    with st.chat_message("user"):
        st.markdown(query)
    llm_pass(query)
    # print("memory",conversation.memory.load_memory_variables(inputs=[])['history'])

