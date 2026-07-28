import os
import requests
import streamlit as st
from langchain.agents import AgentType, initialize_agent
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# Streamlit UI Setup
st.set_page_config(page_title="AI Weather Agent", page_icon="🌤️")
st.title("🌤️ AI Weather Agent")

# Sidebar for API Keys
with st.sidebar:
    st.header("🔑 API Credentials")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    weather_api_key = st.text_input("OpenWeatherMap API Key", type="password")

if not openai_api_key or not weather_api_key:
    st.warning("👈 Please enter both API keys in the sidebar to start!")
    st.stop()

# Weather Tool Definition
@tool
def get_current_weather(city: str) -> str:
    """Fetches current weather for a specified city using OpenWeatherMap API."""
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": weather_api_key,
        "units": "metric"
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
        if response.status_code != 200 or "weather" not in data:
            return f"Could not fetch weather for {city}."
        
        desc = data["weather"][0]["description"].capitalize()
        temp = data["main"]["temp"]
        location = data["name"]
        return f"Weather in {location}: {desc}, {temp}°C"
    except Exception as e:
        return f"Error fetching weather: {str(e)}"

# Initialize Agent
@st.cache_resource
def load_agent(api_key):
    llm = ChatOpenAI(temperature=0, openai_api_key=api_key, model_name="gpt-4o-mini")
    tools = [get_current_weather]
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True
    )

agent_executor = load_agent(openai_api_key)

# Chat Interface Logic
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about the weather in any city..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Checking weather..."):
            try:
                response = agent_executor.run(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error: {e}")
