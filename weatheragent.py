import os
import asyncio
import requests
import streamlit as st
from autogen_agentchat.agents import AssistantAgent 
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Streamlit UI Setup
st.set_page_config(page_title="Weather Agent", page_icon="🌤️")
st.title("🌤️ AutoGen Weather Agent")

# Sidebar for Keys
with st.sidebar:
    st.header("🔑 API Credentials")
    openai_key = st.text_input("OpenAI API Key", type="password")
    weather_key = st.text_input("OpenWeatherMap API Key", type="password")

if not openai_key or not weather_key:
    st.warning("👈 Please enter both API keys in the sidebar to start!")
    st.stop()

# Weather Tool
def get_weather(city: str) -> str:
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": city,
            "appid": weather_key,
            "units": "metric"
        }
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

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about the weather in any city..."):
    # Show User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Show Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Agent is working..."):
            try:
                # Initialize AutoGen Model & Agent
                model_client = OpenAIChatCompletionClient(
                    model="gpt-4o-mini",
                    api_key=openai_key
                )
                
                agent = AssistantAgent(
                    name="weather_agent",
                    model_client=model_client,
                    tools=[get_weather],
                    system_message="You are a helpful AI Weather Assistant. Use the 'get_weather' tool to find realtime weather information.",
                    reflect_on_tool_use=True
                )
                
                # Async function to run AutoGen in Streamlit
                async def run_agent():
                    response = await agent.run(task=prompt)
                    return response.messages[-1].content
                
                # Execute and display
                final_reply = asyncio.run(run_agent())
                st.markdown(final_reply)
                
                # Save to history
                st.session_state.messages.append({"role": "assistant", "content": final_reply})
                
            except Exception as e:
                st.error(f"Error: {e}")
