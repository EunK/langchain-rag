import streamlit as st
import google.generativeai as genai
from supabase import create_client, Client


@st.cache_resource
def get_gemini_client(api_key: str, model: str) -> genai.GenerativeModel:
    """
    Gemini API를 설정하고 초기화합니다.
    """
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model)

@st.cache_resource
def get_supabase_client(url: str, key: str) -> Client:
    """
    Supabase 클라이언트는 그대로 유지됩니다.
    """
    return create_client(url, key)
