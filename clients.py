import streamlit as st
from openai import OpenAI
from supabase import create_client, Client


@st.cache_resource
def get_openai_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


@st.cache_resource
def get_supabase_client(url: str, key: str) -> Client:
    return create_client(url, key)
