"""Streamlit frontend for the E-Commerce Semantic Search API."""

import os

import requests
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="E-Commerce Semantic Search", layout="wide")
st.title("E-Commerce Semantic Search")

text_tab, image_tab = st.tabs(["Text Search", "Image Search"])

with text_tab:
    query = st.text_input("Search the catalog", key="text_query")
    mode = st.selectbox(
        "Mode",
        ["hybrid", "hybrid-rerank", "dense", "bm25"],
        index=0,
        help="hybrid-rerank is slower (cross-encoder reranking) but can be more precise.",
    )
    top_k = st.number_input("Results", min_value=1, max_value=50, value=10, key="text_top_k")

    if st.button("Search", key="text_search_button") and query:
        spinner_text = (
            "Searching (reranking, this takes a few seconds)..."
            if mode == "hybrid-rerank"
            else "Searching..."
        )
        with st.spinner(spinner_text):
            try:
                response = requests.get(
                    f"{API_BASE_URL}/search/text",
                    params={"q": query, "mode": mode, "top_k": top_k},
                    timeout=30,
                )
                response.raise_for_status()
            except requests.RequestException as e:
                st.error(f"Could not reach the search API: {e}")
            else:
                results = response.json()["results"]
                if not results:
                    st.info("No results found.")
                else:
                    st.table(
                        [
                            {
                                "Rank": i + 1,
                                "Name": r["name"],
                                "Brand": r["brand"],
                                "Category": r["category_path"],
                                "Score": round(r["score"], 4),
                            }
                            for i, r in enumerate(results)
                        ]
                    )

with image_tab:
    image_query = st.text_input("Search product images", key="image_query")
    image_top_k = st.number_input("Results", min_value=1, max_value=50, value=10, key="image_top_k")

    if st.button("Search", key="image_search_button") and image_query:
        with st.spinner("Searching..."):
            try:
                response = requests.get(
                    f"{API_BASE_URL}/search/image",
                    params={"q": image_query, "top_k": image_top_k},
                    timeout=30,
                )
                response.raise_for_status()
            except requests.RequestException as e:
                st.error(f"Could not reach the search API: {e}")
            else:
                results = response.json()["results"]
                if not results:
                    st.info("No results found.")
                else:
                    columns = st.columns(5)
                    for i, r in enumerate(results):
                        with columns[i % 5]:
                            st.image(
                                f"{API_BASE_URL}{r['image_url']}",
                                caption=f"{r['display_name']} ({r['category']}) — {r['score']:.4f}",
                            )
