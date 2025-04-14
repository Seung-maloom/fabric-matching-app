
import streamlit as st
from PIL import Image
import os
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_resource
def load_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

model, processor = load_model()

st.title("🧵 바이어 이미지 기반 ERP 원단 추천기")

uploaded_buyer_image = st.file_uploader("바이어 이미지를 업로드하세요", type=["jpg", "jpeg", "png"])
erp_image_dir = st.text_input("ERP 원단 이미지 폴더 경로를 입력하세요 (기본값: ./erp_images)", value="./erp_images")

def get_image_embedding(img, processor, model):
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
    return emb.numpy()

def load_erp_images(folder):
    filenames, images, embeddings = [], [], []
    for file in os.listdir(folder):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            img = Image.open(os.path.join(folder, file)).convert("RGB")
            emb = get_image_embedding(img, processor, model)
            filenames.append(file)
            images.append(img)
            embeddings.append(emb)
    return filenames, images, np.vstack(embeddings)

if uploaded_buyer_image:
    buyer_img = Image.open(uploaded_buyer_image).convert("RGB")
    buyer_emb = get_image_embedding(buyer_img, processor, model)

    with st.spinner("ERP 원단 이미지 분석 중..."):
        filenames, images, embeddings = load_erp_images(erp_image_dir)

    sims = cosine_similarity(buyer_emb, embeddings)[0]
    top_idxs = sims.argsort()[-5:][::-1]

    st.subheader("🔍 유사한 ERP 원단 리스트")
    for idx in top_idxs:
        st.image(images[idx], caption=f"{filenames[idx]} (유사도: {sims[idx]:.4f})", use_column_width=True)
