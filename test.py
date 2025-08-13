import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
import streamlit as st
import pandas as pd

# 知识库路径
KNOWLEDGE_BASE_DIR = "knowledgebase"
METADATA_FILE = Path(KNOWLEDGE_BASE_DIR) / "metadata.json"

# 感知哈希 (pHash)
def phash(image, hash_size=8):
    image = image.convert("L").resize((hash_size * 4, hash_size * 4), Image.LANCZOS)
    pixels = np.array(image, dtype=np.float32)
    dct = np.fft.fft2(pixels)
    dctlowfreq = dct[:hash_size, :hash_size]
    med = np.median(dctlowfreq)
    diff = dctlowfreq > med
    return diff.flatten()

def phash_similarity(img1, img2):
    h1 = phash(img1)
    h2 = phash(img2)
    matches = np.count_nonzero(h1 == h2)
    return matches / len(h1)  # 返回相似度 0~1

# 找到最佳匹配
def find_best_match(uploaded_image):
    kb_dir = Path(KNOWLEDGE_BASE_DIR)
    best_score = -1
    best_name = None

    for file in kb_dir.iterdir():
        if file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            try:
                kb_image = Image.open(file).convert("RGB")
                score = phash_similarity(uploaded_image, kb_image)
                if score > best_score:
                    best_score = score
                    best_name = file.stem
            except Exception as e:
                print(f"无法处理 {file}: {e}")

    return best_name, best_score

# 读取 metadata.json
def load_metadata():
    if METADATA_FILE.exists():
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

# 渲染治理方案
def display_treatment(treatment):
    if isinstance(treatment, (list, dict)):
        df = pd.DataFrame(treatment if isinstance(treatment, list) else [treatment])
        st.table(df)
    else:
        st.write(treatment)

# Streamlit 前端
st.title("图片对比与病虫害治理方案（pHash版）")
uploaded_file = st.file_uploader("上传图片", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption=f"上传图片: {uploaded_file.name}", use_container_width=True)

    match_name, score = find_best_match(image)
    metadata = load_metadata()

    if score >= 0.5:
        st.success(f"匹配结果: {match_name}（相似度: {score:.2%}）")
        if match_name in metadata:
            st.markdown("**治理方案:**")
            display_treatment(metadata[match_name].get("treatment", "无数据"))
    elif score >= 0.4:
        st.warning(f"可能匹配: {match_name}（相似度: {score:.2%}，请人工确认）")
        if match_name in metadata:
            st.markdown("**治理方案（需人工确认）:**")
            display_treatment(metadata[match_name].get("treatment", "无数据"))
    else:
        st.error("未找到匹配图片（相似度低于40%）")
