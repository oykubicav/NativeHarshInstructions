import os, cv2, numpy as np, streamlit as st
from PIL import Image, ImageOps, ImageDraw
from ultralytics import YOLO

from process_image import preprocess_image
from run_yolo import run_yolo
from yolo_to_json import get_yolo_to_json
from json_to_txt import json_to_txt
from create_embedding import llm, embedding_model
from prompt import prompt_template

from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# ----------------------- Globals -----------------------
model = YOLO("best.pt")  # YOLO weights
CLASS_NAMES = model.names

st.set_page_config(page_title="Star Detector", layout="centered")
st.title("🔭 Star Detection with YOLO")

# -------------------- Session State --------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "last_summary" not in st.session_state:
    st.session_state.last_summary = ""
if "last_image" not in st.session_state:
    st.session_state.last_image = None

# ----------------- Helper: draw boxes ------------------

def draw_boxes_on_image(image_np: np.ndarray, results):
    """Overlay YOLO detections on original image."""
    img_pil = Image.fromarray(image_np)
    draw = ImageDraw.Draw(img_pil)

    boxes = results.boxes.xywh.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy().astype(int)

    for (x, y, w, h), c, conf in zip(boxes, classes, confs):
        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)
        label = f"{CLASS_NAMES[c]} {conf:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline="cyan", width=2)
        draw.text((x1, y1 - 10), label, fill="cyan")

    return img_pil

# ---------------------- Upload UI ----------------------
upload_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if upload_file:
    # Display original image
    orig_img = Image.open(upload_file).convert("RGB")
    orig_img = ImageOps.exif_transpose(orig_img).resize((640, 640))
    st.image(orig_img, caption="Uploaded Image", use_column_width=True)

    # Save original
    os.makedirs("uploads", exist_ok=True)
    orig_path = os.path.join("uploads", upload_file.name)
    orig_img.save(orig_path)

    # Pre‑process for YOLO
    pre_np = preprocess_image(np.array(orig_img))
    pre_img = Image.fromarray(pre_np)
    pre_path = orig_path  # overwrite
    pre_img.save(pre_path)
    st.image(pre_img, caption="Processed Image", use_column_width=True)

    if st.button("🔍 Detect Stars"):
        # ------- YOLO inference on processed image -------
        results = run_yolo(pre_path)
        json_result = get_yolo_to_json(results)
        summary_txt = json_to_txt(json_result)
        st.session_state.last_summary = summary_txt  # persist

        # ------- Overlay on original --------
        labeled = draw_boxes_on_image(np.array(orig_img), results)
        st.session_state.last_image = labeled
        st.image(labeled, caption="YOLO Results on Original Image", use_column_width=True)

        # ------- Vector store & retriever -------
        doc = Document(page_content=summary_txt, metadata={"image": upload_file.name})
        if st.session_state.vectorstore is None:
            st.session_state.vectorstore = FAISS.from_documents([doc], embedding_model)
        else:
            st.session_state.vectorstore.add_documents([doc])
        st.session_state.retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})

        st.success("✅ Detection completed")

# -------------- Always show last summary ---------------
if st.session_state.last_summary:
    st.text_area("Detection Summary", st.session_state.last_summary, height=200)
if st.session_state.last_image:
    st.image(st.session_state.last_image, caption="YOLO Results on Original Image", use_column_width=True)



# ----------------------- Q&A ---------------------------
query = st.text_input("Ask a question about the image")
if query:
    if st.session_state.retriever is None:
        st.warning("⚠️ Please upload and analyze an image first.")
    else:
        docs = st.session_state.retriever.get_relevant_documents(query)
        context = ("\n".join(d.page_content for d in docs)
                   if docs else st.session_state.last_summary)
        prompt = prompt_template.format(context=context, question=query)
        answer = llm.invoke(prompt)
        st.markdown("### 📌 Answer")
        st.markdown(answer)
