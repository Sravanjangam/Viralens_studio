import gradio as gr
import requests
import tempfile
from PIL import Image

BACKEND_URL = "http://localhost:8000/full/full/"
CHAT_URL = "http://localhost:8000/chatbot/chat/"

# -------------------------
# Backend call
# -------------------------
def analyze_image(image, caption):
    if image is None:
        return {}, "Please upload an image."

    pil_img = Image.fromarray(image.astype("uint8"))
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        pil_img.save(tmp.name)
        img_path = tmp.name

    files = {"file": open(img_path, "rb")}
    data = {"caption": caption or ""}

    try:
        res = requests.post(BACKEND_URL, files=files, data=data, timeout=180)
    except Exception as e:
        return {}, f"Backend connection error: {e}"

    if res.status_code != 200:
        return {}, f"Backend error: {res.text}"

    return res.json(), "✅ Analysis complete."

# -------------------------
# Chat helper
# -------------------------
def make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    elif hasattr(obj, "item"):
        return obj.item()
    else:
        return obj

def chat_with_ai(user_message, history, analysis_data):
    if not analysis_data:
        history.append((user_message, "Please analyze an image first."))
        return history

    payload = {
        "message": user_message,
        "metrics": make_json_safe(analysis_data),
    }

    try:
        res = requests.post(CHAT_URL, json=payload, timeout=30)
        reply = res.json().get("reply", "No response.") if res.status_code == 200 else res.text
    except Exception as e:
        reply = str(e)

    history.append((user_message, reply))
    return history

# -------------------------
# Helpers
# -------------------------
def fmt_score(x):
    try:
        x = float(x)
    except:
        return "—"
    return f"{round(x * 100, 1)}" if x <= 1 else f"{round(x, 1)}"

def format_kv(d):
    if not isinstance(d, dict):
        return "_No data_"
    return "\n".join([f"- **{k}**: {round(v,3) if isinstance(v,float) else v}" for k,v in d.items()])

# -------------------------
# UI
# -------------------------
with gr.Blocks(title="ViraLens Studio") as app:

    gr.Markdown("# 🔍 **ViraLens — AI Visual Intelligence**")
    gr.Markdown("Upload an image and caption to predict virality and get AI-powered insights.")

    state = gr.State({})

    # ---------- TOP 50% ----------
    with gr.Row(equal_height=True):
        with gr.Column(scale=5):
            image_in = gr.Image(
                type="numpy",
                label="Upload Image",
                height=420
            )

        with gr.Column(scale=5):
            caption_in = gr.Textbox(
                label="Enter Caption",
                lines=8,
                placeholder="Write your caption here..."
            )

    analyze_btn = gr.Button("Analyze Image", size="lg")
    status = gr.Markdown()

    # ---------- SUMMARY ----------
    gr.Markdown("## 🚀 Virality Overview")

    with gr.Row():
        geom_card = gr.Markdown("**Geometry:** —")
        color_card = gr.Markdown("**Color & Lighting:** —")
        aest_card = gr.Markdown("**Aesthetic:** —")

    with gr.Row():
        cap_card = gr.Markdown("**Caption:** —")
        trend_card = gr.Markdown("**Trend:** —")
        viral_card = gr.Markdown("**Virality:** —")

    # ---------- DETAILS ----------
    gr.Markdown("## 🔍 Detailed Analysis")

    with gr.Accordion("Geometry Details", open=False):
        geom_details = gr.Markdown()

    with gr.Accordion("Color & Lighting Details", open=False):
        color_details = gr.Markdown()

    with gr.Accordion("Caption Analysis", open=False):
        caption_details = gr.Markdown()

    with gr.Accordion("Virality Breakdown", open=False):
        virality_details = gr.Markdown()

    # ---------- CHAT ----------
    # gr.Markdown("## 💬 AI Insight Assistant")

    # chatbot = gr.Chatbot(height=260)
    # chat_input = gr.Textbox(placeholder="Ask why this image performs the way it does…")
    # chat_btn = gr.Button("Send")

    # ---------- ACTIONS ----------
    def run_all(image, caption):
        data, msg = analyze_image(image, caption)
        viral = data.get("virality", {}) if data else {}

        return (
            data,
            msg,
            f"**Geometry:** {fmt_score(viral.get('geometry'))}",
            f"**Color:** {fmt_score(viral.get('color_light'))}",
            f"**Aesthetic:** {fmt_score(viral.get('aesthetic'))}",
            f"**Caption:** {fmt_score(viral.get('caption'))}",
            f"**Trend:** {fmt_score(viral.get('trend'))}",
            f"**Virality:** {fmt_score(viral.get('final_score'))}",
            format_kv(data.get("geometry", {})),
            format_kv(data.get("color", {})),
            format_kv(data.get("caption_analysis", {})),
            format_kv(data.get("virality", {})),
        )

    analyze_btn.click(
        run_all,
        inputs=[image_in, caption_in],
        outputs=[
            state,
            status,
            geom_card,
            color_card,
            aest_card,
            cap_card,
            trend_card,
            viral_card,
            geom_details,
            color_details,
            caption_details,
            virality_details,
        ],
    )

    # chat_btn.click(
    #     chat_with_ai,
    #     inputs=[chat_input, chatbot, state],
    #     outputs=chatbot,
    # )

app.launch()
