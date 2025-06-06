import gradio as gr
import whisper
import torch
import numpy as np
from feature import (
    AudioTextEmotionModel,
    extract_audio_features,
    extract_text_features
)

# 設定設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 載入模型
emotion_model = AudioTextEmotionModel(audio_input_dim=180, text_input_dim=768, hidden_dim=128, output_dim=3)
emotion_model.load_state_dict(torch.load("model_weights.pth", map_location=device))
emotion_model.to(device)
emotion_model.eval()

# Whisper 模型
whisper_model = whisper.load_model("base")
EMOTION_LABELS = {0: '正面', 1: '中性', 2: '負面'}

# 情緒預測主函式（支援語音 / 文字 / 雙模）
def analyze_input(audio, text_input):
    audio_feat = None
    text_feat = None
    result_text = ""
    
    # 若有語音輸入
    if audio:
        result = whisper_model.transcribe(audio, language="zh")
        transcribed_text = result["text"]
        result_text += f"🎧 語音轉文字：「{transcribed_text}」\n"
        audio_feat = extract_audio_features(audio)
    else:
        transcribed_text = None

    # 若有文字輸入（用戶輸入或語音轉出）
    text = text_input or transcribed_text
    if text:
        text_feat = extract_text_features(text)
        result_text += f"✏️ 文字內容：「{text}」\n"
    
    if audio_feat is None and text_feat is None:
        return "請提供語音或文字輸入進行情緒辨識。"

    # 製作 tensor 輸入
    audio_tensor = (
        torch.tensor(audio_feat, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        if audio_feat is not None else
        torch.zeros(1, 1, 180).to(device)
    )
    text_tensor = (
        torch.tensor(text_feat, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        if text_feat is not None else
        torch.zeros(1, 1, 768).to(device)
    )

    with torch.no_grad():
        output = emotion_model(audio_tensor, text_tensor)
        pred = torch.argmax(output, dim=1).item()

    result_text += f"📊 預測情緒：{EMOTION_LABELS[pred]}"
    return result_text

# Gradio Chat UI
with gr.Blocks() as demo:
    gr.Markdown("## 🎧 中文語音情緒辨識聊天機器人\n支援語音輸入、文字輸入，或兩者結合分析")

    chatbot = gr.Chatbot()
    with gr.Row():
        audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="語音")
        text_input = gr.Textbox(lines=2, placeholder="輸入文字內容...", label="文字")
    send_btn = gr.Button("送出分析")

    def chat_handler(audio, text, history):
        response = analyze_input(audio, text)
        history = history or []
        history.append(("👤", response))
        return history, None, ""

    send_btn.click(fn=chat_handler,
                   inputs=[audio_input, text_input, chatbot],
                   outputs=[chatbot, audio_input, text_input])

demo.launch(share=True)
