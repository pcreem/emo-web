import gradio as gr
import whisper
from feature import AudioTextEmotionModel, extract_audio_features
import torch

# 設定設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 載入情緒辨識模型
emotion_model = AudioTextEmotionModel(audio_input_dim=180, text_input_dim=768, hidden_dim=128, output_dim=3)
emotion_model.load_state_dict(torch.load("model_weights.pth", map_location=device))
emotion_model.to(device)
emotion_model.eval()

# 載入 Whisper 模型進行語音轉文字
whisper_model = whisper.load_model("base")
EMOTION_LABELS = {0: '正面', 1: '中性', 2: '負面'}

def predict_emotion(audio_path):
    result = whisper_model.transcribe(audio_path, language="zh")
    text = result["text"]
    audio_feat = extract_audio_features(audio_path)
    audio_tensor = torch.tensor(audio_feat, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        output = emotion_model(audio_tensor, torch.zeros(1, 1, 768).to(device))  # dummy text input
        pred = torch.argmax(output, dim=1).item()

    return f"語音轉文字結果：{text}\n預測情緒：{EMOTION_LABELS[pred]}"

def create_interface():
    with gr.Blocks() as demo:
        gr.Markdown("### 🎧 中文語音情緒辨識（EATD）\n說一段話，我會判斷你的情緒（正面 / 中性 / 負面）")
        audio_input = gr.Audio(sources=["microphone"], type="filepath", label="請錄音")
        output = gr.Textbox()
        btn = gr.Button("分析")
        btn.click(fn=predict_emotion, inputs=audio_input, outputs=output)
    return demo

demo = create_interface()
demo.launch(share=True)

