# backend/app.py
import os
import tempfile
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import logging

# backend内のモジュールをインポート
from gemini import chat_with_ai, generate_diary_from_conversation
from notion import save_to_notion

load_dotenv()

app = Flask(__name__)

# CORS設定を環境変数から読み込む
CORS_ORIGIN = os.getenv("CORS_ORIGIN", "http://localhost:8000")
CORS(app, resources={r"/api/*": {"origins": CORS_ORIGIN}})

# ロギング設定
logging.basicConfig(level=logging.INFO)

@app.post("/api/chat")
def chat_endpoint():
    """音声またはテキストと会話履歴を受け取り、AIの応答を返す"""
    app.logger.info("Received request for /api/chat")
    
    if 'conversation' not in request.form:
        app.logger.error("Conversation history part missing in chat request")
        return jsonify(error="会話履歴が見つかりません。"), 400

    conversation_json = request.form["conversation"]
    user_text = request.form.get("text")
    audio_blob = request.files.get("audio")

    if not audio_blob and not user_text:
        return jsonify(error="音声またはテキストデータが必要です。"), 400

    try:
        conversation_history = json.loads(conversation_json)
        if not isinstance(conversation_history, list):
            raise ValueError("Conversation history is not a list.")
    except (json.JSONDecodeError, ValueError) as e:
        app.logger.error(f"Failed to parse conversation history: {e}")
        return jsonify(error="会話履歴の形式が正しくありません。"), 400

    try:
        if audio_blob:
            with tempfile.NamedTemporaryFile(delete=True, suffix=".webm") as tmp:
                audio_blob.save(tmp.name)
                ai_reply = chat_with_ai(conversation_history, user_audio_path=tmp.name, user_text=user_text)
        else:
            ai_reply = chat_with_ai(conversation_history, user_text=user_text)
        
        return jsonify(reply=ai_reply)

    except Exception as e:
        app.logger.error(f"Error processing chat request: {e}", exc_info=True)
        return jsonify(error="AIの応答生成中にエラーが発生しました。"), 500

@app.post("/api/generate_diary")
def generate_diary_endpoint():
    """会話履歴を受け取り、日記を生成してNotionに保存し、日記内容を返す"""
    app.logger.info("Received request for /api/generate_diary")
    data = request.get_json()
    if not data or "conversation" not in data:
        app.logger.error("Conversation history missing in generate_diary request")
        return jsonify(error="会話履歴が必要です。"), 400

    conversation_history = data["conversation"]
    if not isinstance(conversation_history, list):
        app.logger.error("Invalid conversation history format (not a list)")
        return jsonify(error="会話履歴の形式がリストではありません。"), 400

    try:
        diary_content = generate_diary_from_conversation(conversation_history)
        
        notion_url = ""
        try:
            notion_url = save_to_notion(diary_content)
        except Exception as e_notion:
            app.logger.warning(f"Failed to save diary to Notion: {e_notion}", exc_info=True)
            diary_content += "\n\n---\n*警告: Notionへの保存に失敗しました。*"

        return jsonify(summary=diary_content, notion_url=notion_url)

    except Exception as e:
        app.logger.error(f"Error processing generate_diary request: {e}", exc_info=True)
        return jsonify(error="日記の生成中にエラーが発生しました。"), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5500))
    app.run(host="0.0.0.0", port=port, debug=False)
