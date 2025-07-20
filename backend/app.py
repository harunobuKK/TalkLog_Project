# backend/app.py
import os
import tempfile
import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
import logging
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

# backend内のモジュールをインポート
from gemini import chat_with_ai, generate_diary_from_conversation, transcribe_audio # transcribe_audio を追加
from notion import save_to_notion

load_dotenv()

app = Flask(__name__, static_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'frontend'), static_url_path='')

# --- Database Configuration --- #
db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'talklog.db')
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
# --- End Database Configuration --- #

# CORS設定を環境変数から読み込む
CORS_ORIGIN = os.getenv("CORS_ORIGIN", "http://localhost:8000")
CORS(app, resources={r"/api/*": {"origins": CORS_ORIGIN}})

# ロギング設定
logging.basicConfig(level=logging.INFO)

# --- Database Model --- #
class Diary(db.Model):
    """日記エントリを保存するモデル"""
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    diary_content = db.Column(db.Text, nullable=False)
    sentiment_score = db.Column(db.Float, nullable=True)
    highlight_events = db.Column(db.String(255), nullable=True)

    def to_dict(self):
        """モデルオブジェクトを辞書に変換"""
        return {
            'id': self.id,
            'date': self.date.isoformat(),
            'diary_content': self.diary_content,
            'sentiment_score': self.sentiment_score,
            'highlight_events': self.highlight_events
        }
# --- End Database Model --- #

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/dashboard.html')
def serve_dashboard():
    return send_from_directory(app.static_folder, 'dashboard.html')

@app.post("/api/chat")
def chat_endpoint():
    """音声またはテキストと会話履歴を受け取り、AIの応答を返す"""
    app.logger.info("Received request for /api/chat")
    
    if 'conversation' not in request.form:
        app.logger.error("Conversation history part missing in chat request")
        return jsonify(error="会話履歴が見つかりません。"), 400

    conversation_json = request.form["conversation"]
    user_text_input = request.form.get("text") # 元のテキスト入力
    audio_blob = request.files.get("audio")

    actual_user_text = user_text_input # 最終的にAIに渡すテキスト
    transcribed_text = None # 文字起こしされたテキスト

    if not audio_blob and not user_text_input:
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
                transcribed_text = transcribe_audio(tmp.name) # 音声ファイルを文字起こし
                actual_user_text = transcribed_text # 文字起こし結果をAIに渡すテキストとする
                app.logger.info(f"Transcribed audio: {transcribed_text}")
        
        ai_reply = chat_with_ai(conversation_history, user_text=actual_user_text)
        
        return jsonify(reply=ai_reply, user_message_text=actual_user_text) # 文字起こし結果も返す

    except Exception as e:
        app.logger.error(f"Error processing chat request: {e}", exc_info=True)
        return jsonify(error="AIの応答生成中にエラーが発生しました。"), 500

@app.post("/api/generate_diary")
def generate_diary_endpoint():
    """会話履歴から日記を生成し、DBに保存。任意でNotionにも保存。"""
    app.logger.info("Received request for /api/generate_diary")
    data = request.get_json()
    if not data or "conversation" not in data:
        app.logger.error("Conversation history missing")
        return jsonify(error="会話履歴が必要です。"), 400

    conversation_history = data["conversation"]
    if not isinstance(conversation_history, list):
        app.logger.error("Invalid conversation history format")
        return jsonify(error="会話履歴の形式がリストではありません。"), 400

    try:
        diary_content, score, highlights = generate_diary_from_conversation(conversation_history)

        with app.app_context():
            new_diary_entry = Diary(
                diary_content=diary_content,
                sentiment_score=score,
                highlight_events=highlights
            )
            db.session.add(new_diary_entry)
            db.session.commit()
            app.logger.info(f"Saved diary to DB with ID: {new_diary_entry.id}")

        notion_url = ""
        try:
            notion_url = save_to_notion(diary_content)
            app.logger.info("Successfully saved diary to Notion.")
        except Exception as e_notion:
            app.logger.warning(f"Failed to save diary to Notion: {e_notion}")

        return jsonify({
            "summary": diary_content,
            "notion_url": notion_url,
            "sentiment_score": score,
            "highlight_events": highlights
        })

    except Exception as e:
        app.logger.error(f"Error processing generate_diary request: {e}", exc_info=True)
        return jsonify(error="日記の生成中にエラーが発生しました。"), 500

@app.get("/api/diaries")
def get_diaries_endpoint():
    """保存されているすべての日記エントリを取得する"""
    app.logger.info("Received request for /api/diaries")
    try:
        with app.app_context():
            diaries = Diary.query.order_by(Diary.date.desc()).all()
        return jsonify([diary.to_dict() for diary in diaries])
    except Exception as e:
        app.logger.error(f"Error fetching diaries: {e}", exc_info=True)
        return jsonify(error="日記の取得中にエラーが発生しました。"), 500

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    port = int(os.getenv("PORT", 5500))
    app.run(host="0.0.0.0", port=port, debug=False)