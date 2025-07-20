# backend/gemini.py
import os
import base64
import mimetypes
import google.generativeai as genai
from google.generativeai.types import File
from dotenv import load_dotenv
import time
import traceback
import json
from typing import List, Dict, Union, Optional, Tuple

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# --- モデル設定 ---
_MODEL_MODEL_FOR_CHAT = "gemini-1.5-flash" # チャット用モデル
_MODEL_MODEL_FOR_DIARY = "gemini-1.5-flash" # 日記生成用モデル
_MODEL_MODEL_FOR_TRANSCRIPTION = "gemini-1.5-flash" # 文字起こし用モデル (音声入力対応モデル)

# --- プロンプト ---
_CHAT_SYSTEM_PROMPT = """
あなたはユーザーの親しい友人AI「ログとも」です。
ユーザーは今日あった出来事を話したがっています。
あなたの役割は以下の通りです。
- ユーザーの話を注意深く聞き、共感を示してください（例：「そうだったんだね」「大変だったね」「それは楽しそうだね」）。
- 話の内容について、1〜2個の簡単な質問をして、ユーザーがさらに話しやすいように促してください（例：「それで、どうなったの？」「その時どう思った？」）。
- ポジティブで、フレンドリーな口調を保ってください。
- 長すぎる応答は避け、会話のテンポを良くしてください。
- ユーザーが音声で入力した場合、その内容を要約したり繰り返したりする必要はありません。会話の流れに沿った応答を生成してください。
""".strip()

_DIARY_SYSTEM_PROMPT = """
あなたはユーザーの日記作成及び分析アシスタントです。
入力はユーザーとAI（ログとも）の会話履歴です。
必ず以下のJSON形式で日本語のレスポンスを生成してください。

{
  "sentiment_score": <float>,
  "highlight_events": "<string>",
  "diary_content": "<string>"
}

各フィールドの要件は以下の通りです。

1.  `sentiment_score`:
    - 会話全体から感じ取れるユーザーの感情を、-1.0（非常にネガティブ）から1.0（非常にポジティブ）の範囲の浮動小数点数で数値化してください。
    - 中立的な感情は0.0とします。

2.  `highlight_events`:
    - 会話の中から、その日を最も象徴する出来事を、50文字以内の非常に短い日本語の文章で要約してください。
    - 例：「新しいカフェで美味しいケーキを食べた日」「仕事のプロジェクトで大きな進展があった」

3.  `diary_content`:
    - 以下のMarkdown構造で、詳細な日本語の日記文章を生成してください。
    - このフィールドの値は、JSON文字列として正しくエスケープしてください（改行は \n として表現）。

    --- Markdown構造 ---
    ## 今日のハイライト
    - 会話の中から、特に印象的だった出来事やユーザーの発言を1〜3個箇条書きで要約してください。

    ## 感じたこと・気づき
    - 会話内容から推測されるユーザーの感情（喜び、悲しみ、怒り、驚き、学びなど）や、考えたこと、気づいたことを2〜4個箇条書きで記述してください。AI自身の発言ではなく、ユーザーの発言や感情に焦点を当ててください。

    ## ポジティブな点
    - 会話の中から、ユーザーにとって良かったこと、楽しかったこと、感謝したことなどを1〜2個見つけて記述してください。

    ## もっと話したいこと / 次のアクション
    - 会話の中で解決しなかったこと、ユーザーが疑問に思っていそうなこと、または明日につながる簡単なアクション（TODOとは限らない）を1〜2個提案または記述してください。

    ### タグ
    #会話ログ #今日の振り返り #感情メモ など、会話内容に合ったタグを3〜5個付けてください。
""".strip()


# --- ファイルアップロードと待機 ---
def _upload_and_wait_for_file(path: str, timeout: int = 60) -> File:
    """ファイルをアップロードし、ACTIVEになるまで待機する"""
    print(f"Attempting to upload file: {path}")
    try:
        print(f"Uploading with explicit mime_type: audio/webm")
        audio_file = genai.upload_file(path=path, mime_type="audio/webm")
        print(f"File uploaded: name={audio_file.name}, state={audio_file.state.name}")

        start_time = time.time()
        while audio_file.state.name != "ACTIVE":
            if time.time() - start_time > timeout:
                raise TimeoutError(f"File processing timed out for {audio_file.name}")

            print(f"Waiting for file {audio_file.name} to become ACTIVE. Current state: {audio_file.state.name}. Sleeping for 2 seconds...")
            time.sleep(2)

            try:
                fetched_file = genai.get_file(name=audio_file.name)
                audio_file: File = fetched_file
                print(f"Refetched file state for {audio_file.name}: {audio_file.state.name}")
            except Exception as e_get:
                raise ValueError(f"Failed to get updated state for file {audio_file.name}") from e_get

            if audio_file.state.name == "FAILED":
                error_details = ""
                if hasattr(audio_file, 'processing_error') and audio_file.processing_error:
                    error_details = f" Details: {audio_file.processing_error}"
                raise ValueError(f"File upload or processing failed for {path}.{error_details}")

        print(f"File {audio_file.name} is now ACTIVE.")
        return audio_file

    except Exception as e:
        raise

# --- Gemini API 呼び出し ---
def _call_gemini(system_prompt: str, contents: List[Union[str, Dict, File]], model_name: str, is_json_output: bool = False) -> str:
    """Gemini APIを呼び出す共通関数"""
    print(f"Calling Gemini model: {model_name}")
    try:
        generation_config = {"response_mime_type": "application/json"} if is_json_output else None
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_prompt
        )
        res = model.generate_content(
            contents=contents,
            request_options={"timeout": 600},
            generation_config=generation_config
        )
        print("Received response from Gemini.")

        if not res.candidates:
             reason = "Unknown"
             if hasattr(res, 'prompt_feedback') and hasattr(res.prompt_feedback, 'block_reason'):
                 block_reason_val = res.prompt_feedback.block_reason
                 reason = block_reason_val.name if hasattr(block_reason_val, 'name') else str(block_reason_val)
             print(f"Error: No candidates returned. Block Reason: {reason}")
             if hasattr(res, 'prompt_feedback') and hasattr(res.prompt_feedback, 'safety_ratings'):
                 print(f"Safety Ratings (Prompt Feedback): {res.prompt_feedback.safety_ratings}")
             raise ValueError(f"Gemini response blocked or empty. Reason: {reason}")

        candidate = res.candidates[0]
        finish_reason_val = getattr(candidate, 'finish_reason', None)
        STOP_REASON = 1
        is_stopped = (finish_reason_val == STOP_REASON)

        if finish_reason_val is not None and not is_stopped:
             finish_reason_str = finish_reason_val.name if hasattr(finish_reason_val, 'name') else str(finish_reason_val)
             print(f"Warning: Response finished with non-STOP reason: {finish_reason_str}")
             if hasattr(candidate, 'safety_ratings'):
                 print(f"Safety Ratings (Candidate): {candidate.safety_ratings}")
             SAFETY_REASON_VALUE = 2
             if finish_reason_val == SAFETY_REASON_VALUE:
                  raise ValueError(f"Gemini response stopped due to safety settings. Reason code: {finish_reason_val}")

        if res.text:
            return res.text
        elif candidate.content and candidate.content.parts:
             text_parts = [part.text for part in candidate.content.parts if hasattr(part, 'text')]
             if text_parts:
                 print("Warning: Response.text is empty, but found text in parts.")
                 return "\n".join(text_parts)
             else:
                 print("Error: Response text and parts are empty.")
                 raise ValueError("Gemini response does not contain any text content.")
        else:
             print(f"Warning: Response text is empty. Finish reason: {getattr(candidate, 'finish_reason', 'N/A')}")
             return ""

    except Exception as e:
        print(f"An error occurred during Gemini API call:")
        traceback.print_exc()
        if hasattr(e, 'message'):
            raise RuntimeError(f"Gemini API Error: {e.message}") from e
        else:
            raise RuntimeError(f"Gemini API Error: {e}") from e


# ---------- 外部公開関数 ----------

def transcribe_audio(audio_path: str) -> str:
    """音声ファイルをテキストに文字起こしする"""
    print(f"Attempting to transcribe audio: {audio_path}")
    active_audio_file = None
    try:
        active_audio_file = _upload_and_wait_for_file(audio_path)
        model = genai.GenerativeModel(model_name=_MODEL_MODEL_FOR_TRANSCRIPTION)
        response = model.generate_content([active_audio_file])
        transcribed_text = response.text
        print(f"Successfully transcribed audio: {transcribed_text[:50]}...")
        return transcribed_text
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        traceback.print_exc()
        raise RuntimeError(f"音声の文字起こしに失敗しました: {e}")
    finally:
        if active_audio_file and hasattr(active_audio_file, 'name'):
            try:
                print(f"Attempting to delete file: {active_audio_file.name}")
                genai.delete_file(active_audio_file.name)
                print(f"Successfully deleted file: {active_audio_file.name}")
            except Exception as delete_error:
                print(f"Warning: Failed to delete uploaded file {active_audio_file.name}: {delete_error}")


def chat_with_ai(
    conversation_history: List[Dict[str, str]],
    user_text: str # user_audio_path は app.py で処理されるため削除
) -> str:
    """
    テキスト入力と会話履歴を受け取り、AI（友達）としての応答を生成する。
    """
    print("Processing chat request...")
    if not user_text:
        raise ValueError("user_text must be provided.")

    try:
        contents = []
        for message in conversation_history:
            role = "user" if message["role"] == "user" else "model"
            if "content" in message and message["content"]:
                contents.append({"role": role, "parts": [{"text": message["content"]}]})

        # 最新のユーザー入力を追加
        contents.append({"role": "user", "parts": [{"text": user_text}]})

        ai_reply = _call_gemini(_CHAT_SYSTEM_PROMPT, contents, model_name=_MODEL_MODEL_FOR_CHAT, is_json_output=False)
        return ai_reply

    except Exception as e:
        print(f"Failed to get chat response from Gemini: {e}")
        traceback.print_exc()
        return "ごめん、応答を考えるときにエラーが起きちゃったみたい…"

def generate_diary_from_conversation(
    conversation_history: List[Dict[str, str]]
) -> Tuple[str, Optional[float], Optional[str]]:
    """
    会話履歴全体を受け取り、構造化された日記、感情スコア、ハイライトを生成する。
    戻り値: (日記コンテンツ, 感情スコア, ハイライト) のタプル
    """
    print("Processing diary generation request...")
    # エラー時のデフォルト値
    default_diary = "## 日記の生成に失敗しました\n\n会話の履歴が空か、AIが内容を解析できませんでした。"
    default_score = 0.0
    default_highlight = "内容の要約に失敗"

    try:
        relevant_history = [
            f"{msg['role']}: {msg.get('content', '[記録なし]')}"
            for msg in conversation_history
            if msg['role'] in ['user', 'ai'] and msg.get('content')
        ]
        if not relevant_history:
            return default_diary, default_score, default_highlight

        full_conversation = "\n\n".join(relevant_history)
        contents = [full_conversation]

        # Gemini API 呼び出し (日記生成用プロンプトとJSONモードを使用)
        raw_response = _call_gemini(_DIARY_SYSTEM_PROMPT, contents, model_name=_MODEL_MODEL_FOR_DIARY, is_json_output=True)

        # JSONレスポンスをパース
        try:
            data = json.loads(raw_response)
            
            diary_content = data.get("diary_content", default_diary)
            sentiment_score = data.get("sentiment_score")
            highlight_events = data.get("highlight_events", default_highlight)

            # スコアの型をfloatに統一
            score = float(sentiment_score) if isinstance(sentiment_score, (int, float)) else default_score

            return diary_content, score, highlight_events

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Failed to parse JSON response from Gemini: {e}")
            print(f"Raw response was: {raw_response}")
            # パースに失敗した場合、生のレスポンスを日記の内容として返し、その他はデフォルト値を使用
            return raw_response, default_score, default_highlight

    except Exception as e:
        print(f"Failed to generate diary from Gemini: {e}")
        traceback.print_exc()
        error_message = f"## 日記の生成に失敗しました\n\nエラーが発生しました。\n```\n{e}\n```"
        return error_message, default_score, default_highlight