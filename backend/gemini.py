# backend/gemini.py
import os
import base64
import mimetypes
import google.generativeai as genai
from google.generativeai.types import File # ★ File 型をここからインポート
from dotenv import load_dotenv
import time
import traceback
from typing import List, Dict, Union, Optional

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# --- モデル設定 ---
_MODEL_NAME = "gemini-2.5-flash" # 必要に応じて変更

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
あなたはユーザーの日記作成アシスタントです。
入力はユーザーとAI（ログとも）の会話履歴です。
この会話履歴全体を分析し、以下のMarkdown構造で日本語の日記文章を生成してください。

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
        # ★★★ mime_type='audio/webm' を引数に追加 ★★★
        print(f"Uploading with explicit mime_type: audio/webm")
        audio_file = genai.upload_file(path=path, mime_type="audio/webm")
        # ★★★ ここまで変更 ★★★

        print(f"File uploaded: name={audio_file.name}, state={audio_file.state.name}")

        start_time = time.time()
        while audio_file.state.name != "ACTIVE":
            # ... (待機ループは変更なし) ...
            if time.time() - start_time > timeout:
                # ... (タイムアウト処理) ...
                raise TimeoutError(f"File processing timed out for {audio_file.name}")

            print(f"Waiting for file {audio_file.name} to become ACTIVE. Current state: {audio_file.state.name}. Sleeping for 2 seconds...")
            time.sleep(2)

            try:
                fetched_file = genai.get_file(name=audio_file.name)
                audio_file: File = fetched_file
                print(f"Refetched file state for {audio_file.name}: {audio_file.state.name}")
            except Exception as e_get:
                # ... (エラーハンドリング) ...
                raise ValueError(f"Failed to get updated state for file {audio_file.name}") from e_get

            if audio_file.state.name == "FAILED":
                # ... (FAILED時の処理) ...
                error_details = ""
                if hasattr(audio_file, 'processing_error') and audio_file.processing_error:
                    error_details = f" Details: {audio_file.processing_error}"
                raise ValueError(f"File upload or processing failed for {path}.{error_details}")

        print(f"File {audio_file.name} is now ACTIVE.")
        return audio_file

    except Exception as e:
        # ... (エラーハンドリング) ...
        raise

# backend/gemini.py の _call_gemini 関数部分

# --- Gemini API 呼び出し ---
# ★ safety_level 引数を削除し、safety_settings パラメータを省略
def _call_gemini(system_prompt: str, contents: List[Union[str, Dict, File]], model_name: str = _MODEL_NAME) -> str:
    """Gemini APIを呼び出す共通関数"""
    print(f"Calling Gemini model: {model_name}")
    try:
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_prompt
        )
        res = model.generate_content(
            contents=contents,
            request_options={"timeout": 600}
        )
        print("Received response from Gemini.")

        # --- レスポンスチェック ---
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

        # ★★★ finish_reason のチェック方法を変更 ★★★
        finish_reason_val = getattr(candidate, 'finish_reason', None) # finish_reason 属性を取得 (なければ None)
        # STOP を示す値 (通常は 1) 以外の場合に警告/エラー処理
        # 注意: ライブラリのバージョンによっては STOP が 1 以外の場合もあるかもしれないが、多くの場合 1
        STOP_REASON = 1
        is_stopped = (finish_reason_val == STOP_REASON)

        if finish_reason_val is not None and not is_stopped:
             # Enum の可能性があるため .name を試みるが、なければそのまま表示
             finish_reason_str = finish_reason_val.name if hasattr(finish_reason_val, 'name') else str(finish_reason_val)
             print(f"Warning: Response finished with non-STOP reason: {finish_reason_str}")
             if hasattr(candidate, 'safety_ratings'):
                 print(f"Safety Ratings (Candidate): {candidate.safety_ratings}")

             # 安全性設定による停止 (SAFETY = 2 など、値はライブラリによる) の場合も考慮
             # 安全性による停止の場合はエラーとする (値は仮)
             SAFETY_REASON_VALUE = 2 # この値はライブラリのドキュメントで確認が必要
             if finish_reason_val == SAFETY_REASON_VALUE:
                  raise ValueError(f"Gemini response stopped due to safety settings. Reason code: {finish_reason_val}")
        # ★★★ ここまで変更 ★★★

        # テキスト部分を取得する試み (変更なし)
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

def chat_with_ai(
    conversation_history: List[Dict[str, str]],
    user_audio_path: Optional[str] = None,
    user_text: Optional[str] = None
) -> str:
    """
    音声またはテキスト入力と会話履歴を受け取り、AI（友達）としての応答を生成する。
    """
    print("Processing chat request...")
    if not user_audio_path and not user_text:
        raise ValueError("Either user_audio_path or user_text must be provided.")

    active_audio_file = None
    try:
        # 履歴を構築
        contents = []
        for message in conversation_history:
            role = "user" if message["role"] == "user" else "model"
            if "content" in message and message["content"] and message["content"] != "[あなたの音声]":
                contents.append({"role": role, "parts": [{"text": message["content"]}]})

        # 最新のユーザー入力を追加
        user_parts = []
        if user_text:
            user_parts.append({"text": user_text})

        if user_audio_path:
            active_audio_file = _upload_and_wait_for_file(user_audio_path)
            user_parts.append(active_audio_file)

        contents.append({"role": "user", "parts": user_parts})

        # Gemini API 呼び出し
        ai_reply = _call_gemini(_CHAT_SYSTEM_PROMPT, contents)
        return ai_reply

    except Exception as e:
        print(f"Failed to get chat response from Gemini: {e}")
        traceback.print_exc()
        return "ごめん、応答を考えるときにエラーが起きちゃったみたい…"
    finally:
        if active_audio_file and hasattr(active_audio_file, 'name'):
            try:
                print(f"Attempting to delete file: {active_audio_file.name}")
                genai.delete_file(active_audio_file.name)
                print(f"Successfully deleted file: {active_audio_file.name}")
            except Exception as delete_error:
                print(f"Warning: Failed to delete uploaded file {active_audio_file.name}: {delete_error}")


def generate_diary_from_conversation(conversation_history: List[Dict[str, str]]) -> str:
    """
    会話履歴全体を受け取り、構造化された日記を生成する。
    """
    print("Processing diary generation request...")
    try:
        # 1. Geminiに渡すコンテンツを作成 (会話履歴をテキストベースで結合)
        #    システムメッセージやエラーメッセージは除外する
        relevant_history = [
            f"{msg['role']}: {msg.get('content', '[記録なし]')}"
            for msg in conversation_history
            if msg['role'] in ['user', 'ai'] and msg.get('content') # user/ai ロールで content があるもの
        ]
        if not relevant_history:
            return "## 日記\n\n会話の履歴が空のため、日記を生成できませんでした。"

        full_conversation = "\n\n".join(relevant_history)

        # デバッグ用に結合した会話内容を表示
        # print("--- Generating diary from following conversation ---")
        # print(full_conversation)
        # print("----------------------------------------------------")

        contents = [full_conversation] # シンプルなテキスト入力として渡す

        # 2. Gemini API 呼び出し (日記生成用プロンプトを使用)
        diary_text = _call_gemini(_DIARY_SYSTEM_PROMPT, contents)
        return diary_text

    except Exception as e:
        print(f"Failed to generate diary from Gemini: {e}")
        # エラー時は、失敗した旨のテキストを返す
        return f"## 日記の生成に失敗しました\n\nエラーが発生しました。\n```\n{e}\n```"