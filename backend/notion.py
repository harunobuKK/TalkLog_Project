# backend/notion.py
import os
import requests
import json
from datetime import datetime
from dotenv import load_dotenv
import logging # ロギング追加

load_dotenv()
_NOTION_TOKEN   = os.getenv("NOTION_API_KEY")
_DATABASE_ID    = os.getenv("NOTION_DATABASE_ID")
_NOTION_API_URL = "https://api.notion.com/v1/pages"
_HEADERS = {
    "Authorization": f"Bearer {_NOTION_TOKEN}",
    "Notion-Version": "2022-06-28",
    "Content-Type": "application/json"
}

logger = logging.getLogger(__name__) # ロガー取得

def save_to_notion(content: str) -> str:
    """Markdown 日記テキストを Notion データベースに 1 行として登録"""
    if not _NOTION_TOKEN or not _DATABASE_ID:
        logger.error("Notion API Key or Database ID is not set in .env")
        raise ValueError("Notion API Key or Database ID is missing.")

    page_title = f"日記 {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    # Notion API の rich_text は 2000文字制限があるので注意
    # Markdown を Notion のブロック形式に変換するとよりリッチになるが、今回は簡略化
    # backend/notion.py (確認)
    payload = {
        "parent": { "database_id": _DATABASE_ID },
        "properties": {
            "Name": { # これはOK
                "title": [{ "text": { "content": page_title } }]
            },
            "Diary": { # Notion側でこの名前に設定したのでOK
                "rich_text": [{
                    "text": { "content": content[:2000] + ('...' if len(content) > 2000 else '') }
                }]
            }
        }
    }

    logger.info(f"Sending data to Notion DB: {_DATABASE_ID}")
    try:
        res = requests.post(_NOTION_API_URL, headers=_HEADERS, json=payload, timeout=30)
        res.raise_for_status() # HTTPエラーがあれば例外発生
        response_data = res.json()
        page_url = response_data.get("url", "")
        logger.info(f"Successfully created Notion page: {page_url}")
        return page_url
    except requests.exceptions.RequestException as e:
        logger.error(f"Notion API request failed: {e}", exc_info=True)
        # エラーレスポンスの内容もログに出力
        if hasattr(e, 'response') and e.response is not None:
             try:
                 logger.error(f"Notion API response content: {e.response.json()}")
             except json.JSONDecodeError:
                 logger.error(f"Notion API response content (non-JSON): {e.response.text}")
        raise # エラーを再送出して app.py でハンドリングさせる