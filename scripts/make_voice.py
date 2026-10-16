"""Tao san file giong noi canh bao bang edge-tts (giong tieng Viet neural cua Microsoft).

Chi can mang luc chay script nay mot lan; luc demo chi phat file mp3 co san tren may.
Chay: python scripts/make_voice.py
"""
import asyncio
from pathlib import Path

import edge_tts

VOICE = "vi-VN-HoaiMyNeural"
OUT = Path(__file__).resolve().parent.parent / "assets" / "voice"
LINES = {
    "calling": "Cảnh báo. Vui lòng không nghe điện thoại khi đang lái xe.",
    "texting": "Cảnh báo. Vui lòng không sử dụng điện thoại khi đang lái xe.",
    "smoking": "Cảnh báo. Vui lòng không hút thuốc khi đang lái xe.",
}


async def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, text in LINES.items():
        path = OUT / f"{name}.mp3"
        await edge_tts.Communicate(text, VOICE, rate="+10%").save(str(path))
        print(path.name, path.stat().st_size, "bytes")


if __name__ == "__main__":
    asyncio.run(main())
