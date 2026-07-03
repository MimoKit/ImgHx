"""ImgHx — 基于 Gilbert 空间填充曲线的图片混淆/解混淆 GsCore 插件。"""
from __future__ import annotations

import asyncio
import base64
import binascii
import io
import math
from pathlib import Path
from typing import Literal
from urllib.request import Request, urlopen

from PIL import Image

from gsuid_core.bot import Bot
from gsuid_core.logger import logger
from gsuid_core.models import Event
from gsuid_core.segment import MessageSegment
from gsuid_core.sv import Plugins, SV

Plugins(
    name='ImgHx',
    disable_force_prefix=True,
    allow_empty_prefix=True,
)

sv = SV('图片混淆')

Operation = Literal['encrypt', 'decrypt']
LOG_PREFIX = '[ImgHx]'
MAX_IMAGE_BYTES = 20 * 1024 * 1024


# ── Gilbert 2D 空间填充曲线 ───────────────────────────────────────────

def _sign(x: int) -> int:
    return -1 if x < 0 else (1 if x > 0 else 0)


def _generate2d(
    x: int, y: int,
    ax: int, ay: int,
    bx: int, by: int,
    coords: list[tuple[int, int]],
) -> None:
    w = abs(ax + ay)
    h = abs(bx + by)
    dax, day = _sign(ax), _sign(ay)
    dbx, dby = _sign(bx), _sign(by)
    if h == 1:
        for _ in range(w):
            coords.append((x, y))
            x += dax
            y += day
        return
    if w == 1:
        for _ in range(h):
            coords.append((x, y))
            x += dbx
            y += dby
        return
    ax2, ay2 = ax // 2, ay // 2
    bx2, by2 = bx // 2, by // 2
    w2 = abs(ax2 + ay2)
    h2 = abs(bx2 + by2)
    if 2 * w > 3 * h:
        if w2 % 2 and w > 2:
            ax2 += dax
            ay2 += day
        _generate2d(x, y, ax2, ay2, bx, by, coords)
        _generate2d(x + ax2, y + ay2, ax - ax2, ay - ay2, bx, by, coords)
    else:
        if h2 % 2 and h > 2:
            bx2 += dbx
            by2 += dby
        _generate2d(x, y, bx2, by2, ax2, ay2, coords)
        _generate2d(x + bx2, y + by2, ax, ay, bx - bx2, by - by2, coords)
        _generate2d(
            x + (ax - dax) + (bx2 - dbx),
            y + (ay - day) + (by2 - dby),
            -bx2, -by2, -(ax - ax2), -(ay - ay2),
            coords,
        )


def _gilbert2d(width: int, height: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    if width >= height:
        _generate2d(0, 0, width, 0, 0, height, coords)
    else:
        _generate2d(0, 0, 0, height, width, 0, coords)
    return coords


def _process(image_data: bytes, operation: Operation) -> bytes:
    img = Image.open(io.BytesIO(image_data)).convert('RGBA')
    w, h = img.size
    n = w * h
    src = bytearray(img.tobytes())
    dst = bytearray(len(src))
    curve = _gilbert2d(w, h)
    offset = round((math.sqrt(5) - 1) / 2 * n)
    if operation == 'encrypt':
        for i in range(n):
            ox, oy = curve[i]
            nx, ny = curve[(i + offset) % n]
            sp = 4 * (ox + oy * w)
            dp = 4 * (nx + ny * w)
            dst[dp:dp + 4] = src[sp:sp + 4]
    else:
        for i in range(n):
            ox, oy = curve[i]
            nx, ny = curve[(i + offset) % n]
            sp = 4 * (nx + ny * w)
            dp = 4 * (ox + oy * w)
            dst[dp:dp + 4] = src[sp:sp + 4]
    out = Image.frombytes('RGBA', (w, h), bytes(dst))
    buf = io.BytesIO()
    out.save(buf, 'PNG')
    return buf.getvalue()


# ── 图像读取 ──────────────────────────────────────────────────────────

def _read_image_bytes(source: str) -> bytes | None:
    text = source.strip()
    if not text:
        return None
    try:
        if text.startswith('data:image/') and ',' in text:
            data = base64.b64decode(text.split(',', 1)[1], validate=False)
        elif text.startswith('base64://'):
            data = base64.b64decode(text[9:], validate=False)
        else:
            if text.startswith('link://'):
                text = text[7:]
            if text.startswith(('http://', 'https://')):
                req = Request(text, headers={'User-Agent': 'Mozilla/5.0'})
                with urlopen(req, timeout=20) as resp:
                    data = resp.read(MAX_IMAGE_BYTES + 1)
            else:
                p = Path(text)
                if not p.is_file():
                    return None
                data = p.read_bytes()
    except (OSError, ValueError, binascii.Error) as exc:
        logger.warning(f'{LOG_PREFIX} 读取图片失败: {exc}')
        return None
    if not data or len(data) > MAX_IMAGE_BYTES:
        return None
    return data


def _get_image_ref(ev: Event) -> str | None:
    for content in ev.content or []:
        if content.type in {'image', 'img'} and isinstance(content.data, str) and content.data.strip():
            return content.data.strip()
    for item in ev.image_list or []:
        if isinstance(item, str) and item.strip():
            return item.strip()
    if isinstance(ev.image, str) and ev.image.strip():
        return ev.image.strip()
    return None


# ── 命令处理 ──────────────────────────────────────────────────────────

async def _handle(bot: Bot, ev: Event, operation: Operation) -> None:
    ref = _get_image_ref(ev)
    if not ref:
        op_name = '混淆' if operation == 'encrypt' else '解混淆'
        return await bot.send(f'请同时发送图片，例如：{op_name} [图片]')

    logger.info(f'{LOG_PREFIX} 用户 {ev.user_id} 请求{"混淆" if operation == "encrypt" else "解混淆"}')

    raw = await asyncio.to_thread(_read_image_bytes, ref)
    if raw is None:
        return await bot.send('读取图片失败，请重试。')

    try:
        result = await asyncio.to_thread(_process, raw, operation)
    except Exception as exc:
        logger.warning(f'{LOG_PREFIX} 处理图片失败: {exc}')
        return await bot.send('处理图片失败，请重试。')

    await bot.send(MessageSegment.image(result))


@sv.on_command(('混淆', '图片混淆'), block=True)
async def obfuscate(bot: Bot, ev: Event):
    await _handle(bot, ev, 'encrypt')


@sv.on_command(('解混淆', '图片解混淆'), block=True)
async def deobfuscate(bot: Bot, ev: Event):
    await _handle(bot, ev, 'decrypt')
