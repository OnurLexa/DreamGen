###
# Discord slash-command bot that uses "Stability AI" to generate images
#  
# Features:
# - /picture slash command with optional params
# - safety filter handling
# - concurrency limter + per-user cooldown
# - usage logging (sqlite)
# - config via .env 
###

import os
import io
import time
import base64
import json
import sqlite3
import asyncio
from datetime import datetime, timezone

import aiohttp
import discord
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv

# Load .env
load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")
# Default model, değiştirilebilir: örn "stable-diffusion-xl-1024-v1-0" veya "stable-diffusion-v1-5"
DEFAULT_MODEL = os.getenv("STABILITY_MODEL", "stable-diffusion-xl-1024-v1-0")
# Max concurrent generation
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "2"))
# Simple per-user cooldown (seconds)
USER_COOLDOWN = int(os.getenv("USER_COOLDOWN", "10"))
# Optional Guild ID to register commands faster during development (set GUILD_ID)
GUILD_ID = os.getenv("GUILD_ID")  # optional, integer as string

if not DISCORD_TOKEN or not STABILITY_API_KEY:
    raise SystemExit("DISCORD_TOKEN and STABILITY_API_KEY must be set in environment variables (.env).")

# Concurrency control
generation_semaphore = asyncio.Semaphore(MAX_CONCURRENT)

# Cooldown tracking
last_request = {}  # user_id -> timestamp

# Create/Connect to sqlite DB for logs
DB_PATH = os.getenv("USAGE_DB_PATH", "usage.db")
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS usages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    username TEXT,
    prompt TEXT,
    negative_prompt TEXT,
    model TEXT,
    seed INTEGER,
    width INTEGER,
    height INTEGER,
    steps INTEGER,
    samples INTEGER,
    cfg_scale REAL,
    timestamp TEXT
)
""")
conn.commit()

intents = discord.Intents.default()
bot = commands.Bot(command_prefix="!", intents=intents)

# For slash commands
tree = bot.tree

# Helper: basic prompt blacklist (very simple)
BANNED_KEYWORDS = [
    # Don't include explicit items here; instead keep this as a placeholder for admin to expand.
    # For example you may add "child", "illegal-drug", etc. in production.
]
def prompt_blocked(prompt: str) -> bool:
    p = prompt.lower()
    for w in BANNED_KEYWORDS:
        if w in p:
            return True
    return False

async def call_stability_generate(prompt: str,
                                  negative_prompt: str | None = None,
                                  steps: int = 30,
                                  cfg_scale: float = 7.0,
                                  width: int = 512,
                                  height: int = 512,
                                  samples: int = 1,
                                  seed: int | None = None,
                                  model: str = DEFAULT_MODEL,
                                  timeout: int = 180):
    """
    Calls Stability API text-to-image generation endpoint and returns list of dicts:
    [{'bytes': b'...', 'seed': 123, 'finish_reason': 'SUCCESS'}, ...]
    Raises Exception on HTTP error.
    """
    url = f"https://api.stability.ai/v1/generation/{model}/text-to-image"
    headers = {
        "Authorization": f"Bearer {STABILITY_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    text_prompts = [{"text": prompt, "weight": 1.0}]
    if negative_prompt:
        text_prompts.append({"text": negative_prompt, "weight": -1.0})

    payload = {
        "text_prompts": text_prompts,
        "cfg_scale": float(cfg_scale),
        "height": int(height),
        "width": int(width),
        "samples": int(samples),
        "steps": int(steps),
    }
    if seed:
        payload["seed"] = int(seed)

    await generation_semaphore.acquire()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload, timeout=timeout) as resp:
                text = await resp.text()
                if resp.status != 200:
                    raise Exception(f"Stability API error {resp.status}: {text}")
                data = json.loads(text)
    finally:
        generation_semaphore.release()

    artifacts = data.get("artifacts") or data.get("result") or []
    # Normalize nested arrays
    if artifacts and isinstance(artifacts[0], list):
        artifacts = artifacts[0]

    results = []
    for art in artifacts:
        b64 = art.get("base64") or art.get("b64_json") or art.get("b64")
        finish = art.get("finishReason") or art.get("finish_reason") or art.get("finishReason")
        seed_val = art.get("seed")
        if not b64:
            continue
        try:
            img_bytes = base64.b64decode(b64)
        except Exception as e:
            raise Exception("Failed to decode base64 image: " + str(e))
        results.append({"bytes": img_bytes, "seed": seed_val, "finish_reason": finish})
    return results

def log_usage(user: discord.User, prompt: str, negative_prompt: str | None, model: str,
              seed: int | None, width:int, height:int, steps:int, samples:int, cfg_scale:float):
    ts = datetime.now(timezone.utc).isoformat()
    cursor.execute("""
    INSERT INTO usages (user_id, username, prompt, negative_prompt, model, seed, width, height, steps, samples, cfg_scale, timestamp)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (str(user.id), f"{user.name}#{user.discriminator}", prompt, negative_prompt, model, seed, width, height, steps, samples, cfg_scale, ts))
    conn.commit()

def make_result_embed(prompt: str, negative_prompt: str | None, model: str, seed: int | None, width:int, height:int, steps:int, samples:int, cfg_scale:float):
    embed = discord.Embed(title="🖼️ Resim oluşturuldu", description=f"**Prompt:** {prompt[:3500]}", color=0x2ecc71)
    if negative_prompt:
        embed.add_field(name="Negative prompt", value=(negative_prompt[:1024]), inline=False)
    meta = f"Model: `{model}`\nSeed: `{seed}`\n{width}x{height}px • Steps: `{steps}` • Samples: `{samples}` • CFG: `{cfg_scale}`"
    embed.add_field(name="Ayarlar", value=meta, inline=False)
    embed.set_footer(text="Stability AI ile üretildi")
    return embed

@tree.command(name="resim", description="Stability AI ile resim üret (text-to-image)")
@app_commands.describe(
    prompt="Resim için Türkçe/İngilizce prompt (zorunlu)",
    negative_prompt="Olmasını istemediğin şeyler (opsiyonel)",
    steps="Denoise adımı (20-80 arası önerilir)",
    cfg_scale="Ne kadar 'prompt'a bağlı kalsın (4-20 önerilir)",
    width="Genişlik (256/512/768/1024)",
    height="Yükseklik (256/512/768/1024)",
    samples="Bir seferde kaç çıktı (1-4)",
    seed="İsteğe bağlı seed (int), aynı seed aynı sonucu verir",
    model="Model id (opsiyonel, env DEFAULT_MODEL ile değiştirilebilir)"
)
async def slash_resim(interaction: discord.Interaction,
                      prompt: str,
                      negative_prompt: str | None = None,
                      steps: int = 30,
                      cfg_scale: float = 7.0,
                      width: int = 512,
                      height: int = 512,
                      samples: int = 1,
                      seed: int | None = None,
                      model: str | None = None):
    uid = interaction.user.id
    now = time.time()
    last = last_request.get(uid, 0)
    if now - last < USER_COOLDOWN:
        await interaction.response.send_message(f"Lütfen `{int(USER_COOLDOWN - (now - last))}` saniye bekle ve tekrar dene.", ephemeral=True)
        return
    last_request[uid] = now

    if model is None:
        model = DEFAULT_MODEL

    if prompt_blocked(prompt) or (negative_prompt and prompt_blocked(negative_prompt)):
        await interaction.response.send_message("Girilen prompt engellendi (güvenlik/anahtar kelime).", ephemeral=True)
        return

    await interaction.response.defer(thinking=True)

    steps = max(10, min(80, int(steps)))
    cfg_scale = max(1.0, min(30.0, float(cfg_scale)))
    width = int(width)
    height = int(height)
    if width not in (256, 512, 768, 1024): width = 512
    if height not in (256, 512, 768, 1024): height = 512
    samples = max(1, min(4, int(samples)))

    try:
        results = await call_stability_generate(prompt=prompt,
                                                negative_prompt=negative_prompt,
                                                steps=steps,
                                                cfg_scale=cfg_scale,
                                                width=width,
                                                height=height,
                                                samples=samples,
                                                seed=seed,
                                                model=model)
    except Exception as e:
        await interaction.followup.send(f"API isteği sırasında hata oluştu: `{str(e)}`", ephemeral=True)
        return

    if not results:
        await interaction.followup.send("Hiçbir görsel üretilmedi veya beklenmeyen bir cevap alındı.", ephemeral=True)
        return

    sent_files = []
    for idx, r in enumerate(results, start=1):
        finish = (r.get("finish_reason") or "").upper() if r.get("finish_reason") else "UNKNOWN"
        if "FILTER" in finish or "CONTENT" in finish:
            await interaction.followup.send(f"Üretim **güvenlik filtresine takıldı** (finish_reason={finish}). Prompt içeriğini gözden geçir ve tekrar dene.", ephemeral=True)
            continue

        img_bytes = r["bytes"]
        seed_val = r.get("seed")
        filename = f"stability_{int(time.time())}_{idx}.png"
        fp = io.BytesIO(img_bytes)
        fp.seek(0)

        embed = make_result_embed(prompt, negative_prompt, model, seed_val, width, height, steps, samples, cfg_scale)
        file = discord.File(fp, filename=filename)
        await interaction.followup.send(embed=embed, file=file)
        sent_files.append(filename)

    try:
        log_usage(interaction.user, prompt, negative_prompt, model, seed, width, height, steps, samples, cfg_scale)
    except Exception:
        pass

    if not sent_files:
        await interaction.followup.send("Görsel oluşturulamadı (muhtemelen içerik filtresine takıldı).", ephemeral=True)

@bot.event
async def on_ready():
    if GUILD_ID:
        guild = discord.Object(id=int(GUILD_ID))
        await tree.sync(guild=guild)
        print(f"Slash commands synced to guild {GUILD_ID}")
    else:
        await tree.sync()
        print("Slash commands synced globally.")
    print(f"Bot hazır: {bot.user} (ID: {bot.user.id})")

if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)