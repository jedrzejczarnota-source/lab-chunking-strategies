## Step 1 — Setup and data loading

**Do:** Set up your environment and load both content sources (the podcast audio and the PDF).

### 1. Create a new Jupyter notebook

Create a new notebook in a folder for this lab and name it `chunking_strategies.ipynb`.
Put these two files **in the same folder** (this is your *current working folder*):

- Podcast audio: `The_Blueprint_For_Trustworthy_AI.m4a`
- PDF document: `ai_hleg_ethics_guidelines_for_trustworthy_ai-en.pdf`

### 2. Install the required packages

```python
%pip install -q langchain-text-splitters pypdf openai tiktoken sentence-transformers matplotlib pandas python-dotenv
```

Transcribing the audio also needs **ffmpeg** on your computer (it cuts the big audio file into smaller pieces).
Check it in a terminal with `ffmpeg -version`. If it is missing: `brew install ffmpeg` (Mac) or `sudo apt install ffmpeg` (Linux).

Now import everything you need:

```python
import os, re, json, statistics
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)
```

✅ **Expected first output:** the install finishes with no errors and the imports run without an error.

### 3. Load environment variables for API keys

Create a text file called `.env` in the same folder. It has one line (use your own key):

```
OPENAI_API_KEY=sk-...your-key-here...
```

Then load it:

```python
load_dotenv()   # reads the .env file

if os.getenv("OPENAI_API_KEY"):
    print("API key found")
else:
    print("No API key found. Check your .env file (it is needed for transcription).")
```

⚠️ Never share or upload your `.env` file. Never paste your key into the notebook.

### 4. Load the podcast audio file

```python
AUDIO_PATH = Path("The_Blueprint_For_Trustworthy_AI.m4a")

print("Current working folder:", Path.cwd())
if AUDIO_PATH.exists():
    print(f"Audio found: {AUDIO_PATH.name} ({AUDIO_PATH.stat().st_size / 1_000_000:.1f} MB)")
else:
    print("Audio NOT FOUND. Move the file into the folder shown above.")
```

The file is about 30 MB. The transcription service accepts files up to **25 MB**, so the next step cuts the audio into smaller pieces first.

### 5. Transcribe the audio file (save it to a local file)

The text splitters work on **text**, not on sound. So we turn the audio into text, and save the text in `transcript.txt`.

```python
import subprocess, glob

TRANSCRIPT_PATH = Path("transcript.txt")

def transcribe_audio(audio_path, out_path, model="whisper-1", piece_seconds=600):
    """Transcribe audio with the OpenAI API and save the text. Skips all work if out_path already exists."""
    out = Path(out_path)
    if out.exists() and out.stat().st_size > 0:
        print(f"{out} already exists -> skipping transcription")
        return out.read_text(encoding="utf-8")

    # 1) shrink and cut the audio: mono, 64 kbps mp3, 10-minute pieces
    Path("_pieces").mkdir(exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(audio_path), "-ac", "1", "-b:a", "64k",
                    "-f", "segment", "-segment_time", str(piece_seconds), "-reset_timestamps", "1",
                    "_pieces/piece_%02d.mp3"], check=True)

    # 2) transcribe the pieces in order
    from openai import OpenAI
    client = OpenAI()                                  # uses OPENAI_API_KEY from the environment
    texts = []
    for f in sorted(glob.glob("_pieces/piece_*.mp3")):
        with open(f, "rb") as fh:
            texts.append(client.audio.transcriptions.create(model=model, file=fh).text)
        print(f, "done")

    text = " ".join(texts)
    out.write_text(text, encoding="utf-8")             # save to a local file
    return text

podcast_text = transcribe_audio(AUDIO_PATH, TRANSCRIPT_PATH)
print(f"Podcast transcript length: {len(podcast_text)} chars")
print(podcast_text[:200])
```

### 6. Load the PDF document (from a URL or a local file)

Here the PDF is a local file, so we read it directly. (If your teacher gives you a URL instead, download the file first and save it in your folder.)

```python
PDF_PATH = Path("ai_hleg_ethics_guidelines_for_trustworthy_ai-en.pdf")

def load_pdf(path=PDF_PATH):
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

pdf_text = load_pdf()
print(f"PDF length: {len(pdf_text)} chars")
print(pdf_text[:200])
```

✅ **Expected output of Step 1:** a transcript length, a PDF length (about 156,000 characters), and the first 200 characters of each.

---

## Step 2 — Implement Fixed-Size Chunking

**Do:** Create fixed-size chunks for both documents with `CharacterTextSplitter`.

> ⚠️ By default, `CharacterTextSplitter` uses `separator="\n\n"` (a paragraph break). For true fixed-size chunks, change it to `separator=""`:

```python
splitter = CharacterTextSplitter(separator="", chunk_size=500, chunk_overlap=50)
```

### Analysis questions

Answer each question in a markdown cell, **with numbers from your own runs**. Don't just say "yes" or "no".

Your numbers can come from simple maths on the list of chunks, for example `sizes = [len(c) for c in chunks]`.

**1. Does fixed-size chunking break sentences in the middle?**

> 💡 **Hints**
> - Look at the **end** of some chunks (try `chunk[-50:]`) and at the **start** of the next ones. Do you see a word or a sentence cut in half?
> - Count it: how many chunks end with `.`, `?` or `!`? What **percentage** of all chunks is that?
> - Does the answer change when you change the chunk size (500, 1000, 2000) or the overlap (0, 50, 100)?

**2. How does it handle paragraph boundaries?**

> 💡 **Hints**
> - Try both settings from the explanation above: `separator=""` and the default `separator="\n\n"`.
> - For each run, count the **number of chunks**, and find the **average, minimum and maximum** chunk size.
> - Is the maximum bigger than your `chunk_size`? How many chunks are bigger than `chunk_size`?
> - Count the `\n\n` in each text (`text.count("\n\n")`). What does that number tell you about what the default separator can do?

**3. Which content type handles fixed-size chunking better?**

> 💡 **Hints**
> - Put the podcast and the PDF **side by side**: same chunk size, same overlap, same separator.
> - Compare the number of chunks, the average, the minimum and the maximum.
> - Compare how many chunks end cleanly (with `.`, `?` or `!`) in each text.
> - Remember what each text looks like inside: one has many line breaks and paragraphs, the other is more like one long block. Does that explain your numbers?

> ✍️ In 2–3 sentences, write your conclusion and **mention at least two numbers** that support it.

---

## Step 4 — Implement Token-Based Chunking

**Do:** Chunk based on token count (more accurate for LLM context windows).

### The concept

For token chunking we do not cut the text. We cut the **token IDs**:

1. **Create the tokenizer of the model you use.** Always use the tokenizer of your model. We use `cl100k_base`, the tokenizer of **GPT-4** and of the **`text-embedding-3-small`** embedding model.
2. **Pass the PDF text and the podcast transcript through it.** It turns each text into a list of **token IDs** (whole numbers).
3. **Apply the chunking on the token IDs:** a window of `chunk_size` IDs moves over the list. Each new window starts `chunk_size - chunk_overlap` IDs later.
4. **Decode** each window of IDs back into text. That text is your chunk.

**Example:** `chunk_size = 300`, `chunk_overlap = 50`. Each window takes **300 token IDs**. The next window starts 250 IDs later (300 − 50), so the last 50 IDs of one chunk are the first 50 IDs of the next one.

| Chunk | Token IDs used |
|---|---|
| 1 | IDs 0 – 299 |
| 2 | IDs 250 – 549 |
| 3 | IDs 500 – 799 |
| ... | ... |

The last chunk is usually shorter, because the IDs run out.

### How to implement it

#### 1. Use `TokenTextSplitter`

`TokenTextSplitter` does the whole token chunking for you in one call:

```python
splitter = TokenTextSplitter(encoding_name="cl100k_base", chunk_size=300, chunk_overlap=50)
pdf_token_chunks = splitter.split_text(pdf_text)
podcast_token_chunks = splitter.split_text(podcast_text)
print(len(pdf_token_chunks), "PDF chunks,", len(podcast_token_chunks), "podcast chunks")
```

**What happens inside `split_text()`, step by step:**

1. **Creates the tokenizer** named in `encoding_name` (`cl100k_base`).
2. **Encodes the text** into a list of token IDs.
3. **Moves a window over the IDs.** The window takes `chunk_size` IDs (300). The next window starts `chunk_size - chunk_overlap` IDs later (250), so 50 IDs repeat.
4. **Decodes each window** of IDs back into text.
5. **Returns a list of text chunks** (strings). You get text, not IDs.
