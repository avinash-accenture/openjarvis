# Voice, iPhone app, and the 9:10 briefing

Everything here stays on your Mac and your devices. Speech is synthesised by
macOS (`say`, Samantha) or by the iPhone itself; speech-to-text runs on the Mac
with whisper.cpp. Nothing goes to a cloud speech service (local-only mode
blocks them).

## 1. Samantha everywhere

Samantha at 0.72 is now the default voice for chat replies and the briefing.
To change it, in `~/.openjarvis/config.toml`:

```toml
[speech]
tts_backend = "macos_say"
voice_id = "Samantha"
voice_speed = 0.72
```

Audio is produced as WAV for chat replies and AAC (`.m4a`) for the briefing,
which plays on the Mac, on iPhone and in Shortcuts.

## 2. Spoken replies in the app

In the app: **Settings → Voice output** on, then **Autoplay** on. Every answer
is then read aloud in Samantha's voice (citations, links and formatting are
not spoken). Each reply also has a speaker button to replay it.

On iPhone, audio can only start after you have tapped something on the page;
the app unlocks its player on your first tap, so after sending your first
question replies play by themselves.

## 3. Asking by voice

Two ways:

- **iPhone keyboard dictation** (no setup): tap the microphone on the keyboard
  while typing in the chat box.
- **The app's microphone button**: records on the phone, transcribes on your
  Mac with whisper.cpp. Setup on the Mac (once):

  ```bash
  xcode-select --install
  ```

  ```bash
  git clone https://github.com/ggml-org/whisper.cpp ~/Documents/whisper.cpp
  ```

  ```bash
  cd ~/Documents/whisper.cpp && cmake -B build && cmake --build build -j --config Release
  ```

  ```bash
  sh ./models/download-ggml-model.sh base.en && mkdir -p ~/.openjarvis/models && cp models/ggml-base.en.bin ~/.openjarvis/models/
  ```

  If `cmake` is missing, install it with `brew install cmake` (or
  `pip install cmake`). The model download is a one-time public file. Then:

  ```toml
  [speech]
  backend = "whisper_cpp"
  whisper_cpp_binary = "~/Documents/whisper.cpp/build/bin/whisper-cli"
  ```

  Restart the server (`jarvis remote install-server`). Safari's recordings are
  converted with macOS's own `afconvert`; Chrome on the Mac records WebM, which
  also needs `brew install ffmpeg`.

  (`faster-whisper`, the default engine, needs onnxruntime, which has no
  builds for Intel Macs on macOS 12; whisper.cpp does not need it.)

## 4. The app on your iPhone

Safari → your `https://<mac>.<tailnet>.ts.net` address → Share →
**Add to Home Screen**. With no conversation open, the chat screen shows
**Today's briefing** with **Play** and the full text.

If the home-screen icon says the server cannot be found while Safari works:
delete the icon, load the full `…ts.net` address in Safari with Tailscale on,
and add it again from that page.

## 5. The briefing at 9:10

On the Mac, the briefing is generated and played at 9:10 by the LaunchAgent
from Phase 2 (`[digest] schedule = "10 9 * * *"`):

```bash
jarvis briefing install-schedule
```

### Hear it on the iPhone automatically (Shortcuts)

In the **Shortcuts** app → **Automation** → **New Automation** →
**Time of Day**: 9:20 AM (gives the Mac time to finish), Daily, **Run
Immediately**. Add these actions:

1. **Text**: `https://<mac>.<tailnet>.ts.net/api/digest`
2. **Get Contents of URL** (the Text): Method **GET**, Headers:
   `Authorization` = `Bearer <your API key>`
3. **Get Dictionary Value**: key `text` from *Contents of URL*
4. **Speak Text**: *Dictionary Value*, Voice **Samantha**, Rate to taste

The iPhone speaks the briefing text with its own Samantha voice. If no briefing
exists yet the request returns an error; run it again later or check
`jarvis briefing status` on the Mac.

### "Hey Siri, Ask Jarvis"

New Shortcut named **Ask Jarvis**:

1. **Dictate Text**
2. **Get Contents of URL**: `https://<mac>.<tailnet>.ts.net/api/ask`,
   Method **POST**, Headers `Authorization` = `Bearer <your API key>`,
   Request Body **JSON** with field `question` = *Dictated Text*
3. **Get Dictionary Value**: key `spoken` from *Contents of URL*
4. **Speak Text**: *Dictionary Value*, Voice **Samantha**

Say "Hey Siri, Ask Jarvis", then your question. Answers take a while on the
Mac's CPU; if Shortcuts gives up waiting, ask shorter questions or use the app.

Privacy notes:

- Both shortcuts store your API key on the phone, as the app already does.
  If the phone is lost, rotate the key (see remote-access.md).
- **Dictate Text** may send your spoken question to Apple for recognition
  unless the iPhone does on-device dictation for your language. The answer is
  spoken by the iPhone itself. Use the app's microphone button to keep the
  recording between the phone and the Mac only.
