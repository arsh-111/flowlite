# Flowlite architecture

Flowlite is a local-first dictation application. It has two user-facing entry
points—the browser demo and the command-line interface—and a shared Python
pipeline for speech detection, recognition, and transcript cleanup.

## 1. Logical architecture

```mermaid
flowchart TB
    User["User"]

    subgraph Entry["User-facing entry points"]
        Browser["Browser demo<br/>static/index.html"]
        CLI["CLI<br/>live · transcribe · clean · teach"]
        API["FastAPI server<br/>GET / · POST /clean · WS /ws"]
        Browser -->|"HTTP / WebSocket"| API
        CLI -->|"serve command"| API
    end

    subgraph Inputs["Audio and text inputs"]
        Mic["Microphone"]
        Wav["WAV recording"]
        Typed["Typed transcript"]
        Mic --> Browser
        Wav --> CLI
        Typed --> Browser
        Typed --> CLI
    end

    subgraph Core["Shared dictation and cleanup pipeline"]
        Audio["Audio conversion<br/>PCM / WAV → mono float32, 16 kHz"]
        Stream["StreamingTranscriber<br/>frame buffering · utterance state"]
        VAD["Energy VAD<br/>speech detection · silence endpoint"]
        ASR["ASR backend<br/>faster-whisper"]
        Partial["Partial events<br/>stable + tentative words"]
        Final["Final transcript"]
        Memory["Correction memory<br/>apply learned replacements"]
        Cleaner{"Cleaner selection"}
        Rules["Rule-based cleanup"]
        LLM["LLM cleanup<br/>OpenAI-compatible endpoint"]
        Guard{"Validate LLM output"}
        Result["Clean text + timings"]

        Audio --> Stream
        Stream -->|"30 ms frames"| VAD
        VAD -->|"speech / silence decision"| Stream
        Stream -->|"periodic decode"| ASR
        ASR -->|"hypothesis"| Stream
        Stream --> Partial
        Stream -->|"silence endpoint / flush"| ASR
        ASR -->|"final hypothesis"| Stream
        Stream --> Final
        Final --> Memory --> Cleaner
        Cleaner -->|"raw / none"| Result
        Cleaner -->|"rules"| Rules --> Result
        Cleaner -->|"LLM"| LLM --> Guard
        Guard -->|"valid"| Result
        Guard -->|"error / rejected output"| Rules
    end

    subgraph Support["Configuration, learning, and evaluation"]
        Config["Settings and factories<br/>environment + CLI options"]
        Dictionary["Preferred-spelling dictionary"]
        Corrections["Local corrections file<br/>~/.flowlite/corrections.json"]
        Bench["Benchmark runners"]
        Metrics["Quality + latency metrics"]
        Samples["Text samples + recorded WAVs"]
        Reports["JSON / Markdown reports"]
        Tests["Automated tests"]
        Config -.-> ASR
        Config -.-> Cleaner
        Dictionary -.-> ASR
        Dictionary -.-> Cleaner
        Corrections -.-> Memory
        Samples --> Bench --> Metrics --> Reports
        Tests -.-> Core
    end

    User --> Browser
    User --> CLI
    Browser -->|"PCM16 audio"| Audio
    CLI -->|"captured audio"| Audio
    CLI -->|"WAV"| Audio
    Browser -->|"partial / final / clean JSON events"| User
    Result -->|"clean text"| Browser
    Result -->|"console output"| CLI
    Partial -->|"live preview"| Browser
```

### Reading the diagram

- **Entry points** determine how a request enters the app. The browser is a
  static page served by FastAPI; it is not a separate frontend server. The CLI
  can run the same pipeline directly or start the FastAPI server.
- **The streaming core** collects audio into utterances. VAD decides when speech
  starts and when a pause ends an utterance; it does not recognize words.
- **ASR** is faster-whisper. During speech, periodic decodes may produce partial
  previews. At an endpoint, a final decode produces the transcript used by
  cleanup.
- **Cleanup** is selectable: passthrough, deterministic rules, or an
  OpenAI-compatible LLM. LLM output is validated; rejected output or request
  errors use the rules-based fallback.
- **Support components** provide runtime configuration, vocabulary hints,
  correction memory, tests, and offline quality/latency measurement.

## 2. Live dictation sequence

```mermaid
sequenceDiagram
    actor User
    participant UI as Browser demo
    participant API as FastAPI /ws
    participant Audio as PCM conversion
    participant Stream as StreamingTranscriber
    participant VAD as Energy VAD
    participant ASR as faster-whisper
    participant Memory as CorrectionMemory
    participant Cleaner as Cleaner

    User->>UI: Allow microphone and start
    UI->>API: Open WebSocket /ws?style=...
    loop While speaking
        User->>UI: Speak
        UI->>API: PCM16 audio chunks (16 kHz mono)
        API->>Audio: Convert PCM16 to float32
        Audio->>Stream: Feed audio samples
        Stream->>VAD: Check 30 ms frames
        VAD-->>Stream: Speech / silence
        opt Partial interval reached
            Stream->>ASR: Decode growing utterance
            ASR-->>Stream: Current hypothesis
            Stream-->>API: Partial (stable + tentative)
            API-->>UI: partial JSON event
        end
    end
    Note over User,VAD: A silence endpoint (default 500 ms) closes the utterance.
    Stream->>ASR: Decode complete utterance
    ASR-->>API: Final transcript + ASR timing
    API-->>UI: final JSON event
    API->>Memory: Apply learned corrections, if configured
    Memory->>Cleaner: Clean transcript with style and dictionary
    Cleaner-->>API: Clean result, timing, fallback details
    API-->>UI: clean JSON event
    User->>UI: Click Finish (optional)
    UI->>API: Send "flush"
    API-->>UI: flushed event after pending final work
```

The sequence shows the browser server path. `flowlite live` follows the same
speech pipeline, but reads audio from the local microphone and prints events in
the terminal. `flowlite transcribe` loads a WAV and runs the same endpointing
logic with partial decoding disabled.

## 3. Module hierarchy

```text
flowlite
├── Entry points
│   ├── cli.py                 command parsing and command workflows
│   ├── server.py              browser page, HTTP API, WebSocket API
│   └── static/index.html      browser demo UI
├── Speech pipeline
│   ├── audio_io.py            WAV and PCM conversion, audio chunking
│   ├── streaming.py           frame buffering, utterance state, partial/final events
│   ├── vad.py                 energy-based speech/silence detection
│   └── asr.py                 faster-whisper backend and shared-model lock
├── Transcript processing
│   ├── personalization.py     local correction counts and replacements
│   └── cleanup.py             passthrough, rules, LLM, validation, fallback
├── Configuration and evaluation
│   ├── config.py              environment settings and backend factories
│   ├── bench.py               benchmark loading, execution, and report writing
│   └── metrics.py             edit quality and latency calculations
├── data/
│   ├── samples.jsonl          text cleanup benchmark examples
│   ├── dictionary.txt         optional preferred spellings
│   └── audio/                 optional user-recorded benchmark WAVs
├── scripts/record_dataset.py  record benchmark prompts
└── tests/                     unit and integration tests
```

## 4. Runtime and deployment view

```mermaid
flowchart LR
    Browser["Browser<br/>localhost:8000"]
    Server["Flowlite / FastAPI<br/>single server process"]
    Whisper["faster-whisper<br/>local CPU or NVIDIA GPU"]
    Cleaner["Flowlite cleaner"]
    Ollama["Ollama or another<br/>OpenAI-compatible service"]
    Rules["Rules fallback"]
    Memory["Correction JSON<br/>on this machine"]

    Browser <-->|"HTTP + WebSocket"| Server
    Server --> Whisper
    Server --> Cleaner
    Cleaner -->|"configured endpoint"| Ollama
    Cleaner -->|"on error or invalid output"| Rules
    Memory -->|"loads corrections"| Server
```

`flowlite serve` starts the backend and serves the browser UI from the same
origin, so a separate frontend development server is not required. The ASR
model runs on the configured device. The LLM can be local (for example Ollama)
or remote if configured; in the remote case, transcript text is sent to that
service for cleanup.

## 5. Other workflows

| Workflow | Components used | Result |
|---|---|---|
| `flowlite clean "..."` | CLI → correction memory → selected cleaner | Cleaned text; no audio or ASR |
| `POST /clean` | Browser/API → selected cleaner | JSON cleanup result; no audio or ASR |
| `flowlite teach wrong right` | CLI → correction memory | A locally stored correction, active after two repetitions |
| `flowlite bench-cleanup` | Samples → cleaner backends → metrics | Text quality/latency reports |
| `flowlite bench-e2e` | WAVs → streaming/VAD/ASR → cleaners → metrics | End-to-end quality and latency reports |
| `flowlite doctor` | CLI → dependency imports and LLM health check | Environment diagnostics |
