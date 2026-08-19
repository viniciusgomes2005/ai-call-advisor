# Meeting Delegate POC

Protótipo experimental de um assistente de IA para reuniões em tempo real inspirado no paper **MEETING DELEGATE: Benchmarking LLMs on Attending Meetings on Our Behalf**.

O core processa uma nova utterance por vez:

`utterance.final -> atualizar contexto -> decidir EXPLICIT_CUE / IMPLICIT_CUE / CHIME_IN / KEEP_SILENCE -> opcionalmente sugerir intervenção`

`KEEP_SILENCE` é uma saída normal e desejável.

## Instalação Local

### 1. Backend

```bash
cd backend
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[test,asr]"
cp .env.example .env
```

### 2. LM Studio

1. Abra o LM Studio.
2. Baixe/carregue um modelo local.
3. Inicie o servidor local OpenAI-compatible.
4. Confirme que ele responde em `http://localhost:1234/v1/models`.
5. Deixe `LLM_MODEL=` vazio para selecionar pela UI ou preencha com o model id.

### 3. Iniciar Backend

```bash
cd backend
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

Cheque:

```bash
curl http://localhost:8000/health
```

### 4. Frontend

```bash
cd frontend
npm install
npm run dev
```

Acesse `http://localhost:5173`.

## Testar Replay

1. Abra o frontend.
2. Confirme LM Studio `ok`.
3. Selecione um modelo carregado.
4. Use o JSON de replay padrão ou cole outro.
5. Clique `Start replay`.
6. Observe a transcrição incremental e todas as decisões.

O backend nunca envia falas futuras ao prompt; há teste automatizado para isso.

## Testar Entrada Manual

1. Crie uma reunião pela primeira utterance manual.
2. Informe `Speaker` e texto.
3. Clique no botão de envio.
4. A decisão aparece no painel direito e é persistida em `data/sessions/{meeting_id}`.

## Testar Upload de Áudio

1. Clique `Upload WAV/MP3/M4A`.
2. Selecione um arquivo.
3. O backend usa `faster-whisper` local e envia cada segmento ao mesmo MeetingEngine.

`ffmpeg` precisa estar disponível no sistema para formatos comprimidos.

## Testar Google Meet

1. Abra o Google Meet no Chrome.
2. Entre em uma reunião de teste.
3. Abra `http://localhost:5173`.
4. Clique `Start live meeting`.
5. Na seleção do Chrome, escolha especificamente a aba do Google Meet.
6. Marque o compartilhamento de áudio da aba.
7. Autorize o microfone local.
8. Observe eventos de áudio, transcrição, utterances e decisões.

O MVP usa captura de aba do Chrome, não Google Meet API.

## Testar Teams Web

Use o mesmo procedimento com uma aba do Microsoft Teams Web no Chrome. O core não conhece Teams ou Meet; ele recebe eventos genéricos como `utterance.final`.

## Benchmark

```bash
cd backend
source .venv/bin/activate
cd ..
python benchmarks/run_benchmark.py
```

Resultados são salvos em `benchmarks/results/latest.json` com response rate, silence rate, category accuracy, precision/recall e latências.

## Docker

LM Studio continua rodando no host.

```bash
docker compose up --build
```

No Docker, o backend usa por padrão `http://host.docker.internal:1234/v1`. Em Linux, o `docker-compose.yml` inclui `extra_hosts: host-gateway`.

## Troubleshooting

- `lm_studio: error`: inicie o servidor local no LM Studio e confirme a porta `1234`.
- Nenhum modelo aparece: carregue um modelo no LM Studio antes de abrir o frontend.
- ASR `unavailable`: instale com `pip install -e ".[asr]"` e confirme `ffmpeg`.
- Captura de aba sem áudio: no Chrome, selecione uma aba, não janela/tela inteira, e marque compartilhar áudio.
- Teams/Meet sem speaker real: o MVP separa `ME` para microfone local e `REMOTE` para áudio da aba; diarização perfeita fica fora deste protótipo.

