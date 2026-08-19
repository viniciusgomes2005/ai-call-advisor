# Meeting Delegate POC

Protótipo experimental de um assistente de IA para reuniões em tempo real inspirado no paper **MEETING DELEGATE: Benchmarking LLMs on Attending Meetings on Our Behalf**.

O core processa uma nova utterance por vez:

`utterance.final -> atualizar contexto -> decidir EXPLICIT_CUE / IMPLICIT_CUE / CHIME_IN / KEEP_SILENCE -> opcionalmente sugerir intervenção`

`KEEP_SILENCE` é uma saída normal e desejável.

## Requisitos

- Python 3.12+
- Node.js e npm
- `ffmpeg` no sistema para upload/captura de áudio comprimido
- LM Studio rodando localmente para usar uma LLM

## Instalação Local

Rode a instalação a partir da raiz do repositório:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e "backend[test,asr]"
npm --prefix frontend install
cp backend/.env.example backend/.env
```

O extra `asr` instala o `faster-whisper`. Se você não for testar áudio agora, pode instalar apenas:

```bash
pip install -e "backend[test]"
```

## Rodar Frontend e Backend Juntos

Depois da instalação, suba tudo com um comando na raiz:

```bash
npm run dev
```

Esse comando inicia:

- backend FastAPI em `http://localhost:8000`
- frontend Vite em `http://localhost:5173`

Cheque o backend em:

```bash
curl http://localhost:8000/health
```

Se precisar mudar a porta do backend:

```bash
BACKEND_PORT=8010 npm run dev
```

## Rodar Separado

Backend:

```bash
cd backend
../.venv/bin/python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Frontend:

```bash
cd frontend
npm run dev
```

Se o backend estiver em outra URL, informe para o Vite:

```bash
VITE_API_URL=http://localhost:8010 npm --prefix frontend run dev
```

## Plugar a LLM com LM Studio

1. Abra o LM Studio.
2. Baixe e carregue um modelo local.
3. Inicie o servidor local OpenAI-compatible.
4. Confirme que ele responde em `http://localhost:1234/v1/models`.
5. Edite `backend/.env` se precisar mudar a URL, chave ou modelo:

```env
LLM_BASE_URL=http://localhost:1234/v1
LLM_API_KEY=lm-studio
LLM_MODEL=
```

`LLM_MODEL` pode ficar vazio. Nesse caso, o backend lista os modelos carregados no LM Studio e o frontend permite selecionar um modelo na UI. Se quiser fixar um modelo, preencha `LLM_MODEL` com o `id` retornado por `/v1/models`.

O backend usa endpoints OpenAI-compatible:

- `GET /v1/models`
- `POST /v1/chat/completions`

## Conectar com Google Meet

O MVP não usa Google Meet API. A conexão é feita pela captura de aba do Chrome.

1. Abra o Google Meet no Chrome.
2. Entre em uma reunião de teste.
3. Abra `http://localhost:5173`.
4. Clique em `Start live meeting`.
5. Na seleção do Chrome, escolha especificamente a aba do Google Meet.
6. Marque o compartilhamento de áudio da aba.
7. Autorize o microfone local.
8. Observe eventos de áudio, transcrição, utterances e decisões na UI.

O navegador envia dois fluxos:

- `REMOTE_AUDIO`: áudio capturado da aba do Meet
- `LOCAL_MIC_AUDIO`: áudio do seu microfone

O backend rotula esses fluxos como `REMOTE` e `ME`. Diarização real por participante fica fora deste protótipo.

## Conectar com Microsoft Teams

Para Teams Web, use o mesmo caminho de captura de aba:

1. Abra o Teams Web no Chrome.
2. Entre em uma reunião de teste.
3. Abra `http://localhost:5173`.
4. Clique em `Start live meeting`.
5. Na seleção do Chrome, escolha especificamente a aba do Teams.
6. Marque o compartilhamento de áudio da aba.
7. Autorize o microfone local.

Não é necessário Microsoft Graph, registro de bot ou API nativa do Teams para este POC. Uma integração nativa futura deve substituir apenas a origem de mídia e emitir os mesmos eventos internos, principalmente `audio.chunk`, `transcript.partial` e `utterance.final`.

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

## Benchmark

```bash
source .venv/bin/activate
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
- ASR `unavailable`: instale com `pip install -e "backend[asr]"` e confirme `ffmpeg`.
- Captura de aba sem áudio: no Chrome, selecione uma aba, não janela/tela inteira, e marque compartilhar áudio.
- Teams/Meet sem speaker real: o MVP separa `ME` para microfone local e `REMOTE` para áudio da aba.
