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
pip install -e "backend[test]"
npm --prefix frontend install
cp backend/.env.example backend/.env
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

## Local Speech Recognition

O ASR roda localmente com `faster-whisper`; o LM Studio continua sendo usado só para o LLM.

Instalação:

```bash
source .venv/bin/activate
pip install -e "backend[test]"
```

Configuração principal em `backend/.env`:

```env
WHISPER_MODEL=large-v3-turbo
WHISPER_DEVICE=auto
WHISPER_COMPUTE_TYPE=auto
WHISPER_LANGUAGE=
ASR_MIN_SPEECH_MS=500
ASR_SILENCE_END_MS=600
ASR_MAX_UTTERANCE_MS=15000
SAVE_RAW_AUDIO=false
```

Com `WHISPER_LANGUAGE=` vazio, o Whisper detecta o idioma automaticamente. Para forçar, use `pt` ou `en`.

Hardware:

- CPU local: deixe `WHISPER_DEVICE=auto`; sem CUDA detectado, o backend usa `cpu` com `int8`.
- CUDA local: com driver NVIDIA e `nvidia-smi` disponível, `auto` usa `cuda` com `float16`.
- Docker CPU: funciona com a imagem padrão.
- Docker CUDA: requer NVIDIA Container Toolkit e configuração adicional de runtime GPU; o modo local sem Docker é o caminho prioritário.

Verificar status:

```bash
curl http://localhost:8000/api/asr/status
```

Testar arquivo:

```bash
curl -F "file=@sample.wav" http://localhost:8000/api/asr/transcribe
```

A resposta inclui `language`, `segments`, `processing_time_ms`, `audio_duration_seconds` e `real_time_factor`.

Na UI, use `Upload WAV/MP3/M4A/WEBM` para validar o Whisper sem criar uma reunião. A seção `Speech Recognition` mostra provider, modelo, device, status, idioma e latência ASR.

Teste com microfone:

1. Abra o frontend.
2. Clique `Start live meeting`.
3. Autorize captura da aba e microfone.
4. Fale uma frase como `Atualmente utilizamos SAP ECC integrado a um WMS externo.`
5. Ao parar de falar, o backend finaliza uma utterance por silêncio e mostra a latência.

Google Meet e Teams Web usam a mesma estratégia: escolha a aba do Meet/Teams no seletor do Chrome e habilite o áudio da aba. O frontend envia `TAB_AUDIO` para áudio remoto e `MIC` para o microfone local, permitindo mapear `TAB_AUDIO -> REMOTE` e `MIC -> ME` sem diarização.

Logs por sessão ficam em:

```text
backend/data/sessions/{session_id}/
```

Arquivos gerados:

- `audio_metadata.json`
- `transcript.jsonl`
- `utterances.jsonl`
- `asr_metrics.jsonl`
- `decisions.jsonl`
- `visual_contexts.jsonl` (ver [Visual Context Pipeline](#visual-context-pipeline))

Áudio bruto não é salvo por padrão.

Troubleshooting:

- CUDA não detectado: confirme `nvidia-smi`, driver NVIDIA e compatibilidade do CTranslate2.
- Falta de VRAM: use `WHISPER_COMPUTE_TYPE=int8_float16` ou `WHISPER_DEVICE=cpu`.
- Modelo não encontrado: rode com internet na primeira carga para baixar do Hugging Face.
- Permissão de microfone: confira permissões do Chrome para `localhost`.
- Aba sem áudio: selecione uma aba específica, não janela/tela inteira, e marque compartilhar áudio.
- Linux/PipeWire: confirme que o Chrome consegue capturar áudio da aba e que o microfone aparece no seletor do sistema.

Benchmark ASR:

```bash
python benchmarks/run_asr_benchmark.py --audio sample.wav --model large-v3-turbo
python benchmarks/run_asr_benchmark.py --audio sample.wav --model large-v3-turbo --domain-prompt
```

## Visual Context Pipeline

Além do áudio, o live meeting sincroniza contexto visual da tela compartilhada com a transcrição - sem nenhum modelo de visão ligado ainda. Esta etapa constrói só a captura/armazenamento:

```text
Shared Screen (o mesmo MediaStream do getDisplayMedia, sem pedir permissão de novo)
  -> Frame Sampling (ScreenFrameSampler, a cada SCREEN_FRAME_SAMPLE_INTERVAL_MS)
  -> Change Detection (grayscale 64x36, mean abs diff - determinístico, sem IA)
  -> ScreenFrame (metadata: timestamp, dimensões, mime type, change_score)
  -> WebSocket (`screen.frame`, JPEG comprimido em base64, no mesmo /ws/live)
  -> VisualContext (backend; campos semânticos ficam vazios nesta etapa)
  -> MeetingState.recent_visual_contexts (até VISUAL_CONTEXT_MAX_RECENT, mais antigo sai)
```

Pontos importantes:

- O frontend reaproveita o `MediaStream` já usado para o preview e para `TAB_AUDIO` - não abre uma segunda captura de tela.
- `timestamp` do frame usa segundos decorridos desde o início da captura live (mesma família de relógio que `TranscriptSegment.start`/`end`), não `Date.now()`. Isso permite `MeetingState.get_visual_context_near(start, end, tolerance_seconds=5)` encontrar o frame mais próximo de uma utterance.
- Frames quase idênticos são descartados; um "force capture" garante um frame atualizado a cada `SCREEN_FRAME_FORCE_CAPTURE_INTERVAL_MS` mesmo com a tela parada.
- `SAVE_SCREEN_FRAMES=false` (padrão): a imagem passa pelo backend só em memória durante o processamento da mensagem; nunca é gravada em disco e nunca entra no `MeetingState` (só metadata). Com `true`, é salva em `backend/data/sessions/{meeting_id}/frames/frame_000001.jpg`.
- `VisionProvider` (`backend/app/vision/provider.py`) é a interface pronta para um modelo multimodal depois (`Screenshot -> VisionProvider -> VisualContext`). A única implementação hoje é `NullVisionProvider`, que não produz análise nenhuma - `summary`, `visible_text`, `entities`, `systems` e `numbers` ficam vazios/`None` de propósito, nunca inventados.
- Sampling não bloqueia áudio/ASR/LLM: cada frame aceito é processado em uma task assíncrona separada no backend, e o `ScreenFrameSampler` do frontend para sozinho quando a reunião termina, a captura de tela é encerrada pelo usuário, ou a track de vídeo acaba.

Configuração em `backend/.env`:

```env
SCREEN_FRAME_SAMPLING_ENABLED=true
SCREEN_FRAME_SAMPLE_INTERVAL_MS=4000
SCREEN_FRAME_FORCE_CAPTURE_INTERVAL_MS=30000
SCREEN_FRAME_CHANGE_THRESHOLD=0.04
SCREEN_FRAME_MAX_DIMENSION=1280
SCREEN_FRAME_JPEG_QUALITY=0.7
VISUAL_CONTEXT_MAX_RECENT=5
SAVE_SCREEN_FRAMES=false
```

Na UI, o botão de monitor no cabeçalho do preview de tela liga o "Show visual debug": frames sampled/skipped/accepted, o último frame aceito (com thumbnail) e os `VisualContext` recentes recebidos do backend.

Teste manual:

1. Inicie o app e clique `Iniciar captura`.
2. Compartilhe uma janela estática (ex.: um slide parado) e ligue `Show visual debug`.
3. Confirme que poucos frames são aceitos (a maioria fica em "skipped").
4. Troque de janela/slide - um frame deve ser aceito (`change_above_threshold`).
5. Volte para a tela estática - frames voltam a ser descartados.
6. Fale durante esse tempo todo e confirme que a transcrição continua fluida (sem travar).
7. Pare a reunião e confirme, pelos logs (`asr audio debug`/console), que o sampler parou.

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

## Treinar por Contexto para uma Chamada Manual

Para testar se a IA consegue opinar em uma chamada real, use o perfil pronto `Advisor`. Ele não é fine-tuning; é um contexto carregado antes da reunião com papel, objetivos e informações que a IA pode usar.

No frontend:

1. Clique `Load advisor profile`.
2. Confirme que o nome do delegate virou `Advisor`.
3. Abra o `Floating widget`, se quiser acompanhar em modo compacto.
4. Inicie `Start live meeting` ou use `Start replay`.

Perfil salvo em:

```text
backend/profiles/ai_call_advisor_design_review.json
```

Esse perfil dá contexto sobre:

- dashboard web versus floating widget versus extensão Chrome
- captura de áudio em Meet/Teams
- integração com LM Studio
- cautela para não interromper a reunião
- riscos de privacidade, consentimento, latência e confiança
- próximos passos técnicos do produto

Roteiro simples para testar em uma chamada:

```text
Vini: Estamos decidindo se o assistente deve ser só dashboard, floating widget ou extensão do Chrome.
Carol: Minha preocupação é que extensão parece mais produto real, mas talvez aumente muito a complexidade agora.
Vini: Advisor, o que você acha que deveríamos priorizar para o MVP?
```

Resposta esperada: a IA deve reconhecer uma `EXPLICIT_CUE` e sugerir priorizar o floating widget como MVP, deixando extensão para uma fase seguinte.

Outro roteiro para testar opinião espontânea:

```text
Vini: O widget flutuante já funciona, mas ainda não tem aprovação antes de falar por mim.
Carol: Também falta uma forma clara de auditar o que a IA sugeriu durante a reunião.
```

Resposta esperada: a IA pode classificar como `CHIME_IN` e sugerir que o produto fique na fase Assist, com sugestões aprovadas pelo usuário e audit trail, antes de qualquer fala autônoma.

Se quiser perguntar pelo chat do floating widget:

```text
Qual seria o próximo passo mais importante para deixar esse MVP confiável?
```

## Testar Upload de Áudio

1. Clique `Upload WAV/MP3/M4A`.
2. Selecione um arquivo.
3. O backend usa `faster-whisper` local e envia cada segmento ao mesmo MeetingEngine.

`ffmpeg` precisa estar disponível no sistema para formatos comprimidos.

## Smoke Test da IA

Use este teste quando quiser validar rapidamente a comunicação backend -> LM Studio -> LLM sem rodar o benchmark completo.

Pré-condições:

- backend rodando em `http://localhost:8000`
- LM Studio com servidor OpenAI-compatible ligado
- um modelo carregado no LM Studio

```bash
npm run smoke:ai
```

O teste faz três coisas:

- chama `/health` e confirma `lm_studio: ok`
- roda um replay curto com uma pergunta explícita para o delegate
- faz uma pergunta ao endpoint `/meetings/{meeting_id}/questions`

Se o backend estiver em outra URL:

```bash
API_URL=http://localhost:8010 npm run smoke:ai
```

Se quiser forçar um modelo específico:

```bash
MODEL=nome-do-modelo npm run smoke:ai
```

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
- ASR `unavailable`: instale com `pip install -e "backend[test]"` e confirme `ffmpeg`/PyAV.
- Captura de aba sem áudio: no Chrome, selecione uma aba, não janela/tela inteira, e marque compartilhar áudio.
- Teams/Meet sem speaker real: o MVP separa `ME` para microfone local e `REMOTE` para áudio da aba.
