from __future__ import annotations

from datetime import datetime, timezone

from app.meeting.utterance_assembler import (
    TranscriptOrderingBuffer,
    UtteranceAssembler,
    UtteranceAssemblerConfig,
)
from app.schemas import AudioFinalizationReason, TranscriptSegment


def make_segment(
    *,
    id: str,
    source: str = "MIC",
    start: float,
    end: float,
    text: str,
    reason: AudioFinalizationReason | None = AudioFinalizationReason.SILENCE,
) -> TranscriptSegment:
    return TranscriptSegment(
        id=id,
        source=source,
        start=start,
        end=end,
        text=text,
        created_at=datetime.now(timezone.utc),
        finalization_reason=reason,
    )


def test_merges_incomplete_sentence_across_small_gap():
    assembler = UtteranceAssembler(UtteranceAssemblerConfig(merge_max_gap_ms=1200))

    r1 = assembler.push(make_segment(id="seg_0", start=0.0, end=1.5, text="Hoje usamos ECC para"))
    assert r1.updated is not None and not r1.finalized

    r2 = assembler.push(make_segment(id="seg_1", start=1.9, end=3.2, text="controlar a parte financeira."))
    assert r2.updated is not None and not r2.finalized  # merged, buffer still open

    final = assembler.flush_source("MIC")
    assert len(final.finalized) == 1
    semantic = final.finalized[0]
    assert semantic.text == "Hoje usamos ECC para controlar a parte financeira."
    assert semantic.segment_count == 2
    assert semantic.segment_ids == ["seg_0", "seg_1"]


def test_large_gap_after_terminal_punctuation_splits_into_two_utterances():
    assembler = UtteranceAssembler(UtteranceAssemblerConfig(merge_max_gap_ms=1200))

    assembler.push(make_segment(id="seg_0", start=0.0, end=1.0, text="Hoje usamos ECC."))
    result = assembler.push(make_segment(id="seg_1", start=3.5, end=5.0, text="Agora falando sobre logística."))

    assert len(result.finalized) == 1
    assert result.finalized[0].text == "Hoje usamos ECC."
    assert result.finalized[0].segment_count == 1

    final = assembler.flush_source("MIC")
    assert len(final.finalized) == 1
    assert final.finalized[0].text == "Agora falando sobre logística."


def test_merge_prefers_incomplete_sentence_near_gap_limit():
    assembler = UtteranceAssembler(UtteranceAssemblerConfig(merge_max_gap_ms=1000))

    assembler.push(make_segment(id="seg_0", start=0.0, end=1.0, text="O sistema se integra com"))
    # gap of 1400ms: over merge_max_gap_ms (1000) but within the 1.5x tolerance for
    # a sentence that clearly looks incomplete.
    result = assembler.push(make_segment(id="seg_1", start=2.4, end=3.5, text="um WMS externo."))

    assert not result.finalized
    assert result.updated is not None

    final = assembler.flush_source("MIC")
    assert final.finalized[0].text == "O sistema se integra com um WMS externo."


def test_mic_and_tab_audio_are_never_merged():
    assembler = UtteranceAssembler(UtteranceAssemblerConfig(merge_max_gap_ms=1200))

    assembler.push(make_segment(id="seg_mic_0", source="MIC", start=0.0, end=1.0, text="Do lado do mic"))
    assembler.push(make_segment(id="seg_tab_0", source="TAB_AUDIO", start=0.2, end=1.2, text="Do lado da tela"))

    result = assembler.flush_all(reason="meeting_end")
    texts_by_source = {semantic.source: semantic.text for semantic in result.finalized}
    assert texts_by_source == {"MIC": "Do lado do mic", "TAB_AUDIO": "Do lado da tela"}


def test_hard_duration_limit_finalizes_even_mid_merge():
    assembler = UtteranceAssembler(UtteranceAssemblerConfig(merge_max_gap_ms=5000, hard_max_duration_ms=2000))

    assembler.push(make_segment(id="seg_0", start=0.0, end=1.0, text="primeira parte"))
    result = assembler.push(make_segment(id="seg_1", start=1.1, end=2.5, text="segunda parte"))

    assert len(result.finalized) == 1
    assert result.finalized[0].assembly_reason == "hard_duration_limit"
    assert result.updated is None


def test_hard_char_limit_finalizes():
    assembler = UtteranceAssembler(UtteranceAssemblerConfig(merge_max_gap_ms=5000, hard_max_chars=20))

    assembler.push(make_segment(id="seg_0", start=0.0, end=1.0, text="primeira parte curta"))
    result = assembler.push(make_segment(id="seg_1", start=1.1, end=2.0, text="mais um pedaco de texto"))

    assert len(result.finalized) == 1
    assert result.finalized[0].assembly_reason == "hard_char_limit"


def test_meeting_end_flushes_pending_buffer():
    assembler = UtteranceAssembler()
    assembler.push(make_segment(id="seg_0", start=0.0, end=1.0, text="fala pendente"))

    result = assembler.flush_all(reason="meeting_end")

    assert len(result.finalized) == 1
    assert result.finalized[0].assembly_reason == "meeting_end"


def test_manual_flush_flushes_pending_buffer():
    assembler = UtteranceAssembler()
    assembler.push(make_segment(id="seg_0", start=0.0, end=1.0, text="fala pendente"))

    result = assembler.flush_source("MIC", reason="manual_flush")

    assert len(result.finalized) == 1
    assert result.finalized[0].assembly_reason == "manual_flush"


def test_max_duration_cuts_do_not_trigger_silence_timeout_but_silence_does():
    """The core regression this refactor exists to fix: a person talking continuously
    for 14s produces three acoustic ASR chunks (cut at 6s, 12s by ASR_MAX_UTTERANCE_MS,
    then a real pause). Only the last, real silence, may finalize the semantic utterance -
    the two MAX_DURATION cuts must not."""
    config = UtteranceAssemblerConfig(merge_max_gap_ms=1200, finalization_delay_ms=500)
    assembler = UtteranceAssembler(config)

    assembler.push(
        make_segment(id="seg_0", start=0.0, end=6.0, text="Hoje usamos ECC para", reason=AudioFinalizationReason.MAX_DURATION),
        now=0.0,
    )
    # Even though wall-clock time comfortably passes the finalization delay, the buffer
    # must stay open because the last segment was an artificial MAX_DURATION cut.
    still_open = assembler.flush_expired(now=1.0)
    assert not still_open.finalized

    assembler.push(
        make_segment(
            id="seg_1",
            start=6.0,
            end=12.0,
            text="controlar a parte financeira e",
            reason=AudioFinalizationReason.MAX_DURATION,
        ),
        now=1.05,
    )
    still_open = assembler.flush_expired(now=2.0)
    assert not still_open.finalized

    assembler.push(
        make_segment(id="seg_2", start=12.0, end=14.0, text="também um WMS externo.", reason=AudioFinalizationReason.SILENCE),
        now=2.05,
    )
    # Not yet expired.
    assert not assembler.flush_expired(now=2.1).finalized

    # 500ms later (real silence segment), the buffer is eligible and finalizes as one.
    result = assembler.flush_expired(now=2.6)

    assert len(result.finalized) == 1
    semantic = result.finalized[0]
    assert semantic.text == "Hoje usamos ECC para controlar a parte financeira e também um WMS externo."
    assert semantic.segment_count == 3
    assert semantic.assembly_reason == "silence_timeout"


def test_assembler_disabled_restores_one_utterance_per_fragment():
    assembler = UtteranceAssembler(UtteranceAssemblerConfig(enabled=False))

    r1 = assembler.push(make_segment(id="seg_0", start=0.0, end=1.0, text="fragmento um"))
    r2 = assembler.push(make_segment(id="seg_1", start=1.1, end=2.0, text="fragmento dois"))

    assert len(r1.finalized) == 1
    assert len(r2.finalized) == 1
    assert r1.finalized[0].assembly_reason == "assembler_disabled"


def test_ordering_buffer_reorders_out_of_order_asr_completions():
    buffer = TranscriptOrderingBuffer()
    seg0 = make_segment(id="seg_0", start=0.0, end=1.0, text="primeiro")
    seg1 = make_segment(id="seg_1", start=1.0, end=2.0, text="segundo")

    # sequence 1 (whisper for chunk #11) finishes before sequence 0 (chunk #10, slower).
    ready_from_1 = buffer.push_batch("TAB_AUDIO", 1, [seg1])
    assert ready_from_1 == []  # held back: sequence 0 hasn't arrived yet

    ready_from_0 = buffer.push_batch("TAB_AUDIO", 0, [seg0])
    assert [segment.id for segment in ready_from_0] == ["seg_0", "seg_1"]


def test_ordering_buffer_does_not_block_forever_on_a_failed_sequence():
    buffer = TranscriptOrderingBuffer()
    seg1 = make_segment(id="seg_1", start=1.0, end=2.0, text="segundo")

    # sequence 0 failed / returned empty transcript - still must be marked complete.
    ready = buffer.push_batch("MIC", 0, [])
    assert ready == []

    ready = buffer.push_batch("MIC", 1, [seg1])
    assert [segment.id for segment in ready] == ["seg_1"]


def test_ordering_buffer_keeps_sources_independent():
    buffer = TranscriptOrderingBuffer()
    mic_seg = make_segment(id="seg_mic_0", source="MIC", start=0.0, end=1.0, text="mic")
    tab_seg = make_segment(id="seg_tab_0", source="TAB_AUDIO", start=0.0, end=1.0, text="tab")

    # TAB_AUDIO sequence 0 arrives while MIC sequence 0 never has (no MIC audio yet).
    ready_tab = buffer.push_batch("TAB_AUDIO", 0, [tab_seg])
    assert [segment.id for segment in ready_tab] == ["seg_tab_0"]

    ready_mic = buffer.push_batch("MIC", 0, [mic_seg])
    assert [segment.id for segment in ready_mic] == ["seg_mic_0"]
