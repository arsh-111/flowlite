from conftest import FakeBackend, silence, tone
import numpy as np

from flowlite.streaming import StreamingTranscriber, common_prefix_len


def feed_in_odd_chunks(tr, audio, size=1000):
    events = []
    for i in range(0, len(audio), size):
        events += tr.feed(audio[i:i + size])
    return events


def test_common_prefix():
    assert common_prefix_len(["a", "b", "c"], ["a", "b", "x"]) == 2
    assert common_prefix_len([], ["a"]) == 0


def test_single_utterance_endpointed():
    be = FakeBackend(["hello world"])
    tr = StreamingTranscriber(be, partial_interval_s=None)
    audio = np.concatenate([silence(0.2), tone(0.8), silence(0.8)])
    events = feed_in_odd_chunks(tr, audio)
    finals = [e for e in events if e.kind == "final"]
    assert len(finals) == 1 and finals[0].text == "hello world"
    assert finals[0].audio_s >= 0.8 and finals[0].endpoint_at > 0
    assert be.calls == 1


def test_blip_is_ignored():
    be = FakeBackend(["noise"])
    tr = StreamingTranscriber(be, partial_interval_s=None)
    events = feed_in_odd_chunks(tr, np.concatenate([tone(0.1), silence(1.0)]))
    assert events == [] and be.calls == 0


def test_two_utterances():
    be = FakeBackend(["first", "second"])
    tr = StreamingTranscriber(be, partial_interval_s=None)
    audio = np.concatenate([tone(0.6), silence(0.8), tone(0.6), silence(0.8)])
    finals = [e for e in feed_in_odd_chunks(tr, audio) if e.kind == "final"]
    assert [f.text for f in finals] == ["first", "second"]
    assert [f.utt_id for f in finals] == [0, 1]


def test_partials_use_local_agreement():
    be = FakeBackend(["hello", "hello world", "hello world again", "hello world again today"])
    tr = StreamingTranscriber(be, partial_interval_s=0.5)
    events = feed_in_odd_chunks(tr, np.concatenate([tone(1.6), silence(0.8)]))
    partials = [e for e in events if e.kind == "partial"]
    finals = [e for e in events if e.kind == "final"]
    assert len(partials) >= 2
    assert partials[0].stable == "" and partials[0].tentative == "hello"
    assert partials[1].stable == "hello" and partials[1].tentative == "world"
    assert len(finals) == 1 and finals[0].text.startswith("hello world again")


def test_flush_finalises_open_utterance():
    be = FakeBackend(["cut off mid sentence"])
    tr = StreamingTranscriber(be, partial_interval_s=None)
    assert feed_in_odd_chunks(tr, tone(1.0)) == []
    finals = tr.flush()
    assert len(finals) == 1 and finals[0].text == "cut off mid sentence"


def test_max_utterance_forces_endpoint():
    be = FakeBackend(["long"])
    tr = StreamingTranscriber(be, partial_interval_s=None, max_utterance_s=1.0)
    events = feed_in_odd_chunks(tr, tone(2.0))
    assert any(e.kind == "final" for e in events)
