from writers import format_srt_time, to_srt, to_txt, write_outputs

SEGMENTS = [
    {"start": 0.0, "end": 2.5, "text": " Hello world."},
    {"start": 2.5, "end": 2.5, "text": "   "},
    {"start": 3661.2346, "end": 3662.0, "text": "你好"},
]


def test_format_srt_time():
    assert format_srt_time(0) == "00:00:00,000"
    assert format_srt_time(3661.2346) == "01:01:01,235"
    assert format_srt_time(-1) == "00:00:00,000"


def test_srt_skips_empty_and_numbers_sequentially():
    assert to_srt(SEGMENTS) == (
        "1\n00:00:00,000 --> 00:00:02,500\nHello world.\n"
        "\n"
        "2\n01:01:01,235 --> 01:01:02,000\n你好\n"
    )


def test_txt_one_line_per_segment():
    assert to_txt(SEGMENTS) == "Hello world.\n你好\n"


def test_write_outputs_never_overwrites(tmp_path):
    first = write_outputs(tmp_path, "会议", SEGMENTS)
    second = write_outputs(tmp_path, "会议", SEGMENTS)
    assert first["txt"].name == "会议.txt"
    assert second["txt"].name == "会议_2.txt"
    assert second["srt"].name == "会议_2.srt"
    assert first["txt"].read_bytes().startswith(b"\xef\xbb\xbf")
    assert not first["srt"].read_bytes().startswith(b"\xef\xbb\xbf")
