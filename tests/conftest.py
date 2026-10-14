import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

TRAIN_SENTENCES = [
    "The sky is blue and the grass is green.",
    "A man mixes flour and water in a large bowl.",
    "She opened the jar by tapping the lid on the table.",
    "Plants need water, light and air to grow.",
    "He walked to the market and bought some bread.",
    "The children played in the garden until it got dark.",
    "To make tea, first boil some water in a kettle.",
    "The train arrived late because of the snow.",
    "Hello world.",
    "Trailing space here and then everything changed.",
    "Because of the weather in the middle of the night.",
]


@pytest.fixture(scope="session")
def tokenizer():
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=300,
        special_tokens=["<eos>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(iter(TRAIN_SENTENCES), trainer)
    assert tok.token_to_id("<eos>") is not None
    return tok
