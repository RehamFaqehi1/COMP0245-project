"""Small architecture/API smoke test; does not access project datasets or PyBullet."""
import torch
from lstm_model import LSTMPolicy


def main():
    torch.manual_seed(42)
    model = LSTMPolicy(input_size=37, output_size=11, hidden_size=16,
                       num_layers=2, dropout=0.1, sequence_length=8)
    model.eval()
    sequence = torch.randn(1, 8, 37)
    with torch.inference_mode():
        batch_output = model(sequence)
        model.reset_state()
        step_outputs = [model(sequence[:, i, :]) for i in range(sequence.shape[1])]
    assert batch_output.shape == (1, 11)
    assert step_outputs[-1].shape == (1, 11)
    assert torch.allclose(batch_output, step_outputs[-1], atol=1e-6, rtol=1e-5)
    print("LSTM model smoke test passed: batch and causal step APIs agree.")


if __name__ == "__main__":
    main()
