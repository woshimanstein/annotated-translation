import torch

def greedy_search(decoder, encoder_output, attention_mask, h, c, max_length=100, bos_token=0, eos_token=2):
    cur_token = bos_token

    res = [bos_token]
    for _ in range(max_length):
        decoder_input = encoder_output.new_full((1, encoder_output.shape[1]), cur_token, dtype=torch.long)

        logits, h, c, _ = decoder(decoder_input, h, c, encoder_output, attention_mask, train=False)
        # logits.shape: (1, vocab_size)

        # greedy decoding
        cur_token = torch.argmax(logits.squeeze()).item()
        res.append(cur_token)

        # exit if </s> is produced
        if cur_token == eos_token:
            break

    return res