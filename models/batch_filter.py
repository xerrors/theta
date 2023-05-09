import torch

def list_save_index(l, value):
    try:
        return l.index(value)
    except:
        return -1


def batch_filter(batch, sep_id, pad_id, hidden_state=None):
    input_ids, attention_mask, pos, triples, ent_maps, sent_mask, _ = batch

    batch_size = input_ids.shape[0]

    device = input_ids.device
    input_ids = input_ids.cpu()
    attention_mask = attention_mask.cpu()
    pos = pos.cpu()
    triples = triples.cpu()
    ent_maps = ent_maps.cpu()
    sent_mask = sent_mask.cpu()
    if hidden_state is not None:
        hidden_state = hidden_state.cpu()

    f_input_ids = []
    f_attention_mask = []
    f_pos = []
    f_triples = []
    f_ent_maps = []
    f_sent_mask = []
    f_hidden_state = []
    for b in range(batch_size):
        sent_start, sent_end = pos[b, 0], pos[b, 1]
        sep_token_idx = input_ids[b].tolist().index(sep_id)
        ids = [0] + list(range(sent_start, sent_end)) + [sep_token_idx]
        f_input_ids.append(input_ids[b, ids])
        f_attention_mask.append(attention_mask[b, ids])
        f_pos.append(pos[b] - sent_start + 1)
        for t in triples[b]:
            t[0] = list_save_index(ids, t[0])
            t[1] = list_save_index(ids, t[1])
            t[2] = list_save_index(ids, t[2])
            t[3] = list_save_index(ids, t[3])
        f_triples.append(triples[b])
        f_ent_maps.append(ent_maps[b, ids])
        # f_sent_mask.append(sent_mask[b, ids])
        if hidden_state is not None:
            f_hidden_state.append(hidden_state[b, ids])

    # padding
    max_len = max([len(f_input_ids[i]) for i in range(batch_size)])

    for b in range(batch_size):
        zero_pad = torch.zeros(max_len - len(f_input_ids[b]), dtype=torch.long)

        if hidden_state is not None:
            hidden_size = hidden_state.shape[-1]
            zero_state_pad = torch.zeros(max_len - len(f_input_ids[b]), hidden_size)
            f_hidden_state[b] = torch.cat([f_hidden_state[b], zero_state_pad], dim=0)

        f_input_ids[b] = torch.cat([f_input_ids[b], zero_pad.fill_(pad_id)])
        f_attention_mask[b] = torch.cat([f_attention_mask[b], zero_pad])
        f_ent_maps[b] = torch.cat([f_ent_maps[b], zero_pad])

    f_input_ids = torch.stack(f_input_ids, dim=0)
    f_attention_mask = torch.stack(f_attention_mask, dim=0)
    f_pos = torch.stack(f_pos, dim=0)
    f_triples = torch.stack(f_triples, dim=0)
    f_ent_maps = torch.stack(f_ent_maps, dim=0)
    # f_sent_mask = torch.stack(f_sent_mask, dim=0)

    assert f_input_ids.shape == f_attention_mask.shape

    f_input_ids = f_input_ids.to(device)
    f_attention_mask = f_attention_mask.to(device)
    f_pos = f_pos.to(device)
    f_triples = f_triples.to(device)
    f_ent_maps = f_ent_maps.to(device)
    # f_sent_mask = f_sent_mask.to(device)

    if hidden_state is not None:
        f_hidden_state = torch.stack(f_hidden_state, dim=0)
        f_hidden_state = f_hidden_state.to(device)
        return (f_input_ids, f_attention_mask, f_pos, f_triples, f_ent_maps, f_sent_mask), f_hidden_state
    else:
        return (f_input_ids, f_attention_mask, f_pos, f_triples, f_ent_maps, f_sent_mask)
