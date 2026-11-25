import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SOS_ID = 0
EOS_ID = 0


class Attention(nn.Module):
    def __init__(self, input_dim, source_dim=None, output_dim=None, bias=False):
        super(Attention, self).__init__()
        if source_dim is None:
            source_dim = input_dim
        if output_dim is None:
            output_dim = input_dim
        self.input_dim = input_dim
        self.source_dim = source_dim
        self.output_dim = output_dim
        self.input_proj = nn.Linear(input_dim, source_dim, bias=bias)
        self.output_proj = nn.Linear(input_dim + source_dim, output_dim, bias=bias)
        self.mask = None
    
    def set_mask(self, mask):
        self.mask = mask
    
    def forward(self, input, source_hids):
        batch_size = input.size(0)
        source_len = source_hids.size(1)

        # (batch, tgt_len, input_dim) -> (batch, tgt_len, source_dim)
        x = self.input_proj(input)

        # (batch, tgt_len, source_dim) * (batch, src_len, source_dim) -> (batch, tgt_len, src_len)
        attn = torch.bmm(x, source_hids.transpose(1, 2))
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn = F.softmax(attn.view(-1, source_len), dim=1).view(batch_size, -1, source_len)
        
        # (batch, tgt_len, src_len) * (batch, src_len, source_dim) -> (batch, tgt_len, source_dim)
        mix = torch.bmm(attn, source_hids)
        
        # concat -> (batch, tgt_len, source_dim + input_dim)
        combined = torch.cat((mix, input), dim=2)
        # output -> (batch, tgt_len, output_dim)
        output = torch.tanh(self.output_proj(combined.view(-1, self.input_dim + self.source_dim))).view(batch_size, -1, self.output_dim)
        
        return output, attn


class Decoder(nn.Module):
    KEY_ATTN_SCORE = 'attention_score'
    KEY_LENGTH = 'length'
    KEY_SEQUENCE = 'sequence'
    
    def __init__(self,
                 layers,
                 vocab_size,
                 hidden_size,
                 dropout,
                 length,
                 encoder_length,
                 num_nodes=5,
                 ):
        super(Decoder, self).__init__()
        self.layers = layers
        self.hidden_size = hidden_size
        self.length = length
        self.encoder_length = encoder_length
        self.vocab_size = vocab_size
        self.num_nodes = num_nodes
        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, self.layers, batch_first=True, dropout=dropout)
        self.sos_id = SOS_ID
        self.eos_id = EOS_ID
        self.init_input = None
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.attention = Attention(self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.vocab_size)
    
    def forward_step(self, x, hidden, encoder_outputs):
        batch_size = x.size(0)
        output_size = x.size(1)
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        output, hidden = self.rnn(embedded, hidden)
        output, attn = self.attention(output, encoder_outputs)
        
        predicted_softmax = F.log_softmax(self.out(output.contiguous().view(-1, self.hidden_size)), dim=1)
        predicted_softmax = predicted_softmax.view(batch_size, output_size, -1)
        return predicted_softmax, hidden, attn
    
    def forward(self, x, encoder_hidden=None, encoder_outputs=None):
        ret_dict = dict()
        ret_dict[Decoder.KEY_ATTN_SCORE] = list()
        if x is None:
            inference = True
        else:
            inference = False
        x, batch_size, length = self._validate_args(x, encoder_hidden, encoder_outputs)
        assert length == self.length
        decoder_hidden = self._init_state(encoder_hidden)
        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([length] * batch_size)
        
        # Compute dynamic ranges based on decoder structure
        num_nodes = self.num_nodes  # nodes per cell (NOT length//4, since length covers 2 cells)
        num_inputs = 2  # two cell inputs (s0, s1)
        # Input indices are encoded as 1..num_nodes+1, so ops start at num_nodes+2.
        op_start = num_nodes + 2
        
        def decode(step, step_output, step_attn):
            decoder_outputs.append(step_output)
            ret_dict[Decoder.KEY_ATTN_SCORE].append(step_attn)
            if step % 2 == 0:  # sample node index
                # Node position within current cell: each node has 2 index steps
                node_in_cell = step // 2 % (2 * num_nodes) // 2
                # Node i can reference: 2 inputs + i previous nodes = i+2 options
                # Encoded as token IDs 1 to (num_inputs + node_in_cell)
                max_token = num_inputs + node_in_cell + 1  # exclusive upper bound
                symbols = decoder_outputs[-1][:, 1:max_token].topk(1)[1] + 1
            else:  # sample operation
                symbols = decoder_outputs[-1][:, op_start:].topk(1)[1] + op_start
            
            sequence_symbols.append(symbols)
            
            eos_batches = symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)
            return symbols
        
        decoder_input = x[:, 0].unsqueeze(1)
        for di in range(length):
            if not inference:
                decoder_input = x[:, di].unsqueeze(1)
            decoder_output, decoder_hidden, step_attn = self.forward_step(decoder_input, decoder_hidden, encoder_outputs)
            step_output = decoder_output.squeeze(1)
            symbols = decode(di, step_output, step_attn)
            decoder_input = symbols
        
        ret_dict[Decoder.KEY_SEQUENCE] = sequence_symbols
        ret_dict[Decoder.KEY_LENGTH] = lengths.tolist()
        
        return decoder_outputs, decoder_hidden, ret_dict
    
    def _init_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([h for h in encoder_hidden])
        else:
            encoder_hidden = encoder_hidden
        return encoder_hidden
    
    def _validate_args(self, x, encoder_hidden, encoder_outputs):
        if encoder_outputs is None:
            raise ValueError("Argument encoder_outputs cannot be None when attention is used.")
        
        # inference batch size
        if x is None and encoder_hidden is None:
            batch_size = 1
        else:
            if x is not None:
                batch_size = x.size(0)
            else:
                batch_size = encoder_hidden[0].size(1)
        
        # set default input and max decoding length
        if x is None:
            x = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1).cuda()
            max_length = self.length
        else:
            max_length = x.size(1)
        
        return x, batch_size, max_length
    
    def eval(self):
        return
    
    def infer(self, x, encoder_hidden=None, encoder_outputs=None):
        decoder_outputs, decoder_hidden, _ = self(x, encoder_hidden, encoder_outputs)
        return decoder_outputs, decoder_hidden
