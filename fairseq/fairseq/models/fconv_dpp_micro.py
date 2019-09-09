# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from fairseq import options, utils
from fairseq.modules import (
    AdaptiveSoftmax,
    BeamableMM,
    GradMultiply,
    LearnedPositionalEmbedding,
    LinearizedConvolution,
)

from . import (
    FairseqEncoder,
    FairseqIncrementalDecoder,
    FairseqModel,
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
    FairseqKLModel,
)


@register_model('fconv_dpp_micro')
class FConvDPPMicroModel(FairseqKLModel):
    """
    A fully convolutional model, i.e. a convolutional encoder and a
    convolutional decoder, as described in `"Convolutional Sequence to Sequence
    Learning" (Gehring et al., 2017) <https://arxiv.org/abs/1705.03122>`_.

    Args:
        encoder (FConvEncoder): the encoder
        decoder (FConvDecoder): the decoder

    The Convolutional model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.fconv_parser
        :prog:
    """

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)
        self.encoder.num_attention_layers = sum(
            layer is not None for layer in decoder.attention)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument(
            '--dropout', type=float, metavar='D', help='dropout probability')
        parser.add_argument(
            '--encoder-embed-dim',
            type=int,
            metavar='N',
            help='encoder embedding dimension')
        parser.add_argument(
            '--encoder-embed-path',
            type=str,
            metavar='STR',
            help='path to pre-trained encoder embedding')
        parser.add_argument(
            '--encoder-layers',
            type=str,
            metavar='EXPR',
            help='encoder layers [(dim, kernel_size), ...]')
        parser.add_argument(
            '--decoder-embed-dim',
            type=int,
            metavar='N',
            help='decoder embedding dimension')
        parser.add_argument(
            '--decoder-embed-path',
            type=str,
            metavar='STR',
            help='path to pre-trained decoder embedding')
        parser.add_argument(
            '--decoder-layers',
            type=str,
            metavar='EXPR',
            help='decoder layers [(dim, kernel_size), ...]')
        parser.add_argument(
            '--decoder-out-embed-dim',
            type=int,
            metavar='N',
            help='decoder output embedding dimension')
        parser.add_argument(
            '--decoder-attention',
            type=str,
            metavar='EXPR',
            help='decoder attention [True, ...]')
        parser.add_argument(
            '--share-input-output-embed',
            action='store_true',
            help='share input and output embeddings (requires'
            ' --decoder-out-embed-dim and --decoder-embed-dim'
            ' to be equal)')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure that all args are properly defaulted (in case there are any new ones)
        base_architecture(args)

        encoder_embed_dict = None
        if args.encoder_embed_path:
            encoder_embed_dict = utils.parse_embedding(args.encoder_embed_path)
            utils.print_embed_overlap(encoder_embed_dict,
                                      task.source_dictionary)

        decoder_embed_dict = None
        if args.decoder_embed_path:
            decoder_embed_dict = utils.parse_embedding(args.decoder_embed_path)
            utils.print_embed_overlap(decoder_embed_dict,
                                      task.target_dictionary)

        encoder = FConvEncoder(
            dictionary=task.source_dictionary,
            embed_dim=args.encoder_embed_dim,
            embed_dict=encoder_embed_dict,
            convolutions=eval(args.encoder_layers),
            dropout=args.dropout,
            max_positions=args.max_source_positions,
        )
        decoder = FConvDecoder(
            dictionary=task.target_dictionary,
            embed_dim=args.decoder_embed_dim,
            embed_dict=decoder_embed_dict,
            convolutions=eval(args.decoder_layers),
            out_embed_dim=args.decoder_out_embed_dim,
            attention=eval(args.decoder_attention),
            dropout=args.dropout,
            max_positions=args.max_target_positions,
            share_embed=args.share_input_output_embed,
        )
        return FConvDPPMicroModel(encoder, decoder)


class FConvEncoder(FairseqEncoder):
    """
    Convolutional encoder consisting of `len(convolutions)` layers.
    Args:
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_dim (int, optional): embedding dimension
        embed_dict (str, optional): filename from which to load pre-trained
            embeddings
        max_positions (int, optional): maximum supported input sequence length
        convolutions (list, optional): the convolutional layer structure. Each
            list item `i` corresponds to convolutional layer `i`. Layers are
            given as ``(out_channels, kernel_width, [residual])``. Residual
            connections are added between layers when ``residual=1`` (which is
            the default behavior).
        dropout (float, optional): dropout to be applied before each conv layer
    """

    def __init__(
            self,
            dictionary,
            embed_dim=512,
            embed_dict=None,
            max_positions=1024,
            convolutions=((512, 3), ) * 20,
            dropout=0.1,
    ):
        super().__init__(dictionary)
        self.dropout = dropout
        self.num_attention_layers = None

        num_embeddings = len(dictionary)
        self.padding_idx = dictionary.pad()
        self.embed_tokens = Embedding(num_embeddings, embed_dim,
                                      self.padding_idx)
        if embed_dict:
            self.embed_tokens = utils.load_embedding(
                embed_dict, self.dictionary, self.embed_tokens)

        self.embed_positions = PositionalEmbedding(
            max_positions,
            embed_dim,
            self.padding_idx,
        )

        convolutions = extend_conv_spec(convolutions)
        in_channels = convolutions[0][0]
        self.fc1 = Linear(embed_dim, in_channels, dropout=dropout)
        self.projections = nn.ModuleList()
        self.convolutions = nn.ModuleList()
        self.residuals = []

        layer_in_channels = [in_channels]
        for _, (out_channels, kernel_size,
                residual) in enumerate(convolutions):
            if residual == 0:
                residual_dim = out_channels
            else:
                residual_dim = layer_in_channels[-residual]
            self.projections.append(
                Linear(residual_dim, out_channels
                       ) if residual_dim != out_channels else None)
            if kernel_size % 2 == 1:
                padding = kernel_size // 2
            else:
                padding = 0
            self.convolutions.append(
                ConvTBC(
                    in_channels,
                    out_channels * 2,
                    kernel_size,
                    dropout=dropout,
                    padding=padding))
            self.residuals.append(residual)
            in_channels = out_channels
            layer_in_channels.append(out_channels)
        self.fc2 = Linear(in_channels, embed_dim)

    def forward(self, src_tokens, src_lengths):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): lengths of each source sentence of shape
                `(batch)`
        Returns:
            dict:
                - **encoder_out** (tuple): a tuple with two elements, where the
                  first element is the last encoder layer's output and the
                  second element is the same quantity summed with the input
                  embedding (used for attention). The shape of both tensors is
                  `(batch, src_len, embed_dim)`.
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        # embed tokens and positions
        x = self.embed_tokens(src_tokens) + self.embed_positions(src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)
        input_embedding = x

        # project to size of convolution
        x = self.fc1(x)

        # used to mask padding in input
        encoder_padding_mask = src_tokens.eq(self.padding_idx).t()  # -> T x B
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        residuals = [x]
        # temporal convolutions
        for proj, conv, res_layer in zip(self.projections, self.convolutions,
                                         self.residuals):
            if res_layer > 0:
                residual = residuals[-res_layer]
                residual = residual if proj is None else proj(residual)
            else:
                residual = None

            if encoder_padding_mask is not None:
                x = x.masked_fill(encoder_padding_mask.unsqueeze(-1), 0)

            x = F.dropout(x, p=self.dropout, training=self.training)
            if conv.kernel_size[0] % 2 == 1:
                # padding is implicit in the conv
                x = conv(x)
            else:
                padding_l = (conv.kernel_size[0] - 1) // 2
                padding_r = conv.kernel_size[0] // 2
                x = F.pad(x, (0, 0, 0, 0, padding_l, padding_r))
                x = conv(x)
            x = F.glu(x, dim=2)

            if residual is not None:
                x = (x + residual) * math.sqrt(0.5)
            residuals.append(x)

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)

        # project back to size of embedding
        x = self.fc2(x)

        if encoder_padding_mask is not None:
            encoder_padding_mask = encoder_padding_mask.t()  # -> B x T
            x = x.masked_fill(encoder_padding_mask.unsqueeze(-1), 0)

        # scale gradients (this only affects backward, not forward)
        x = GradMultiply.apply(x, 1.0 / (2.0 * self.num_attention_layers))

        # add output to input embedding for attention
        y = (x + input_embedding) * math.sqrt(0.5)

        return {
            'encoder_out': (x, y),
            'encoder_padding_mask': encoder_padding_mask,  # B x T
            'encoder_input_embedding': input_embedding,
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = (
                encoder_out['encoder_out'][0].index_select(0, new_order),
                encoder_out['encoder_out'][1].index_select(0, new_order),
            )
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.embed_positions.max_positions()


class DppMicroAttentionLayer(nn.Module):
    def __init__(self, conv_channels, embed_dim, bmm=None):
        super().__init__()
        # projects from output of convolution to embedding dimension
        self.in_projection = Linear(conv_channels, embed_dim)
        # projects from embedding dimension to convolution size
        self.out_projection = Linear(embed_dim, conv_channels)
        USE_CUDA = torch.cuda.is_available()
        self.device = torch.device("cuda" if USE_CUDA else "cpu")
        self.bmm = bmm if bmm is not None else torch.bmm
        self.kl_loss = torch.nn.KLDivLoss(reduction="sum")

    def BFGMInference(self, matrix_l, max_samples):
        with torch.no_grad():
            count = 1 # counts of chosen elements
            N = matrix_l.size()[1] # counts of elements
            B = matrix_l.size()[0] # batch size
            C = torch.zeros(B, N, 1).to(self.device)  # extendable vector set

            index = torch.LongTensor().to(self.device)
            for _ in range(B):
                index = torch.cat([index, torch.linspace(0, N - 1, N).long().to(self.device)])
            index = index.view(B, 1, -1)  # index for select diagnol elements
            
            D = matrix_l.gather(1, index).view(B, -1)  # diagnol elements(quality) set
            mask = torch.ones(B, N).to(self.device) # chosen elements mask tensor which represent chosen elements as 0
            J = torch.argmax(torch.log(D * mask), dim=1)  # current chosen elements set
            mask = mask.scatter_(1, J.view(-1, 1), 0.0) # mask chosen elements
            
            increment = torch.LongTensor([]).to(self.device)
            for idx in range(B):
                increment = torch.cat([increment, torch.LongTensor([idx * N]).to(self.device)]) # for transforming index batchwise to non batchwise
                
            while count < max_samples:
                candidate = torch.nonzero(mask)[:, 1].view(B, -1)  # unchosen elements
                c_extend = torch.zeros(B, N, 1).to(self.device) # changements record tensor for c
                d_minus = torch.zeros(B, N).to(self.device) # changements record tensor for d
                
                for idx in range(N - count): # iterate all cadidate elements
                    i = candidate[:, idx] + increment 
                    j = J + increment # get one candidate index batchwise
                    
                    temp = matrix_l.contiguous().view(B * N, -1)
                    temp = torch.index_select(temp, 0, j) 
                    matrix_select = torch.index_select(temp.view(1, -1), 1, i).view(-1) # make shape transformation twice to select tensor items based on index matrix
                    
                    c_select_i = torch.index_select(C.contiguous().view(B * N,-1), 0, i).view(B, -1, 1)
                    c_select_j = torch.index_select(C.contiguous().view(B * N,-1), 0, j).view(B, -1, 1) # make shape transformation once to select c elements(vector) based on index

                    c_dot = torch.bmm(c_select_i.permute(0,2,1), c_select_j).view(-1) # compute tensor(vector) inner product
                    
                    d_select_j =D.gather(1, J.view(B,-1)).view(-1) # select elements in d based on index j
                    
                    e = (matrix_select - c_dot) / d_select_j # compute e according to formula
                    
                    c_extend.scatter_(1, (i - increment).view(B, 1, 1), e.view(B, 1, 1)) # store extended part for c
                    d_minus.scatter_(1, (i - increment).view(B, 1), (e*e).view(B, 1)) # store minus part for d
                
                C = torch.cat([C, c_extend], dim=2)
                D = D - d_minus # apply changements after iteration            
                J = torch.argmax(torch.log(D * mask), dim=1)  # select max elements based on modified D excluding chosen elements
                mask = mask.scatter_(1, J.view(-1, 1), 0.0) # mask chosen elements
                count += 1

            return torch.nonzero(1 - mask)[:, 1].view(B, -1)

    def createGM(self, mu_list, sigma, length):
        with torch.no_grad():
            l = mu_list.size()[1]
            batch = mu_list.size()[0]
            pos = torch.linspace(0,length-1,length).view(1, length).repeat(batch, 1).float().to(self.device)
            distribution = torch.zeros(batch,length).to(self.device)
            for i in range(l):
                mu = mu_list[:,i].view(batch, -1).repeat(1,length).float()
                distribution += torch.exp(-(pos-mu)**2 / (2*sigma**2)) / (math.sqrt(2*math.pi)*sigma)
        return distribution
    
    def get_matrix(self, attn, sim_vec, stride):
        with torch.no_grad():
            random_start = np.random.randint(stride)
            batch, ts, e = sim_vec.size()

            norm_embeddings = torch.norm(sim_vec, dim=2).view(batch, ts, 1)
            normalized_embeddings = sim_vec / norm_embeddings # [B, Ts, E]

            attn_avg = torch.mean(attn, dim=1).view(batch, ts, 1) # [B, Ts, 1]
            sim_matrix_l = torch.bmm(normalized_embeddings[:, random_start::stride, :],normalized_embeddings[:, random_start::stride, :].permute(0, 2, 1)) # [B, Ts', Ts']
            quality_matrix_l = torch.bmm(attn_avg[:, random_start::stride, :], attn_avg[:, random_start::stride, :].permute(0, 2, 1)) # [B, Ts', Ts']

            matrix_l = quality_matrix_l.mul(sim_matrix_l)
        return matrix_l
        

    def forward(self, x, target_embedding, encoder_out, encoder_padding_mask):
        residual = x

        # attention
        x = (self.in_projection(x) + target_embedding) * math.sqrt(0.5)
        x = self.bmm(x, encoder_out[0])

        # don't attend over padding
        if encoder_padding_mask is not None:
            x = x.float().masked_fill(
                encoder_padding_mask.unsqueeze(1), float('-inf')).type_as(
                    x)  # FP16 support: cast to float and back

        # softmax over last dim
        sz = x.size()
        x = F.softmax(x.view(sz[0] * sz[1], sz[2]), dim=1)
        x = x.view(sz)

        if self.training:
            stride = 20
            max_samples = 20
            sigma = 3
            # get gaussian mixture distribution to calculate loss
            if encoder_padding_mask is None:
                try:
                    matrix_l = self.get_matrix(x, encoder_out[1], stride)
                    mu_list = self.BFGMInference(matrix_l, max_samples)
                    distribution = self.createGM(mu_list, sigma, sz[2]).contiguous()
                    distribution = F.softmax(distribution, 1)
                    x_mean = torch.mean(x, dim=1)
                    kl_loss = self.kl_loss(distribution, x_mean)
                except RuntimeError:
                    kl_loss = torch.FloatTensor([0]).to(self.device)
            else:
                kl_loss = torch.FloatTensor([0]).to(self.device)
        else:
            kl_loss = torch.FloatTensor([0]).to(self.device)

        attn_scores = x
        # weight the context
        x = self.bmm(x, encoder_out[1])

        # scale attention output (respecting potentially different lengths)
        s = encoder_out[1].size(1)
        if encoder_padding_mask is None:
            x = x * (s * math.sqrt(1.0 / s))
        else:
            s = s - encoder_padding_mask.type_as(x).sum(
                dim=1, keepdim=True)  # exclude padding
            s = s.unsqueeze(-1)
            x = x * (s * s.rsqrt())

        # project back
        x = (self.out_projection(x) + residual) * math.sqrt(0.5)
        return x, attn_scores, -kl_loss

    def make_generation_fast_(self, beamable_mm_beam_size=None, **kwargs):
        """Replace torch.bmm with BeamableMM."""
        if beamable_mm_beam_size is not None:
            del self.bmm
            self.add_module('bmm', BeamableMM(beamable_mm_beam_size))


class FConvDecoder(FairseqIncrementalDecoder):
    """Convolutional decoder"""

    def __init__(
            self,
            dictionary,
            embed_dim=512,
            embed_dict=None,
            out_embed_dim=256,
            max_positions=1024,
            convolutions=((512, 3), ) * 20,
            attention=True,
            dropout=0.1,
            share_embed=False,
            positional_embeddings=True,
            adaptive_softmax_cutoff=None,
            adaptive_softmax_dropout=0,
    ):
        super().__init__(dictionary)
        self.register_buffer('version', torch.Tensor([2]))
        self.dropout = dropout
        self.need_attn = True
        USE_CUDA = torch.cuda.is_available()
        self.device = torch.device("cuda" if USE_CUDA else "cpu")

        convolutions = extend_conv_spec(convolutions)
        in_channels = convolutions[0][0]
        if isinstance(attention, bool):
            # expand True into [True, True, ...] and do the same with False
            attention = [attention] * len(convolutions)
        if not isinstance(attention,
                          list) or len(attention) != len(convolutions):
            raise ValueError(
                'Attention is expected to be a list of booleans of '
                'length equal to the number of layers.')

        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
        if embed_dict:
            self.embed_tokens = utils.load_embedding(
                embed_dict, self.dictionary, self.embed_tokens)

        self.embed_positions = PositionalEmbedding(
            max_positions,
            embed_dim,
            padding_idx,
        ) if positional_embeddings else None

        self.fc1 = Linear(embed_dim, in_channels, dropout=dropout)
        self.projections = nn.ModuleList()
        self.convolutions = nn.ModuleList()
        self.attention = nn.ModuleList()
        self.residuals = []

        layer_in_channels = [in_channels]
        for i, (out_channels, kernel_size,
                residual) in enumerate(convolutions):
            if residual == 0:
                residual_dim = out_channels
            else:
                residual_dim = layer_in_channels[-residual]
            self.projections.append(
                Linear(residual_dim, out_channels
                       ) if residual_dim != out_channels else None)
            self.convolutions.append(
                LinearizedConv1d(
                    in_channels,
                    out_channels * 2,
                    kernel_size,
                    padding=(kernel_size - 1),
                    dropout=dropout))
            self.attention.append(
                DppMicroAttentionLayer(out_channels, embed_dim
                                       ) if attention[i] else None)
            self.residuals.append(residual)
            in_channels = out_channels
            layer_in_channels.append(out_channels)

        self.adaptive_softmax = None
        self.fc2 = self.fc3 = None

        if adaptive_softmax_cutoff is not None:
            assert not share_embed
            self.adaptive_softmax = AdaptiveSoftmax(
                num_embeddings,
                in_channels,
                adaptive_softmax_cutoff,
                dropout=adaptive_softmax_dropout)
        else:
            self.fc2 = Linear(in_channels, out_embed_dim)
            if share_embed:
                assert out_embed_dim == embed_dim, \
                    "Shared embed weights implies same dimensions " \
                    " out_embed_dim={} vs embed_dim={}".format(out_embed_dim, embed_dim)
                self.fc3 = nn.Linear(out_embed_dim, num_embeddings)
                self.fc3.weight = self.embed_tokens.weight
            else:
                self.fc3 = Linear(
                    out_embed_dim, num_embeddings, dropout=dropout)

    def forward(self,
                prev_output_tokens,
                encoder_out_dict=None,
                incremental_state=None):
        if encoder_out_dict is not None:
            encoder_out = encoder_out_dict['encoder_out']
            encoder_padding_mask = encoder_out_dict['encoder_padding_mask']
            encoder_input_embedding = encoder_out_dict['encoder_input_embedding']

            # split and transpose encoder outputs
            encoder_a, encoder_b = self._split_encoder_out(
                encoder_out, incremental_state)

        if self.embed_positions is not None:
            pos_embed = self.embed_positions(prev_output_tokens,
                                             incremental_state)
        else:
            pos_embed = 0

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
        x = self._embed_tokens(prev_output_tokens, incremental_state)

        # embed tokens and combine with positional embeddings
        x += pos_embed
        x = F.dropout(x, p=self.dropout, training=self.training)
        target_embedding = x

        # project to size of convolution
        x = self.fc1(x)

        # B x T x C -> T x B x C
        x = self._transpose_if_training(x, incremental_state)

        # temporal convolutions
        avg_attn_scores = None
        num_attn_layers = len(self.attention)
        residuals = [x]
        kl_loss_total = torch.FloatTensor([0]).to(self.device)
        for proj, conv, attention, res_layer in zip(
                self.projections, self.convolutions, self.attention,
                self.residuals):
            if res_layer > 0:
                residual = residuals[-res_layer]
                residual = residual if proj is None else proj(residual)
            else:
                residual = None

            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, incremental_state)
            x = F.glu(x, dim=2)

            # attention
            if attention is not None:
                x = self._transpose_if_training(x, incremental_state)

                x, attn_scores, kl_loss = attention(x, target_embedding,
                                           (encoder_a, encoder_b),
                                           encoder_padding_mask)
                kl_loss_total += kl_loss

                if not self.training and self.need_attn:
                    attn_scores = attn_scores / num_attn_layers
                    if avg_attn_scores is None:
                        avg_attn_scores = attn_scores
                    else:
                        avg_attn_scores.add_(attn_scores)

                x = self._transpose_if_training(x, incremental_state)

            # residual
            if residual is not None:
                x = (x + residual) * math.sqrt(0.5)
            residuals.append(x)

        # T x B x C -> B x T x C
        x = self._transpose_if_training(x, incremental_state)

        # project back to size of vocabulary if not using adaptive softmax
        if self.fc2 is not None and self.fc3 is not None:
            x = self.fc2(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.fc3(x)

        return x, avg_attn_scores, kl_loss_total

    def reorder_incremental_state(self, incremental_state, new_order):
        super().reorder_incremental_state(incremental_state, new_order)
        encoder_out = utils.get_incremental_state(self, incremental_state,
                                                  'encoder_out')
        if encoder_out is not None:
            encoder_out = tuple(
                eo.index_select(0, new_order) for eo in encoder_out)
            utils.set_incremental_state(self, incremental_state, 'encoder_out',
                                        encoder_out)

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.embed_positions.max_positions(
        ) if self.embed_positions is not None else float('inf')

    def upgrade_state_dict(self, state_dict):
        if utils.item(state_dict.get('decoder.version', torch.Tensor(
            [1]))[0]) < 2:
            # old models use incorrect weight norm dimension
            for i, conv in enumerate(self.convolutions):
                # reconfigure weight norm
                nn.utils.remove_weight_norm(conv)
                self.convolutions[i] = nn.utils.weight_norm(conv, dim=0)
            state_dict['decoder.version'] = torch.Tensor([1])
        return state_dict

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn

    def _embed_tokens(self, tokens, incremental_state):
        if incremental_state is not None:
            # keep only the last token for incremental forward pass
            tokens = tokens[:, -1:]
        return self.embed_tokens(tokens)

    def _split_encoder_out(self, encoder_out, incremental_state):
        """Split and transpose encoder outputs.

        This is cached when doing incremental inference.
        """
        cached_result = utils.get_incremental_state(self, incremental_state,
                                                    'encoder_out')
        if cached_result is not None:
            return cached_result

        # transpose only once to speed up attention layers
        encoder_a, encoder_b = encoder_out
        encoder_a = encoder_a.transpose(1, 2).contiguous()
        result = (encoder_a, encoder_b)

        if incremental_state is not None:
            utils.set_incremental_state(self, incremental_state, 'encoder_out',
                                        result)
        return result

    def _transpose_if_training(self, x, incremental_state):
        if incremental_state is None:
            x = x.transpose(0, 1)
        return x


def extend_conv_spec(convolutions):
    """
    Extends convolutional spec that is a list of tuples of 2 or 3 parameters
    (kernel size, dim size and optionally how many layers behind to look for residual)
    to default the residual propagation param if it is not specified
    """
    extended = []
    for spec in convolutions:
        if len(spec) == 3:
            extended.append(spec)
        elif len(spec) == 2:
            extended.append(spec + (1, ))
        else:
            raise Exception('invalid number of parameters in convolution spec '
                            + str(spec) + '. expected 2 or 3')
    return tuple(extended)


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, 0, 0.1)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def PositionalEmbedding(num_embeddings, embedding_dim, padding_idx):
    m = LearnedPositionalEmbedding(num_embeddings, embedding_dim, padding_idx)
    nn.init.normal_(m.weight, 0, 0.1)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, dropout=0):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features)
    nn.init.normal_(
        m.weight, mean=0, std=math.sqrt((1 - dropout) / in_features))
    nn.init.constant_(m.bias, 0)
    return nn.utils.weight_norm(m)


def LinearizedConv1d(in_channels,
                     out_channels,
                     kernel_size,
                     dropout=0,
                     **kwargs):
    """Weight-normalized Conv1d layer optimized for decoding"""
    m = LinearizedConvolution(in_channels, out_channels, kernel_size, **kwargs)
    std = math.sqrt((4 * (1.0 - dropout)) / (m.kernel_size[0] * in_channels))
    nn.init.normal_(m.weight, mean=0, std=std)
    nn.init.constant_(m.bias, 0)
    return nn.utils.weight_norm(m, dim=2)


def ConvTBC(in_channels, out_channels, kernel_size, dropout=0, **kwargs):
    """Weight-normalized Conv1d layer"""
    from fairseq.modules import ConvTBC
    m = ConvTBC(in_channels, out_channels, kernel_size, **kwargs)
    std = math.sqrt((4 * (1.0 - dropout)) / (m.kernel_size[0] * in_channels))
    nn.init.normal_(m.weight, mean=0, std=std)
    nn.init.constant_(m.bias, 0)
    return nn.utils.weight_norm(m, dim=2)


@register_model_architecture('fconv_dpp_micro', 'fconv_dpp_micro')
def base_architecture(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_layers = getattr(args, 'encoder_layers', '[(512, 3)] * 20')
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_layers = getattr(args, 'decoder_layers', '[(512, 3)] * 20')
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 256)
    args.decoder_attention = getattr(args, 'decoder_attention', 'True')
    args.share_input_output_embed = getattr(args, 'share_input_output_embed',
                                            False)


@register_model_architecture('fconv_dpp_micro', 'summ_dpp_micro')
def summarization_architecture(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    args.encoder_embed_path = getattr(args, 'encoder_embed_path',
                                      "./data_bin/tldr/model_tldr_256.vec")
    args.encoder_layers = getattr(args, 'encoder_layers', '[(256, 5)] * 20')
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path',
                                      "./data_bin/tldr/model_tldr_256.vec")
    args.decoder_layers = getattr(args, 'decoder_layers', '[(256, 3)] * 4')
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 256)
    args.decoder_attention = getattr(args, 'decoder_attention', 'True')
    args.share_input_output_embed = getattr(args, 'share_input_output_embed',
                                            False)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff',
                                           [5000, 10000, 20000])
