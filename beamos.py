import tensorflow as tf
import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import rnn_cell, rnn, nn_ops, math_ops, embedding_ops, array_ops

linear = rnn_cell._linear

def _extract_argmax_and_embed(embedding, update_embedding=True):
    """Get a loop_function that extracts the previous symbol and embeds it.
    Args:
      embedding: embedding tensor for symbols.
      update_embedding: Boolean; if False, the gradients will not propagate
        through the embeddings.
    Returns:
      A loop function.
    """
    def loop_function(prev, _):
        prev_symbol = math_ops.argmax(prev, 1)
        
        # Note that gradients will not propagate through the second parameter of
        # embedding_lookup.
        emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)
        
        if not update_embedding:
            emb_prev = array_ops.stop_gradient(emb_prev)
        return emb_prev
    return loop_function

def embedding_attention_encoder_seq2seq(enc_inp, cell, num_encoder_symbols, embedding_size):
    with variable_scope.variable_scope("embedding_attention_seq2seq"):
        # Encoder.
        encoder_cell = rnn_cell.EmbeddingWrapper(cell, embedding_classes=num_encoder_symbols, embedding_size=embedding_size)
        encoder_outputs, encoder_state = rnn.rnn(encoder_cell, enc_inp, dtype=dtypes.float32)
                                        
    return encoder_outputs, encoder_state
                                        
def embedding_attention_seq2seq_beam(dec_inp, use_initial, supplied_prev, supplied_state, supplied_attns, cell, num_decoder_symbols, embedding_size, encoder_outputs, encoder_state):
    with variable_scope.variable_scope("embedding_attention_seq2seq"):
        # First calculate a concatenation of encoder outputs to put attention on.
        top_states = [array_ops.reshape(e, [-1, 1, cell.output_size]) for e in encoder_outputs]
        attention_states = array_ops.concat(1, top_states)

        # Decoder.
        cell = rnn_cell.OutputProjectionWrapper(cell, num_decoder_symbols)

        with variable_scope.variable_scope("embedding_attention_decoder"):
            embedding = variable_scope.get_variable("embedding", [num_decoder_symbols, embedding_size])
            loop_function = _extract_argmax_and_embed(embedding)
            emb_inp = [embedding_ops.embedding_lookup(embedding, i) for i in dec_inp]

            decoder_inputs = emb_inp
            initial_state = encoder_state
            output_size = cell.output_size

            # Attention Decoder
            with variable_scope.variable_scope("attention_decoder"):
                batch_size_ = array_ops.shape(decoder_inputs[0])[0]  # Needed for reshaping.
                attn_length = attention_states.get_shape()[1].value
                attn_size = attention_states.get_shape()[2].value

                # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
                hidden = array_ops.reshape(attention_states, [-1, attn_length, 1, attn_size])
                hidden_features = []
                v = []
                attention_vec_size = attn_size  # Size of query vectors for attention.

                for a in xrange(1):
                    k = variable_scope.get_variable("AttnW_%d" % a, [1, 1, attn_size, attention_vec_size])
                    hidden_features.append(nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
                    v.append(variable_scope.get_variable("AttnV_%d" % a, [attention_vec_size]))

                state = tf.cond(use_initial > 0, lambda: initial_state, lambda: supplied_state)

                def attention(query):
                    """Put attention masks on hidden using hidden_features and query."""
                    ds = []  # Results of attention reads will be stored here.

                    for a in xrange(1):
                        with variable_scope.variable_scope("Attention_%d" % a):
                            y = linear(query, attention_vec_size, True)
                            y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])

                            # Attention mask is a softmax of v^T * tanh(...).
                            s = math_ops.reduce_sum(v[a] * math_ops.tanh(hidden_features[a] + y), [2, 3])
                            a = nn_ops.softmax(s)

                            # Now calculate the attention-weighted vector d.
                            d = math_ops.reduce_sum(array_ops.reshape(a, [-1, attn_length, 1, 1]) * hidden, [1, 2])
                            ds.append(array_ops.reshape(d, [-1, attn_size]))

                    return ds

                outputs = []
                prev = None
                batch_attn_size = array_ops.pack([batch_size_, attn_size])
                attns = [tf.cond(use_initial > 0, lambda: array_ops.zeros(batch_attn_size, dtype=dtypes.float32), lambda: supplied_attns) for _ in xrange(1)]

                for a in attns:  # Ensure the second shape of attention vectors is set.
                    a.set_shape([None, attn_size])

                with variable_scope.variable_scope("loop_function", reuse=True):
                    #inp = tf.cond(use_initial > 0, lambda: decoder_inputs[0], lambda: loop_function(supplied_prev, 0))
                    inp = decoder_inputs[0]

                input_size = inp.get_shape().with_rank(2)[1]

                if input_size.value is None:
                    raise ValueError("Could not infer input size from input: %s" % inp.name)

                x = linear([inp] + attns, input_size, True)

                # Run the RNN.
                cell_output, state = cell(x, state)

                # Run the attention mechanism.
                attns = attention(state)

                with variable_scope.variable_scope("AttnOutputProjection"):
                    output = linear([cell_output] + attns, output_size, True)

            return output, state, attns[0]