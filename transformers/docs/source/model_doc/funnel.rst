Funnel Transformer
------------------

Overview
~~~~~~~~~~~~~~~~~~~~~

The Funnel Transformer model was proposed in the paper
`Funnel-Transformer: Filtering out Sequential Redundancy for Efficient Language Processing
<https://arxiv.org/abs/2006.03236>`__.
It is a bidirectional transformer model, like BERT, but with a pooling operation after each block of layers, a bit
like in traditional convolutional neural networks (CNN) in computer vision.

The abstract from the paper is the following:

*With the success of language pretraining, it is highly desirable to develop more efficient architectures of good
scalability that can exploit the abundant unlabeled data at a lower cost. To improve the efficiency, we examine the
much-overlooked redundancy in maintaining a full-length token-level presentation, especially for tasks that only
require a single-vector presentation of the sequence. With this intuition, we propose Funnel-Transformer which
gradually compresses the sequence of hidden states to a shorter one and hence reduces the computation cost. More
importantly, by re-investing the saved FLOPs from length reduction in constructing a deeper or wider model, we further
improve the model capacity. In addition, to perform token-level predictions as required by common pretraining
objectives, Funnel-Transformer is able to recover a deep representation for each token from the reduced hidden sequence
via a decoder. Empirically, with comparable or fewer FLOPs, Funnel-Transformer outperforms the standard Transformer on
a wide variety of sequence-level prediction tasks, including text classification, language understanding, and reading
comprehension.*

Tips:

- Since Funnel Transformer uses pooling, the sequence length of the hidden states changes after each block of layers.
  The base model therefore has a final sequence length that is a quarter of the original one. This model can be used
  directly for tasks that just require a sentence summary (like sequence classification or multiple choice). For other
  tasks, the full model is used; this full model has a decoder that upsamples the final hidden states to the same
  sequence length as the input.
- The Funnel Transformer checkpoints are all available with a full version and a base version. The first ones should
  be used for :class:`~transformers.FunnelModel`, :class:`~transformers.FunnelForPreTraining`,
  :class:`~transformers.FunnelForMaskedLM`, :class:`~transformers.FunnelForTokenClassification` and
  class:`~transformers.FunnelForQuestionAnswering`. The second ones should be used for
  :class:`~transformers.FunnelBaseModel`, :class:`~transformers.FunnelForSequenceClassification` and
  :class:`~transformers.FunnelForMultipleChoice`.

The original code can be found `here <https://github.com/laiguokun/Funnel-Transformer>`_.


FunnelConfig
~~~~~~~~~~~~

.. autoclass:: transformers.FunnelConfig
    :members:


FunnelTokenizer
~~~~~~~~~~~~~~~

.. autoclass:: transformers.FunnelTokenizer
    :members: build_inputs_with_special_tokens, get_special_tokens_mask,
        create_token_type_ids_from_sequences, save_vocabulary


FunnelTokenizerFast
~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.FunnelTokenizerFast
    :members:


Funnel specific outputs
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.modeling_funnel.FunnelForPreTrainingOutput
    :members:


FunnelBaseModel
~~~~~~~~~~~~~~~

.. autoclass:: transformers.FunnelBaseModel
    :members:


FunnelModel
~~~~~~~~~~~

.. autoclass:: transformers.FunnelModel
    :members:


FunnelModelForPreTraining
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.FunnelForPreTraining
    :members:


FunnelForMaskedLM
~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.FunnelForMaskedLM
    :members:


FunnelForSequenceClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.FunnelForSequenceClassification
    :members:


FunnelForMultipleChoice
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.FunnelForMultipleChoice
    :members:


FunnelForTokenClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.FunnelForTokenClassification
    :members:


FunnelForQuestionAnswering
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.FunnelForQuestionAnswering
    :members:
