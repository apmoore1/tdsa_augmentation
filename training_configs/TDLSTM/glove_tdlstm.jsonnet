{
    "dataset_reader": {
      "type": "target_sentiment",
      "token_indexers": {
        "tokens": {
          "type": "single_id",
          "lowercase_tokens": true,
          "token_min_padding_length": 1
        }
      },
      "incl_target": true,
      "left_right_contexts": true,
      "reverse_right_context": true
    },
    "model": {
      "type": "split_contexts_classifier",
      "dropout": 0.5,
      "regularizer": [[".*", {"type": "l2", "alpha": 0.0001}]],
      "text_field_embedder": {
        "tokens": {
          "type": "embedding",
          "pretrained_file": "./resources/word_embeddings/glove.840B.300d.txt",
          "embedding_dim": 300,
          "trainable": false
        }
      },
      "left_text_encoder": {
        "type": "lstm",
        "input_size": 600,
        "hidden_size": 300,
        "bidirectional": false,
        "num_layers": 1
      },
      "right_text_encoder": {
        "type": "lstm",
        "input_size": 600,
        "hidden_size": 300,
        "bidirectional": false,
        "num_layers": 1
      },
      "target_encoder": {
        "type": "boe",
        "embedding_dim": 300
      }
    },
    "iterator": {
      "type": "basic",
      "batch_size": 32
    },
    "trainer": {
      "optimizer": {
        "type": "adam"
      },
      "shuffle": true,
      "patience": 10,
      "num_epochs": 100,
      "cuda_device": 0,
      "validation_metric": "+accuracy"
    }
  }