{
  "dataset_reader": {
    "type": "target_sentiment",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      }
    }
  },
  "model": {
    "type": "atae_classifier",
    "dropout": 0.5,
    "regularizer": [[".*", {"type": "l2", "alpha": 0.0001}]],
    "AE": true,
    "AttentionAE": false,
    "context_field_embedder": {
      "tokens": {
        "type": "embedding",
        "embedding_dim": 300,
        "pretrained_file": "./resources/word_embeddings/glove.840B.300d.txt",
        "trainable": false
      }
    },
    "context_encoder": {
      "type": "lstm",
      "input_size": 600,
      "hidden_size": 300,
      "bidirectional": false,
      "num_layers": 1
    },
    "target_encoder": {
        "type": "boe",
        "embedding_dim": 300,
        "averaged": true
    },
    "inter_target_encoding": {
      "type": "gru",
      "input_size": 300,
      "hidden_size": 300,
      "bidirectional": false,
      "num_layers": 1
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