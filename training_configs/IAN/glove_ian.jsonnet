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
    "type": "interactive_attention_network_classifier",
    "dropout": 0.5,
    "regularizer": [[".*", {"type": "l2", "alpha": 0.0001}]],
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
      "input_size": 300,
      "hidden_size": 300,
      "bidirectional": false,
      "num_layers": 1
    },
    "target_encoder": {
      "type": "lstm",
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