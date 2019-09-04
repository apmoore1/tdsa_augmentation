{
  "dataset_reader": {
    "type": "target_sentiment",
    "token_indexers": {
      "elmo": {
            "type": "elmo_characters"
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
      "elmo": {
            "type": "bidirectional_lm_token_embedder",
            "archive_file": "./resources/language_models/election_model.tar.gz",
            "bos_eos_tokens": ["<S>", "</S>"],
            "remove_bos_eos": true,
            "requires_grad": false
        }
    },
    "context_encoder": {
      "type": "lstm",
      "input_size": 2048,
      "hidden_size": 300,
      "bidirectional": false,
      "num_layers": 1
    },
    "target_encoder": {
        "type": "boe",
        "embedding_dim": 1024,
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