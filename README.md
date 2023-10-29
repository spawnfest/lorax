# Lorax

This package implements Low-Rank Adaptation (LoRA), a popular method for fine-tuning large language models.
LoRA introduces a more efficient method for adaptation by freezing the original model weights and injecting trainable rank decomposition matrices. This results in a dramatic reduction in the number of parameters needing updates (by up to 10,000x for some configurations) and cuts down on the GPU memory needed, making it a much more affordable and practical solution for users who don’t have access to high-end GPU hardware.

## How To Fine-tune an LLM

1. Import your model
2. Freeze your model
3. Inject trainable LoRA parameters

```
{:ok, model} = Bumblebee.load_model({:hf, "gpt2"})

lora_model =
  model
  |> Axon.freeze()
  |> Lorax.foo(%Lorax.Config{r: 2, alpha: 4, lora_dropout: 0.05})

# train model
```

## Default Settings

The default configs target only the query and value matrices.
r is set to 1, alpha to 2.

The original LoRA paper found that configuring query and value matrices was effective enough for fine-tuning. Even a r value of 1 is enough to fine-tune a model, though in practice I found that it's necessary to use values of 2, 4, or 8.   

## Limitations

While the LoRA algorithm significantly reduces the GPU requirements for fine-tuning a model, using LoRA on LLMs that are bigger than GPT2 still requires a GPU with high vRAM. Most of the examples here were fine-tuned on an A10G on Huggingface Spaces. Attempting to fine-tune Mistral 7B on Huggingface's A10x4 (the largest available w/ 96 vRAM) will cause cuda OOM errors. To fine-tune on consumer GPUs, quantization work needs to be done to implement the QLoRA algorithm.

## Installation

If [available in Hex](https://hex.pm/docs/publish), the package can be installed
by adding `lorax` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:lorax, "~> 0.1.0"}
    # or
    {:lorax, git: "https://github.com/spawnfest/lorax.git"},
  ]
end
```

Documentation can be generated with [ExDoc](https://github.com/elixir-lang/ex_doc)
and published on [HexDocs](https://hexdocs.pm). Once published, the docs can
be found at <https://hexdocs.pm/lorax>.

