Use cases for probelib:

- Be able to define datasets and get tokenized versions of the data with good mask support.
  - Easy and ergonomic to add new datasets.
  - Easy to add new masks.
  - Easy to visualize masks.
- Training multiple probes on a model's layers. Probes by default should be agnostic to the model architecture and layers. All probes should take in inputs of the same type (i.e. a global Activations object). The library by default should select the activation collection plan that will take in the least amount of memory and is most efficient (this means pooling the activations and filtering for detection/attention tokens as soon as possible).
  - Be able to train multiple probes at the same time on different layers without having to recompute the activations.
  - Library should plan ahead what activations to collect (i.e. if there's two probes, one requires pre-pooling with mean and one with max, the library should pre-pool with mean and max and store the pooled ones, instead of storing the full activations and then pooling inside the probe).
  - The plan should happen inside train_probes. By default the library should do the expected thing, ie collect_activations returns all activations in a dense format, and training a probe via .fit() works.
- Using a probe to generate signal for model finetuning (i.e. a backpropable signal that can be used to finetune the model).
  - Probes should just have a .predict() method that returns a tensor of predictions. The .predict() op should be differentiable.

It's important to focus on:

- simplicity and not surprising: ie a researcher using the high level api should just understand what it does
- efficiency: ie the library should be efficient and use as little memory as possible, optimizing wherever possible when it can assume things (i.e. inside train_probes, where all i/o is known)
