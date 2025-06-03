Given sentences in citation paper:
```
"prev": "Quantization and Losses:\n\nTo learn a discrete latent space, we quantize our encoder outputs into the entries of a learned codebook via the vector quantization (VQ) objective in VQVAEs [45],\nL VQ = sg(z) − e 2 2 + β z − sg(e) 2 2 ,(1)\nwhere sg(x) ≡ x, and d dx sg(x) ≡ 0 is the stop-gradient operator, β is the commitment loss weight, and e is a codebook vector from codebook E. The index to the codebook vector closest to z is found by i = argmin j z − E j 2 2 .",
"curr": "In addition to the VQ objective, we adopt the factorized and 2normalized codes from ViT-VQGAN [58] to improve codebook usage and reconstruction quality.",
"next": "To train our model, we use a combination of L 2 loss, image perceptual loss L IP [20,61], video perceptual loss L VP by using the I3D network [6] as feature extractor, and adversarial loss L Adv with StyleGAN architecture [21]."
```

# Trials

## 1
prompt:
```
"Below is a passage from a research paper.\n"
f"Previous sentence: {prev}\n"
f"Current sentence: {curr}\n"
f"Next sentence: {next_}\n\n"
"Generate a natural language query suitable for a paper search engine, "
"so that the cited paper in the current sentence (i.e., the paper linked by the citation paper id in curr) can be found. "
"Reflect the context and citation intent of curr as much as possible, and include the paper title, key concepts, methods, techniques, model names, or main claims if relevant. "
"Output only the search query itself, without any unnecessary prefixes or explanations."

>>> ViT-VQGAN factorized L2-normalized codebook reconstruction quality codebook usage improvement
```

Query is too specific.

## 2
prompt:
```
"Below are three consecutive sentences from a research paper.\n"
f"Previous sentence: {prev}\n"
f"Current sentence: {curr}\n"
f"Next sentence: {next_}\n\n"
"Imagine you are a researcher who, based on the overall context of these sentences, wants to find related papers or prior work, "
"but you do not have access to the exact citation or a specific sentence. "
"Generate a broad, abstract, and exploratory natural language query that could be used in a paper search engine to discover relevant literature. "
"Do not copy or closely paraphrase the current sentence; instead, focus on the general research topic, methods, or concepts implied by the context. "
"Output only the search query itself, without any unnecessary prefixes or explanations."

>>> discrete latent representations with vector quantization codebook optimization hybrid loss functions perceptual and adversarial training image video reconstruction generative neural networks
```

It doesn't match the LitSearch query data.

## 3

I think it's necessary to add a few-shot example to generate queries as I want.

There is no citation sentence data corresponding to the query. Therefore, I generated fake sentences and query data using GPT-4.1.

prompt:
```
"Below are three consecutive sentences from a research paper.\n"
"Given these, generate a natural language query in the style of a researcher asking, for example, 'Are there any research papers on ...', 'Are there any studies that ...', or 'Are there any tools or resources for ...'.\n"
"The query should be broad, abstract, and exploratory, aiming to discover relevant literature based on the overall context, not just the current sentence.\n"
"Do not copy or closely paraphrase the current sentence; instead, focus on the general research topic, methods, or concepts implied by the context.\n"
"Output only the search query itself, without any unnecessary prefixes or explanations.\n\n"
"Example 1:\n"
"Previous sentence: Deep generative models have shown remarkable success in image synthesis tasks.\n"
"Current sentence: Recent advances leverage adversarial training to improve sample quality.\n"
"Next sentence: However, training instability remains a significant challenge.\n"
"Query: Are there any research papers on generative models for image synthesis and adversarial training methods?\n\n"
"Example 2:\n"
"Previous sentence: Temporal consistency is crucial for video generation.\n"
"Current sentence: Several approaches use optical flow to enforce smooth transitions between frames.\n"
"Next sentence: Despite these efforts, artifacts still occur in challenging scenarios.\n"
"Query: Are there any studies that explore methods for improving temporal consistency in video generation using optical flow?\n\n"
"Example 3:\n"
"Previous sentence: Self-supervised learning has gained popularity in representation learning.\n"
"Current sentence: Contrastive loss functions are commonly used to train such models.\n"
"Next sentence: These methods have been applied to various domains including vision and language.\n"
"Query: Are there any research papers on self-supervised representation learning with contrastive loss in vision and language?\n\n"
"Now, generate a query for the following context:\n"
f"Previous sentence: {prev}\n"
f"Current sentence: {curr}\n"
f"Next sentence: {next_}\n"
"Query:"

>>> Are there any research papers on generative models that combine quantized latent spaces with perceptual and adversarial loss functions to enhance codebook utilization and reconstruction quality?
```

It achieves a recaal@20 score of 0.21 with BM25. This is much lower than the original inline-citation rewriting dataset in LitSearch, which achieves 0.3999. We need to generate more specific queries.


## 4

```
"Below are three consecutive sentences from a research paper.\n"
"Given these, generate a natural language query in the style of a researcher asking, for example, 'Are there any research papers on ...', 'Are there any studies that ...', or 'Are there any tools or resources for ...'.\n"
"The query should reflect the main research topic, method, or concept implied by the context, focusing on what the cited paper is likely about.\n"
"Do not copy or closely paraphrase the current sentence; instead, use the overall context to infer the research topic or method.\n"
"The query should be specific enough to retrieve relevant papers, but not limited to the exact wording of the current sentence.\n"
"Output only the search query itself, without any unnecessary prefixes or explanations.\n\n"
"Example 1:\n"
"Previous sentence: Deep generative models have shown remarkable success in image synthesis tasks.\n"
"Current sentence: Recent advances leverage adversarial training to improve sample quality.\n"
"Next sentence: However, training instability remains a significant challenge.\n"
"Query: Are there any research papers on generative models for image synthesis and adversarial training methods?\n\n"
"Example 2:\n"
"Previous sentence: Temporal consistency is crucial for video generation.\n"
"Current sentence: Several approaches use optical flow to enforce smooth transitions between frames.\n"
"Next sentence: Despite these efforts, artifacts still occur in challenging scenarios.\n"
"Query: Are there any studies that explore methods for improving temporal consistency in video generation using optical flow?\n\n"
"Example 3:\n"
"Previous sentence: Self-supervised learning has gained popularity in representation learning.\n"
"Current sentence: Contrastive loss functions are commonly used to train such models.\n"
"Next sentence: These methods have been applied to various domains including vision and language.\n"
"Query: Are there any research papers on self-supervised representation learning with contrastive loss in vision and language?\n\n"
"Now, generate a query for the following context:\n"
f"Previous sentence: {prev}\n"
f"Current sentence: {curr}\n"
f"Next sentence: {next_}\n"
"Query:"

>>> Are there any research papers on generative models employing vector quantization with codebook optimization and multi-loss training strategies for improved reconstruction quality?
```

There are slightly more detailed words than in the previous one, but it still doesn't seem to be enough.

As expected, the recall@20 score is 0.2229, which is slightly higher than the previous one, but still not enough.

We need to use the information from the title and abstract of the paper.

## 5

prompt:
```
"Below are three consecutive sentences from a research paper, along with the paper's title and abstract.\n"
"Given these, generate a natural language query in the style of a researcher asking, for example, 'Are there any research papers on ...', 'Are there any studies that ...', or 'Are there any tools or resources for ...'.\n"
"The query should reflect the main research topic, method, or concept implied by the context, focusing on what the cited paper is likely about.\n"
"Do not copy or closely paraphrase the current sentence; instead, use the overall context to infer the research topic or method.\n"
"The query should be specific enough to retrieve relevant papers, but not limited to the exact wording of the current sentence.\n"
"Output only the search query itself, without any unnecessary prefixes or explanations.\n\n"
"Example 1:\n"
"Previous sentence: Deep generative models have shown remarkable success in image synthesis tasks.\n"
"Current sentence: Recent advances leverage adversarial training to improve sample quality.\n"
"Next sentence: However, training instability remains a significant challenge.\n"
"Title: Generative Adversarial Networks for Image Synthesis\n"
"Abstract: This paper explores the use of generative adversarial networks for high-quality image synthesis, discussing recent advances and challenges in training stability.\n"
"Query: Are there any research papers on generative models for image synthesis and adversarial training methods?\n\n"
"Example 2:\n"
"Previous sentence: Temporal consistency is crucial for video generation.\n"
"Current sentence: Several approaches use optical flow to enforce smooth transitions between frames.\n"
"Next sentence: Despite these efforts, artifacts still occur in challenging scenarios.\n"
"Title: Improving Temporal Consistency in Video Generation\n"
"Abstract: We propose new methods for enforcing temporal consistency in video generation, focusing on the use of optical flow and addressing common artifacts.\n"
"Query: Are there any studies that explore methods for improving temporal consistency in video generation using optical flow?\n\n"
"Example 3:\n"
"Previous sentence: Self-supervised learning has gained popularity in representation learning.\n"
"Current sentence: Contrastive loss functions are commonly used to train such models.\n"
"Next sentence: These methods have been applied to various domains including vision and language.\n"
"Title: Self-Supervised Representation Learning with Contrastive Loss\n"
"Abstract: This work surveys self-supervised learning techniques using contrastive loss, with applications in vision and language domains.\n"
"Query: Are there any research papers on self-supervised representation learning with contrastive loss in vision and language?\n\n"
"Now, generate a query for the following context:\n"
f"Previous sentence: {prev}\n"
f"Current sentence: {curr}\n"
f"Next sentence: {next_}\n"
f"Title: {title}\n"
f"Abstract: {abstract}\n"
"Query:"s

>>> Are there any research papers on off-policy maximum entropy deep reinforcement learning algorithms with a stochastic actor for continuous control tasks?
```

Prompt includes title and abstract. However, the queries seems to be too specific.

The recall@20 score is 0.83 with BM25. Each query is too specific to find the cited paper. (We should aim to achieve a score of around 0.40.)

## 6

removing the abstract from the prompts.

prompt:
```
"Below are three consecutive sentences from a research paper, along with the paper's title and abstract.\n"
"Given these, generate a natural language query in the style of a researcher asking, for example, 'Are there any research papers on ...', 'Are there any studies that ...', or 'Are there any tools or resources for ...'.\n"
"The query should reflect the main research topic, method, or concept implied by the context, focusing on what the cited paper is likely about.\n"
"Do not copy or closely paraphrase the current sentence; instead, use the overall context to infer the research topic or method.\n"
"The query should be specific enough to retrieve relevant papers, but not limited to the exact wording of the current sentence.\n"
"Output only the search query itself, without any unnecessary prefixes or explanations.\n\n"
"Example 1:\n"
"Previous sentence: Deep generative models have shown remarkable success in image synthesis tasks.\n"
"Current sentence: Recent advances leverage adversarial training to improve sample quality.\n"
"Next sentence: However, training instability remains a significant challenge.\n"
"Title: Generative Adversarial Networks for Image Synthesis\n"
"Abstract: This paper explores the use of generative adversarial networks for high-quality image synthesis, discussing recent advances and challenges in training stability.\n"
"Query: Are there any research papers on generative models for image synthesis and adversarial training methods?\n\n"
"Example 2:\n"
"Previous sentence: Temporal consistency is crucial for video generation.\n"
"Current sentence: Several approaches use optical flow to enforce smooth transitions between frames.\n"
"Next sentence: Despite these efforts, artifacts still occur in challenging scenarios.\n"
"Title: Improving Temporal Consistency in Video Generation\n"
"Abstract: We propose new methods for enforcing temporal consistency in video generation, focusing on the use of optical flow and addressing common artifacts.\n"
"Query: Are there any studies that explore methods for improving temporal consistency in video generation using optical flow?\n\n"
"Example 3:\n"
"Previous sentence: Self-supervised learning has gained popularity in representation learning.\n"
"Current sentence: Contrastive loss functions are commonly used to train such models.\n"
"Next sentence: These methods have been applied to various domains including vision and language.\n"
"Title: Self-Supervised Representation Learning with Contrastive Loss\n"
"Abstract: This work surveys self-supervised learning techniques using contrastive loss, with applications in vision and language domains.\n"
"Query: Are there any research papers on self-supervised representation learning with contrastive loss in vision and language?\n\n"
"Now, generate a query for the following context:\n"
f"Previous sentence: {prev}\n"
f"Current sentence: {curr}\n"
f"Next sentence: {next_}\n"
f"Title: {title}\n"
"Query:"

>>> Are there any research papers on improved vector-quantized image modeling techniques using factorized codebooks, combined perceptual and adversarial losses, and enhanced VQGAN architectures?
```

The recall@20 score is 0.61 with BM25.

I'm going to revise the prompt slightly to focus on the general context.


## 7

prompt:
```
"The query should be broad, abstract, and exploratory, aiming to discover relevant literature based on the overall context, not just the current sentence.\n"
"Do not copy or closely paraphrase the current sentence; instead, focus on the general research topic, methods, or concepts implied by the context.\n"
"Output only the search query itself, without any unnecessary prefixes or explanations.\n\n"
"Example 1:\n"
"Previous sentence: Deep generative models have shown remarkable success in image synthesis tasks.\n"
"Current sentence: Recent advances leverage adversarial training to improve sample quality.\n"
"Next sentence: However, training instability remains a significant challenge.\n"
"Title: Generative Adversarial Networks for Image Synthesis\n"
"Abstract: This paper explores the use of generative adversarial networks for high-quality image synthesis, discussing recent advances and challenges in training stability.\n"
"Query: Are there any research papers on generative models for image synthesis and adversarial training methods?\n\n"
"Example 2:\n"
"Previous sentence: Temporal consistency is crucial for video generation.\n"
"Current sentence: Several approaches use optical flow to enforce smooth transitions between frames.\n"
"Next sentence: Despite these efforts, artifacts still occur in challenging scenarios.\n"
"Title: Improving Temporal Consistency in Video Generation\n"
"Abstract: We propose new methods for enforcing temporal consistency in video generation, focusing on the use of optical flow and addressing common artifacts.\n"
"Query: Are there any studies that explore methods for improving temporal consistency in video generation using optical flow?\n\n"
"Example 3:\n"
"Previous sentence: Self-supervised learning has gained popularity in representation learning.\n"
"Current sentence: Contrastive loss functions are commonly used to train such models.\n"
"Next sentence: These methods have been applied to various domains including vision and language.\n"
"Title: Self-Supervised Representation Learning with Contrastive Loss\n"
"Abstract: This work surveys self-supervised learning techniques using contrastive loss, with applications in vision and language domains.\n"
"Query: Are there any research papers on self-supervised representation learning with contrastive loss in vision and language?\n\n"
"Now, generate a query for the following context:\n"
f"Previous sentence: {prev}\n"
f"Current sentence: {curr}\n"
f"Next sentence: {next_}\n"
f"Title: {title}\n"
"Query:"

>>> Are there any research papers on vector-quantized image modeling with enhanced VQGAN architectures and combined perceptual and adversarial loss strategies?
```

recall@20 = 0.48

Ooh! It's the score what I want. :D
