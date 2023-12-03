---
layout: page
title: ArXiv-MDTE Extraction
description: An automated parsing system to extract contextual info from ArXiv.
img: assets/img/proj_4/cover.jpg
importance: 2
category: work
giscus_comments: true
---
I implemented a versatile tool to extract specific contextual information from ArXiv papers as an assistance for my research in MLP group. The [ArXiv dataset](https://huggingface.co/datasets/arxiv_dataset) contains 1.7 million [arXiv](https://arxiv.org/) articles for applications like trend analysis, paper recommender engines, category prediction, co-citation networks, knowledge graph construction and semantic search interfaces. In my research, I want to build a new dataset containing mathematical symbols(tokens) from arXiv papers with corresponding definition, attributes, and original context sentences. To learn more about this resaarch, please refer to our [previous work](https://link.springer.com/chapter/10.1007/978-3-031-16681-5_23) and the [ongoing research](https://mlpgroup.xyz/projects/Token-Classification.html).

Through this extraction tool, we are able to build the dataset, namely MTDE (mathematical token definition extraction). Then we will use the new dataset as a benchmark to evaluate language models' ability to extract and classify mathematical symbols. 

An high-level description of the tool's framework is shown below, please refer to my [github page](https://github.com/jiaruzouu/arXiv-MDTE-key-word-extraction) on how to use the tool if you are interested.
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/proj_4/dataframe.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    code can be found <a href="https://github.com/jiaruzouu/arXiv-MDTE-key-word-extraction" style="text-decoration: underline; color: inherit;">here</a>.
</div>

