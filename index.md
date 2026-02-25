---
layout: default
title: Hello
---

**Hi, I'm Pablo!**
I’m a PhD student in Artificial Intelligence supervised by [Alejandro Martín](https://scholar.google.es/citations?user=b3J9VRsAAAAJ&hl=en) and [Javier Huertas-Tato](https://scholar.google.es/citations?user=5XOhXooAAAAJ&hl=es).  Before this, I completed an MSc in Machine Learning and Big Data at the Technical University of Madrid, and a double BSc in Mathematics and Computer Science at the University of Murcia.

My research focuses on applying deep learning to natural language processing (NLP), with work spanning AI-generated text detection, authorship attribution, natural language inference (NLI), and efficient Transformer architectures. The overarching theme is the study of inductive biases in models, the capabilities adquired from pre-training, and how to apply or post-train these models for downstream tasks in a way that *generalizes robustly* especially when data are scarce or affected by spurious correlations.

Feel free to check out my publications, occasional posts, or reach out if you’re interested in collaborating!

<ul class="social-list">
	{% for item in site.social %}
		<li>
			<span class="label">{{ item.name }}:</span>
			{% if item.url %}
				<a href="{{ item.url }}" target="_blank" rel="noopener noreferrer">{{ item.text }}</a>
			{% else %}
				<span class="value">{{ item.text }}</span>
			{% endif %}
		</li>
	{% endfor %}
</ul>