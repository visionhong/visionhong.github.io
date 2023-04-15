---
layout: archive
permalink: deep_learning
title: "Deep Learning"
types: posts

author_profile: true
sidebar:
  nav: "sidebar-category"
---

{% assign posts = site.categories['deep_learning']%}
{% for post in posts %}
  {% include archive-single.html type=page.entries_layout %}
{% endfor %}