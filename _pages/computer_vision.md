---
layout: archive
permalink: computer_vision
title: "Computer Vision"
types: posts

author_profile: true
sidebar:
  nav: "sidebar-category"
---

{% assign posts = site.categories['computer_vision']%}
{% for post in posts %}
  {% include archive-single.html type=page.entries_layout %}
{% endfor %}