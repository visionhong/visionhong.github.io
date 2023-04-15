---
layout: archive
permalink: design_pattern
title: "Design Pattern"
types: posts

author_profile: true
sidebar:
  nav: "sidebar-category"
---

{% assign posts = site.categories['design_pattern']%}
{% for post in posts %}
  {% include archive-single.html type=page.entries_layout %}
{% endfor %}