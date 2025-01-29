---
layout: archive
permalink: machine_learning
title: "Machine Learning"
types: posts
entries_layout: grid

author_profile: true
sidebar:
  nav: "sidebar-category"
---

<div class="grid__wrapper">
  {% assign posts = site.categories['machine_learning'] %}
  {% for post in posts %}
    {% include archive-single.html type="grid" %}
  {% endfor %}
</div>