---
layout: archive
permalink: design_pattern
title: "Design Pattern"
types: posts
entries_layout: grid

author_profile: true
sidebar:
  nav: "sidebar-category"
---

<div class="grid__wrapper">
  {% assign posts = site.categories['design_pattern'] %}
  {% for post in posts %}
    {% include archive-single.html type="grid" %}
  {% endfor %}
</div>