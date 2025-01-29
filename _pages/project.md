---
# layout: archive
layout: collection
entries_layout: grid
permalink: project
title: "project"
types: posts
entries_layout: grid

author_profile: true
sidebar:
  nav: "sidebar-category"
---

<div class="grid__wrapper">
  {% assign posts = site.categories['project'] %}
  {% for post in posts %}
    {% include archive-single.html type="grid" %}
  {% endfor %}
</div>