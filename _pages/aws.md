---
layout: archive
permalink: aws
title: "AWS"
types: posts
entries_layout: grid
author_profile: true
sidebar:
  nav: "sidebar-category"
---

<div class="grid__wrapper">
  {% assign posts = site.categories['aws'] %}
  {% for post in posts %}
    {% include archive-single.html type="grid" %}
  {% endfor %}
</div>